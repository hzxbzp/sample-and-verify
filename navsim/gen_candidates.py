#!/usr/bin/env python3
"""Greedy plan + K samples of the text-waypoint NAVSIM planners with vLLM (offline), from prep_prompts.py prompts.

  --kind curious   Curious-VLA (Qwen2.5-VL-3B; SFT or RL checkpoint). Served officially by LLaMA-Factory (template
         qwen2_vl, vLLM backend, --image_max_pixels 262144) and queried by its OpenAI client with the image part first
         and "<image>" removed from the text; we build the same prompt string and resize the (agent-re-encoded) front
         image the way LLaMA-Factory does before handing it to vLLM. Output parsed with the agent's yaw parser and
         de-normalised with its training statistics; on a parse failure the agent falls back to constant velocity.
  --kind recog     ReCogDrive-VLM-8B (InternVL3-8B), officially run by lmdeploy (template
         internvl2_5 with the agent's meta instruction, max_new_tokens 512, min_new_tokens 50); InternVL dynamic tiling
         with the model's max_dynamic_patch; agent regex parser, zeros on failure.

Official decoding of both = temperature 0 (greedy). Samples: temperature T, top_p 1, no top_k. For every candidate
we keep the summed log-probability of the trajectory tokens (from the trajectory field to the end).

    python navsim/gen_candidates.py --kind recog --model-path CKPT --prompts prompts/test.jsonl --json-dir json/test \
        --out runs/test_recogdrive/cands_r0.jsonl --rank 0 --world 4 --k 16
    python navsim/gen_candidates.py --kind curious --model-path CKPT --curious-stats trajectory_stats_train.json ...

``--curious-stats`` is the trajectory normalisation file of the Curious-VLA release (stats/trajectory_stats_train.json).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import time
import zlib

import numpy as np
from PIL import Image

KINDS = {"curious": dict(max_tokens=4096, min_tokens=0), "recog": dict(max_tokens=512, min_tokens=50)}
MEANS = STDS = None                                               # Curious-VLA de-normalisation (--curious-stats)


# ----------------------------------------------------------------------------- prompts and images
def curious_chat(system: str, user: str) -> str:
    text = user.replace("<image>", "").strip()                 # convert_to_openai_format: [image_url, text]
    return ("<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>" + text + "<|im_end|>\n<|im_start|>assistant\n")


def recog_chat(system: str, user: str) -> str:                  # lmdeploy internvl2_5 template
    return "<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"


def lf_image(path: str, max_pixels: int = 262144) -> Image.Image:
    """LLaMA-Factory API: decode the base64 JPEG to RGB, then mm_plugin._preprocess_image (area cap)."""
    image = Image.open(path).convert("RGB")
    if image.width * image.height > max_pixels:
        f = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize((int(image.width * f), int(image.height * f)))
    return image if image.mode == "RGB" else image.convert("RGB")


# ----------------------------------------------------------------------------- parsers (official)
def parse_curious(text: str):
    m = re.findall(r'[\[\(]\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*[\]\)]', text)
    if m and len(m) >= 8:
        return np.asarray([(float(x), float(y), float(h)) for x, y, h in m[:8]]) * STDS + MEANS
    return None


def parse_recog(text: str):
    full = re.search(r'\[PT(?:, )?((?:\([-+]?\d*\.\d+, [-+]?\d*\.\d+, [-+]?\d*\.\d+\)(?:, )?){8})\]', text)
    if full:
        c = re.findall(r'\(([-+]?\d*\.\d+), ([-+]?\d*\.\d+), ([-+]?\d*\.\d+)\)', full.group(1))
        if len(c) == 8:
            return np.asarray([tuple(map(float, x)) for x in c])
    return None


def span_logprob(out, kind: str):
    """Summed log-prob of the tokens from the trajectory field to the end, and their count."""
    text = out.text
    key = '"future_trajectory"' if kind == "curious" else "[PT"
    start = text.rfind(key)
    if start < 0 or not out.logprobs:
        return [float(out.cumulative_logprob or 0.0), len(out.token_ids)]
    pos, s, n = 0, 0.0, 0
    for tid, lp in zip(out.token_ids, out.logprobs):
        d = lp.get(tid)
        piece = d.decoded_token if d is not None and d.decoded_token is not None else ""
        if pos + len(piece) > start:
            s += float(d.logprob) if d is not None else 0.0
            n += 1
        pos += len(piece)
    return [s, n]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=sorted(KINDS), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--name", default=None, help="model name stored with the candidates (default: the kind)")
    ap.add_argument("--curious-stats", default=None, help="Curious-VLA trajectory_stats_train.json")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--json-dir", required=True, help="scene JSONs of run_preprocessing.py (ego state, logged future)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args(argv)
    from vllm import LLM, SamplingParams
    global MEANS, STDS
    cfg = KINDS[args.kind]
    name = args.name or args.kind
    if args.kind == "curious":
        stats = json.load(open(args.curious_stats))
        MEANS, STDS = np.asarray(stats["mean"], float), np.asarray(stats["std"], float)
    rows = [json.loads(l) for l in open(args.prompts)]
    rows = [r for r in rows if "error" not in r][args.rank::args.world]
    if args.limit:
        rows = rows[: args.limit]
    done = set()                                                  # tokens done by ANY shard of this run (resharding-safe)
    for path in glob.glob(os.path.join(os.path.dirname(os.path.abspath(args.out)), "cands_r*.jsonl")):
        for l in open(path):
            try:
                done.add(json.loads(l)["token"])
            except Exception:
                pass
    rows = [r for r in rows if r["token"] not in done]
    print(f"[vllm] {name} rank {args.rank}/{args.world}: {len(rows)} to do ({len(done)} done)", flush=True)
    llm = LLM(model=args.model_path, trust_remote_code=True, dtype="bfloat16", max_model_len=8192, seed=0,
              limit_mm_per_prompt={"image": 1, "video": 0}, gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=True)
    kind = args.kind
    greedy_sp = SamplingParams(temperature=0.0, max_tokens=cfg["max_tokens"], min_tokens=cfg["min_tokens"], logprobs=1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fout = open(args.out, "a")
    t0 = time.time()
    for c0 in range(0, len(rows), args.chunk):
        chunk = rows[c0:c0 + args.chunk]
        reqs = []
        for r in chunk:
            if kind == "curious":
                reqs.append({"prompt": curious_chat(r["curious_system"], r["curious_user"]),
                             "multi_modal_data": {"image": lf_image(r["img_reencoded"])}})
            else:
                reqs.append({"prompt": recog_chat(r["recog_system"], r["recog_user"]),
                             "multi_modal_data": {"image": Image.open(r["img_orig"]).convert("RGB")}})
        g_out = llm.generate(reqs, greedy_sp, use_tqdm=False)
        s_sps = [SamplingParams(n=args.k, temperature=args.temperature, top_p=1.0, top_k=-1,
                                max_tokens=cfg["max_tokens"], min_tokens=cfg["min_tokens"], logprobs=1,
                                seed=zlib.crc32(r["token"].encode()) & 0x7FFFFFFF) for r in chunk]
        s_out = llm.generate(reqs, s_sps, use_tqdm=False)
        parse = parse_curious if kind == "curious" else parse_recog
        for r, go, so in zip(chunk, g_out, s_out):
            gtxt = go.outputs[0].text
            gp = parse(gtxt)
            if gp is not None:
                official = gp
            elif kind == "curious":                                           # agent fallback: constant velocity
                v = r["ego_velocity"][0]
                official = np.asarray([[v * 0.5 * (i + 1), 0.0, 0.0] for i in range(8)])
            else:                                                             # agent fallback: zeros
                official = np.zeros((8, 3))
            samples, lps = [], []
            for o in so.outputs:
                p = parse(o.text)
                samples.append(np.round(p, 3).tolist() if p is not None else None)
                lps.append(span_logprob(o, kind))
            rec = {"token": r["token"], "model": name, "official": np.round(official, 3).tolist(),
                   "greedy": np.round(gp, 3).tolist() if gp is not None else None,
                   "greedy_lp": span_logprob(go.outputs[0], kind), "greedy_text": gtxt[:3000],
                   "greedy_ntok": len(go.outputs[0].token_ids), "samples": samples, "samples_lp": lps,
                   "samples_ntok": [len(o.token_ids) for o in so.outputs], "temperature": args.temperature}
            try:
                js = json.load(open(os.path.join(args.json_dir, f"{r['token']}.json")))
                for k in ("velocity", "acceleration", "instruction", "gt_trajectory", "his_trajectory"):
                    rec[k] = js.get(k)
            except Exception:
                pass
            fout.write(json.dumps(rec) + "\n")
        fout.flush()
        el = time.time() - t0
        n = min(c0 + args.chunk, len(rows))
        print(f"[vllm] {name} rank {args.rank}: {n}/{len(rows)}  {el / n:.2f} s/token  "
              f"eta {(len(rows) - n) * el / n / 60:.0f} min", flush=True)
    fout.close()
    print(f"[vllm] done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
