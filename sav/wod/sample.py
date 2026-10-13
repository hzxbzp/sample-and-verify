#!/usr/bin/env python3
"""Candidate plans of a VLM planner: its greedy plan and K temperature samples, with their log-likelihoods.

Works with Hugging Face vision-language models whose processor renders a ``[system, user(image, text)]``
conversation (Qwen2.5-VL / Qwen3-VL families, e.g. a fine-tuned Cosmos-Reason2 or Qwen3-VL planner). Samples use
temperature T without top-k / top-p truncation; every candidate keeps [summed log p, number of tokens] of its
answer. Output (one line per frame, resumable)::

    {"scene_id", "frame_id", "fdir", "greedy_xy": [[x, y]] * 20 | null, "greedy_lp": [sum, n],
     "samples_xy": [... K plans or null ...], "samples_lp": [[sum, n]] * K, "greedy_text", "sample_texts"}

    python -m sav.wod.sample --model CKPT --frames-root FRAMES --out cands.jsonl --k 16 [--shard 0 --nshards 4]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Optional, Tuple

import numpy as np

from sav.wod import frames as F
from sav.wod.prompt import chat_messages
from sav.wod.traj import traj_from_text

TERMINATORS = ("<|im_end|>", "<|endoftext|>")


def eos_ids(tokenizer) -> List[int]:
    ids = []
    i = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(i, int) and i >= 0:
        ids.append(i)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(int(tokenizer.eos_token_id))
    return ids


def strip_terminators(text: str) -> str:
    cut = len(text)
    for t in TERMINATORS:
        i = text.find(t)
        if i >= 0:
            cut = min(cut, i)
    return text[:cut]


def load_model(path: str, device: str, attn: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation=attn)
    return model.to(device).eval(), AutoProcessor.from_pretrained(path)


def build_inputs(processor, frame: dict, image_path: str, long_edge: Optional[int], intent: bool):
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    if long_edge and max(img.size) != long_edge:
        s = long_edge / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
    text = processor.apply_chat_template(chat_messages(frame, intent), add_generation_prompt=True, tokenize=False)
    px = img.width * img.height
    return processor(text=[text], images=[img], return_tensors="pt", images_kwargs={"min_pixels": px, "max_pixels": px})


def generate(model, processor, inputs, *, n: int, temperature: float, max_new_tokens: int,
             chunk: int = 8) -> List[Tuple[str, List[float]]]:
    """n answers (greedy if temperature == 0) as (text, [sum log p, n tokens]) up to the first terminator."""
    import torch
    tok = processor.tokenizer
    eos = eos_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    kw = {k: v.to(model.device) for k, v in inputs.items()}
    plen = int(kw["input_ids"].shape[1])
    out: List[Tuple[str, List[float]]] = []
    left = n
    with torch.no_grad():
        while left > 0:
            m = min(chunk, left) if temperature > 0 else 1
            sample = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0) if temperature > 0 \
                else dict(do_sample=False)
            g = model.generate(**kw, max_new_tokens=max_new_tokens, num_return_sequences=m, eos_token_id=eos,
                               pad_token_id=pad, use_cache=True, output_scores=True, return_dict_in_generate=True,
                               **sample)
            lp = model.compute_transition_scores(g.sequences, g.scores, normalize_logits=True).float().cpu().numpy()
            for i, row in enumerate(g.sequences[:, plen:].tolist()):
                cut = next((j for j, t in enumerate(row) if t in eos), len(row) - 1)
                v = lp[i, :cut + 1]
                v = v[np.isfinite(v)]
                out.append((strip_terminators(tok.decode(row[:cut + 1], skip_special_tokens=False)),
                            [float(v.sum()), int(len(v))]))
            left -= m
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="fine-tuned planner (Hugging Face checkpoint directory)")
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=8, help="samples per generate() call")
    ap.add_argument("--long-edge", type=int, default=2916, help="panorama long edge given to the processor")
    ap.add_argument("--no-intent", action="store_true", help="planner trained without the route command in the prompt")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attn", default="flash_attention_2")
    args = ap.parse_args()
    frames = F.list_frames(args.frames_root)[args.shard::args.nshards]
    if args.limit:
        frames = frames[: args.limit]
    done = set()
    if os.path.isfile(args.out):
        done = {(r["scene_id"], r["frame_id"]) for r in map(json.loads, open(args.out))}
    todo = [f for f in frames if (f[0], f[1]) not in done]
    print(f"[sample] {len(frames)} frames, {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return
    model, processor = load_model(args.model, args.device, args.attn)
    t0 = time.time()
    with open(args.out, "a") as fh:
        for i, (scene, fid, fdir) in enumerate(todo, 1):
            frame = F.load(fdir)
            hist = F.history_xy(frame)
            inputs = build_inputs(processor, frame, os.path.join(fdir, F.IMAGE_NAME), args.long_edge,
                                  not args.no_intent)
            (gtext, glp), = generate(model, processor, inputs, n=1, temperature=0.0, max_new_tokens=args.max_new_tokens)
            smp = generate(model, processor, inputs, n=args.k, temperature=args.temperature,
                           max_new_tokens=args.max_new_tokens, chunk=args.chunk)
            g = traj_from_text(gtext, hist)
            xs = [traj_from_text(t, hist) for t, _ in smp]
            fh.write(json.dumps({"scene_id": scene, "frame_id": fid, "fdir": fdir,
                                 "greedy_xy": g.tolist() if g is not None else None, "greedy_lp": glp,
                                 "samples_xy": [x.tolist() if x is not None else None for x in xs],
                                 "samples_lp": [lp for _, lp in smp], "greedy_text": gtext,
                                 "sample_texts": [t for t, _ in smp]}) + "\n")
            fh.flush()
            if i % 20 == 0:
                el = time.time() - t0
                print(f"[sample] {i}/{len(todo)}  {el / i:.2f} s/frame", flush=True)


if __name__ == "__main__":
    main()
