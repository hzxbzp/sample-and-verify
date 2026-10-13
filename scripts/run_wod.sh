#!/bin/bash
# Sample-and-verify on WOD-E2E. Set the planner checkpoint and the frames roots of the splits, then run from the
# repository root: bash scripts/run_wod.sh
set -euo pipefail
PLANNER=${PLANNER:?fine-tuned planner checkpoint}
TRAIN=${TRAIN:?training frames}
DEV=${DEV:?dev frames (held-out training scenes)}
VAL=${VAL:?rated validation frames}
TEST=${TEST:-}
CLUSTERS=${CLUSTERS:-}
OUT=${OUT:-out}
K=${K:-16}
NGPU=${NGPU:-1}
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
mkdir -p $OUT/cands $OUT/feats

sample() {   # split frames_root: one process per GPU
  for i in $(seq 0 $((NGPU - 1))); do
    python -m sav.wod.sample --model $PLANNER --frames-root $2 --out $OUT/cands/$1_s$i.jsonl --k $K \
      --shard $i --nshards $NGPU --device cuda:$i &
  done
  wait
}

for split in train dev val; do
  root=$(eval echo \$${split^^})
  sample $split $root
  python -m sav.wod.build --cands $OUT/cands/${split}_s*.jsonl --out $OUT/feats/$split.npz
done

for lam in 1 0; do
  python -m sav.verifier train --train $OUT/feats/train.npz --val $OUT/feats/dev.npz --lam $lam --out $OUT/verifier_lam$lam.pt
done

python -m sav.wod.evaluate --data $OUT/feats/val.npz --frames-root $VAL --verifier $OUT/verifier_lam1.pt $OUT/verifier_lam0.pt \
  ${CLUSTERS:+--clusters $CLUSTERS} --json $OUT/val_results.json

if [ -n "$TEST" ]; then
  sample test $TEST
  python -m sav.wod.build --cands $OUT/cands/test_s*.jsonl --no-labels --out $OUT/feats/test.npz
  python -m sav.wod.predict --data $OUT/feats/test.npz --verifier $OUT/verifier_lam1.pt --out $OUT/preds_test.json
fi
