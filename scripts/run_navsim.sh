#!/bin/bash
# Sample-and-verify on NAVSIM for one public planner. Run one step at a time from the repository root:
#   bash scripts/run_navsim.sh <step>
# Environment: NAVSIM_DEVKIT_ROOT (NAVSIM v1 devkit), OPENSCENE_DATA_ROOT (navsim_logs/, sensor_blobs/),
# AUTOVLA_ROOT (public AutoVLA code, scene preprocessing), METRIC_CACHE_TEST / METRIC_CACHE_TRAIN (devkit metric caches),
# KIND (curious | recog), MODEL_PATH (planner checkpoint), CURIOUS_STATS (Curious-VLA trajectory_stats_train.json).
set -euo pipefail
export PYTHONPATH=$PWD:$NAVSIM_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}
W=${W:-work}
KIND=${KIND:-recog}
NAME=${NAME:-$KIND}
NGPU=${NGPU:-1}
NW=${NW:-16}
mkdir -p $W

gen() {   # split prompts_split
  for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$i python navsim/gen_candidates.py --kind $KIND --model-path $MODEL_PATH --name $NAME \
      ${CURIOUS_STATS:+--curious-stats $CURIOUS_STATS} --prompts $W/prompts/$1.jsonl --json-dir $W/json/$1 \
      --out $W/runs/$1_$NAME/cands_r$i.jsonl --rank $i --world $NGPU --k ${K:-16} &
  done
  wait
}

case "$1" in
  subset)      # navtrain scenes for the verifier
    python navsim/make_subset.py --n ${N:-8000} --out-dir $W/splits ;;
  json)        # scene JSONs with the AutoVLA preprocessing; CFG = its dataset config for the split, SPLIT = test | train
    (cd $AUTOVLA_ROOT && HISTORY_SENSORS_ONLY=${HISTORY:-0} python $OLDPWD/navsim/run_preprocessing.py --config $CFG \
      --output_dir $OLDPWD/$W/json/$SPLIT --num_workers $NW) ;;
  prompts)
    python navsim/prep_prompts.py --split test --tokens-from $W/json/test --out $W/prompts/test.jsonl --img-dir $W/prompts/img_test --workers $NW
    python navsim/prep_prompts.py --split trainval --tokens-from $W/json/train --out $W/prompts/train.jsonl --img-dir $W/prompts/img_train --workers $NW ;;
  gen)
    gen test; gen train ;;
  score)
    python navsim/score_candidates.py --cands "$W/runs/test_$NAME/cands_r*.jsonl" --metric-cache $METRIC_CACHE_TEST \
      --out $W/runs/test_$NAME/pdms.jsonl --workers $NW
    python navsim/score_candidates.py --cands "$W/runs/train_$NAME/cands_r*.jsonl" --metric-cache $METRIC_CACHE_TRAIN \
      --out $W/runs/train_$NAME/pdms.jsonl --workers $NW ;;
  verify)      # features, verifiers with the log target (lambda 1, no PDMS) and the PDMS target (lambda 0), evaluation
    python navsim/sav_navsim.py build --cands "$W/runs/train_$NAME/cands_r*.jsonl" --logs $W/splits/navtrain_sub.tsv \
      --out $W/train_${NAME}_log.npz
    python navsim/sav_navsim.py build --cands "$W/runs/train_$NAME/cands_r*.jsonl" --logs $W/splits/navtrain_sub.tsv \
      --pdms $W/runs/train_$NAME/pdms.jsonl --out $W/train_${NAME}_pdms.npz
    python navsim/sav_navsim.py build --cands "$W/runs/test_$NAME/cands_r*.jsonl" --pdms $W/runs/test_$NAME/pdms.jsonl \
      --out $W/test_$NAME.npz
    for s in 0 1 2; do
      python -m sav.verifier train --train $W/train_${NAME}_log.npz --lam 1 --seed $s --out $W/${NAME}_v_log_s$s.pt
      python -m sav.verifier train --train $W/train_${NAME}_pdms.npz --lam 0 --seed $s --out $W/${NAME}_v_pdms_s$s.pt
    done
    python navsim/sav_navsim.py eval --test $W/test_$NAME.npz --pdms $W/runs/test_$NAME/pdms.jsonl \
      --verifier $W/${NAME}_v_*_s*.pt --json $W/results_$NAME.json ;;
  crossfit)    # diagnostic: verifier cross-fitted on navtest folds
    for lam in 0 1; do
      python navsim/crossfit.py --test $W/test_$NAME.npz --pdms $W/runs/test_$NAME/pdms.jsonl --prompts $W/prompts/test.jsonl \
        --lam $lam --out $W/crossfit_${NAME}_lam$lam.json
    done ;;
  starts)
    python navsim/start_analysis.py $W/test_$NAME.npz ;;
  *) echo "unknown step $1"; exit 1 ;;
esac
