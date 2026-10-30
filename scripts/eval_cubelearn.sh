#!/usr/bin/env bash
# CubeLearn HAR / HGR / AGR: in-set and out-of-set test from one head training.
#
#   scripts/eval_cubelearn.sh <checkpoint> <encoder config> <har|hgr|agr> <seed> [tag]
source "$(dirname "$0")/_env.sh"
CKPT=${1:?"usage: eval_cubelearn.sh <checkpoint> <config> <har|hgr|agr> <seed> [tag]"}
CFG=${2:?"encoder config required"}
TASK=${3:?"task required (har|hgr|agr)"}
SEED=${4:?"seed required"}
TAG=${5:-cubelearn_${TASK}_$(cfg_tag_of "$CFG")}
RUN="${TAG}_s${SEED}"
case "$TASK" in
    har) FRAMES=20; NGEST=6;  DATA_ROOT=$CUBELEARN_ROOT/activity_organized ;;
    hgr) FRAMES=10; NGEST=12; DATA_ROOT=$CUBELEARN_ROOT/hand_organized ;;
    agr) FRAMES=10; NGEST=12; DATA_ROOT=$CUBELEARN_ROOT/arm_organized ;;
    *) echo "FATAL: unknown task '$TASK'"; exit 1 ;;
esac
EMBED_DIM=$(embed_dim_of "$CFG")
KEY=${TASK}_$(cfg_tag_of "$CFG")_$(ckpt_hash_of "$CKPT")
EMB=$FR_CACHE/cubelearn_emb/$KEY
SEQ=$FR_CACHE/cubelearn_seq/$KEY
OOS=$FR_CACHE/cubelearn_seq/${KEY}_outofset
CLS=$(dirname "$CKPT")/classification/cubelearn_${TASK}_s${SEED}

echo "=== benchmark: $RUN ==="
if [ ! -f "$SEQ/.done" ] || [ ! -f "$OOS/.done" ]; then
    rm -rf "$EMB" "$SEQ" "$OOS"; mkdir -p "$EMB" "$SEQ" "$OOS"
    # in-set users 0-5 (default) and out-of-set users 6-7, all 30 samples each
    python -m frozenradar.extract_embeddings_cubelearn \
        --config "$CFG" --checkpoint "$CKPT" --cubelearn-root "$DATA_ROOT" --task "$TASK" \
        --output-dir "$EMB/inset" --training.batch_size 32 --training.num_workers "$FR_WORKERS"
    python -m frozenradar.extract_embeddings_cubelearn \
        --config "$CFG" --checkpoint "$CKPT" --cubelearn-root "$DATA_ROOT" --task "$TASK" \
        --output-dir "$EMB/outofset" --users 6 7 \
        --training.batch_size 32 --training.num_workers "$FR_WORKERS"
    python -m frozenradar.classification.build_sequences_cubelearn \
        --embedded-dir "$EMB/inset" --output-dir "$SEQ" \
        --n-gestures "$NGEST" --frames-per-recording "$FRAMES"
    python -m frozenradar.classification.build_sequences_cubelearn \
        --embedded-dir "$EMB/outofset" --output-dir "$OOS" \
        --n-gestures "$NGEST" --frames-per-recording "$FRAMES" --split-mode outofset
    touch "$SEQ/.done" "$OOS/.done"
fi
python -m frozenradar.classification.train \
    --config configs/downstream/cubelearn.yaml \
    --dataset.sequences_dir "$SEQ" --dataset.outofset_test_dir "$OOS/test" \
    --model.embed_dim "$EMBED_DIM" --model.num_classes "$NGEST" \
    --model.alpha_schedule fixed --model.alpha_max 0.0 \
    --training.epochs 200 --training.num_workers "$FR_WORKERS" \
    --training.checkpoint_dir "$CLS" --training.seed "$SEED" \
    --training.eval_only "$EVAL_ONLY" \
    --wandb.run_name "$RUN" 2>&1 | tee "$FR_LOGS/$RUN.log"
