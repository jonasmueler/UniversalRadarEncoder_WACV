#!/usr/bin/env bash
# CI4R-Gait: embeddings, sequence caches, head training and test for one encoder and seed.
#
#   scripts/eval_ci4r_gait.sh <checkpoint> <encoder config> <seed> [tag]
source "$(dirname "$0")/_env.sh"
CKPT=${1:?"usage: eval_ci4r_gait.sh <checkpoint> <config> <seed> [tag]"}
CFG=${2:?"encoder config required"}
SEED=${3:?"seed required"}
TAG=${4:-ci4r_gait_$(cfg_tag_of "$CFG")}
RUN="${TAG}_s${SEED}"
EMBED_DIM=$(embed_dim_of "$CFG")
KEY=$(cfg_tag_of "$CFG")_$(ckpt_hash_of "$CKPT")
EMB=$FR_CACHE/ci4r_gait_emb/$KEY
SEQ=$FR_CACHE/ci4r_gait_seq/$KEY
CLS=$(dirname "$CKPT")/classification/ci4r_gait_s${SEED}

echo "=== benchmark: $RUN ==="
if [ ! -f "$SEQ/.done" ]; then
    if [ ! -f "$EMB/.done" ]; then
        rm -rf "$EMB"
        python -m frozenradar.extract_embeddings_multi3 \
            --config "$CFG" --checkpoint "$CKPT" --root "$GAIT_ROOT/raw" --output-dir "$EMB" \
            --training.batch_size 32 --training.num_workers "$FR_WORKERS"
        touch "$EMB/.done"
    fi
    rm -rf "$SEQ"; mkdir -p "$SEQ"
    python -m frozenradar.classification.build_sequences_multi3 \
        --embedded-dir "$EMB" --output-dir "$SEQ" \
        --max-frames 512 --test-size 0.2 --val-size 0.1 --seed 1
    touch "$SEQ/.done"
fi
python -m frozenradar.classification.train \
    --config configs/downstream/base.yaml \
    --dataset.sequences_dir "$SEQ" --model.embed_dim "$EMBED_DIM" \
    --model.num_classes 11 --model.temporal_layers 4 \
    --model.alpha_schedule fixed --model.alpha_max 0.0 \
    --training.epochs 200 --training.num_workers "$FR_WORKERS" \
    --training.checkpoint_dir "$CLS" --training.seed "$SEED" \
    --training.eval_only "$EVAL_ONLY" \
    --wandb.run_name "$RUN" 2>&1 | tee "$FR_LOGS/$RUN.log"
