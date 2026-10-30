#!/usr/bin/env bash
# mmRadPose activity recognition: embeddings, sequence caches, head training and test.
#
#   scripts/eval_mmradpose.sh <checkpoint> <encoder config> <seed> [tag]
source "$(dirname "$0")/_env.sh"
CKPT=${1:?"usage: eval_mmradpose.sh <checkpoint> <config> <seed> [tag]"}
CFG=${2:?"encoder config required"}
SEED=${3:?"seed required"}
TAG=${4:-mmradpose_$(cfg_tag_of "$CFG")}
RUN="${TAG}_s${SEED}"
EMBED_DIM=$(embed_dim_of "$CFG")
KEY=$(cfg_tag_of "$CFG")_$(ckpt_hash_of "$CKPT")
EMB=$FR_CACHE/mmradpose_emb/$KEY
SEQ=$FR_CACHE/mmradpose_seq/$KEY
CLS=$(dirname "$CKPT")/classification/mmradpose_s${SEED}

echo "=== benchmark: $RUN ==="
if [ ! -f "$SEQ/.done" ]; then
    rm -rf "$EMB" "$SEQ"; mkdir -p "$EMB" "$SEQ"
    python -m frozenradar.extract_embeddings \
        --config "$CFG" --checkpoint "$CKPT" \
        --dataset.root "$MMRADPOSE_ROOT" --output-dir "$EMB" \
        --training.batch_size 32 --training.num_workers "$FR_WORKERS"
    python -m frozenradar.classification.build_sequences \
        --embedded-dir "$EMB" --data-root "$MMRADPOSE_ROOT" --output-dir "$SEQ"
    touch "$SEQ/.done"
fi
python -m frozenradar.classification.train \
    --config configs/downstream/base.yaml \
    --dataset.sequences_dir "$SEQ" --model.embed_dim "$EMBED_DIM" \
    --model.alpha_schedule fixed --model.alpha_max 0.0 \
    --training.epochs 200 --training.num_workers "$FR_WORKERS" \
    --training.checkpoint_dir "$CLS" --training.seed "$SEED" \
    --training.eval_only "$EVAL_ONLY" \
    --wandb.run_name "$RUN" 2>&1 | tee "$FR_LOGS/$RUN.log"
