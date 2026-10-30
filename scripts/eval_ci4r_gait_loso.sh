#!/usr/bin/env bash
# CI4R-Gait leave-one-subject-out: one fold subject is the test set.
#
#   scripts/eval_ci4r_gait_loso.sh <checkpoint> <encoder config> <seed> <0010|0014|0029|0030> [tag]
#
# Refuses encoders whose pretraining pool contains CI4R-Gait.
source "$(dirname "$0")/_env.sh"
CKPT=${1:?"usage: eval_ci4r_gait_loso.sh <checkpoint> <config> <seed> <fold> [tag]"}
CFG=${2:?"encoder config required"}
SEED=${3:?"seed required"}
FOLD=${4:?"fold subject required (0010|0014|0029|0030)"}
TAG=${5:-ci4r_gait_loso_$(cfg_tag_of "$CFG")}
RUN="${TAG}_f${FOLD}_s${SEED}"
GAIT_IN_POOL=$(python -c "from frozenradar.utils.config import load_config; print('yes' if 'multi3' in (load_config('$CFG').dataset.sources or []) else 'no')")
if [ "$GAIT_IN_POOL" != "no" ]; then
    echo "FATAL: $CFG pretrains on CI4R-Gait ('multi3' in dataset.sources); its pool"
    echo "       contains recordings of every subject, including fold $FOLD."
    exit 1
fi
EMBED_DIM=$(embed_dim_of "$CFG")
KEY=$(cfg_tag_of "$CFG")_$(ckpt_hash_of "$CKPT")
EMB=$FR_CACHE/ci4r_gait_emb/$KEY
SEQ=$FR_CACHE/ci4r_gait_seq/${KEY}_loso_f${FOLD}
CLS=$(dirname "$CKPT")/classification/ci4r_gait_loso_f${FOLD}_s${SEED}

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
    python -m frozenradar.classification.build_sequences_multi3_loso \
        --embedded-dir "$EMB" --output-dir "$SEQ" \
        --fold-subject "$FOLD" --max-frames 512 --val-size 0.1 --seed 1
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
