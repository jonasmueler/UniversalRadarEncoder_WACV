#!/usr/bin/env bash
# CI4R-ASL: embeddings, sequence caches, head training and test for one encoder and seed.
#
#   scripts/eval_ci4r_asl.sh <checkpoint> <encoder config> <seed> [all|hearing] [tag]
#
# "hearing" restricts the fixed split to the hearing participants.
source "$(dirname "$0")/_env.sh"
CKPT=${1:?"usage: eval_ci4r_asl.sh <checkpoint> <config> <seed> [all|hearing] [tag]"}
CFG=${2:?"encoder config required"}
SEED=${3:?"seed required"}
GROUP=${4:-all}
TAG=${5:-ci4r_asl_$(cfg_tag_of "$CFG")}
GROUP_TAG=""; [ "$GROUP" = hearing ] && GROUP_TAG="_hearing"
RUN="${TAG}${GROUP_TAG}_s${SEED}"
EMBED_DIM=$(embed_dim_of "$CFG")
KEY=$(cfg_tag_of "$CFG")_$(ckpt_hash_of "$CKPT")
EMB=$FR_CACHE/ci4r_asl_emb/$KEY
SEQ=$FR_CACHE/ci4r_asl_seq/${KEY}${GROUP_TAG}
CLS=$(dirname "$CKPT")/classification/ci4r_asl${GROUP_TAG}_s${SEED}
SUBJECTS=$(python -c "from frozenradar.data.ci4r_splits import CANONICAL_SUBJECTS as S; print(' '.join(S))")

echo "=== benchmark: $RUN ==="
if [ ! -f "$SEQ/.done" ]; then
    # Per-subject embeddings, extracted once per (encoder, checkpoint).
    for s in $SUBJECTS; do
        if [ ! -f "$EMB/subject_$s/.done" ]; then
            rm -rf "$EMB/subject_$s"
            python -m frozenradar.extract_embeddings_ci4r \
                --config "$CFG" --checkpoint "$CKPT" \
                --root "$ASL_ROOT/raw/subject $s" --output-dir "$EMB/subject_$s" \
                --training.batch_size 32 --training.num_workers "$FR_WORKERS"
            touch "$EMB/subject_$s/.done"
        fi
    done
    # The split this head evaluates must be the partition the pretraining store
    # was shielded against: both hash to the pinned canonical digest.
    python - <<PYEOF
from pathlib import Path
from frozenradar.classification.build_sequences_ci4r import _enumerate_units
from frozenradar.data.ci4r_splits import CANONICAL_DIGEST, CANONICAL_UNIT_COUNT, _digest
units = _enumerate_units(Path("$EMB"), Path("$ASL_LABELS"), "$SUBJECTS".split())
d = _digest([f"{u['subject']}|{u['base']}|{u['rep']}" for u in units])
assert d == CANONICAL_DIGEST["ci4r"] and len(units) == CANONICAL_UNIT_COUNT["ci4r"], (
    f"enumerated split {d} ({len(units)} units) != canonical "
    f"{CANONICAL_DIGEST['ci4r']} ({CANONICAL_UNIT_COUNT['ci4r']} units)")
print(f"split digest verified: {d} ({len(units)} units)")
PYEOF
    rm -rf "$SEQ"; mkdir -p "$SEQ"
    python -m frozenradar.classification.build_sequences_ci4r \
        --embedded-root "$EMB" --labels-root "$ASL_LABELS" --output-dir "$SEQ" \
        --subjects $SUBJECTS --subject-group "$GROUP" \
        --min-seg-len 5 --max-seg-len 512 --test-size 0.2 --val-size 0.1 --seed 1
    touch "$SEQ/.done"
fi
python -m frozenradar.classification.train \
    --config configs/downstream/base.yaml \
    --dataset.sequences_dir "$SEQ" --model.embed_dim "$EMBED_DIM" \
    --model.num_classes 18 --model.temporal_layers 4 \
    --model.alpha_schedule fixed --model.alpha_max 0.0 \
    --training.epochs 200 --training.num_workers "$FR_WORKERS" \
    --training.checkpoint_dir "$CLS" --training.seed "$SEED" \
    --training.eval_only "$EVAL_ONLY" \
    --wandb.run_name "$RUN" 2>&1 | tee "$FR_LOGS/$RUN.log"
