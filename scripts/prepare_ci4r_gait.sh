#!/usr/bin/env bash
# CI4R-Gait: download, rebuild and check the split manifest, build the pretraining store.
#
#   scripts/prepare_ci4r_gait.sh
#   SKIP_DOWNLOAD=1 scripts/prepare_ci4r_gait.sh
source "$(dirname "$0")/_env.sh"
mkdir -p "$GAIT_ROOT/raw"

if [ -z "${SKIP_DOWNLOAD:-}" ]; then
    python -m frozenradar.data.download_ci4r_gait --output-dir "$GAIT_ROOT/raw"
fi
echo "activity dirs: $(ls -d "$GAIT_ROOT"/raw/*/ | wc -l)   recordings: $(find -L "$GAIT_ROOT/raw" -name '*.bin' | wc -l) (expected 652)"

python -m frozenradar.data.ci4r_splits --dataset multi3 \
    --raw-root "$GAIT_ROOT/raw" \
    --out "$GAIT_ROOT/ci4r_gait_manifest.json" --expected-digest a8d6e1becf5193ac

if [ ! -d "$GAIT_STORE" ]; then
    python -m frozenradar.data.extract_ci4r_reps --dataset multi3 \
        --manifest splits/ci4r_gait_manifest.json --raw-root "$GAIT_ROOT/raw" \
        --out-root "$GAIT_STORE"
fi
python -m frozenradar.data.verify_pretrain_stores \
    --multi3-store "$GAIT_STORE" --multi3-manifest splits/ci4r_gait_manifest.json
echo "CI4R-Gait ready: raw=$GAIT_ROOT/raw store=$GAIT_STORE"
