#!/usr/bin/env bash
# CI4R-ASL: download, unpack, rebuild and check the split manifest, build the pretraining store.
#
#   scripts/prepare_ci4r_asl.sh
#   SKIP_DOWNLOAD=1 scripts/prepare_ci4r_asl.sh
source "$(dirname "$0")/_env.sh"
mkdir -p "$ASL_ROOT/raw" "$ASL_ROOT/processed"

if [ -z "${SKIP_DOWNLOAD:-}" ]; then
    python -m frozenradar.data.download_ci4r_asl --out-dir "$ASL_ROOT"
fi

# Every subject directory the benchmark enumerates (order is fixed in
# frozenradar/data/ci4r_splits.py; here only the set matters). Only the 77 GHz
# stream is unpacked; -n never overwrites, so reruns are safe.
ALL_SUBJECTS="6 7 8 9 12 14-15 1 3 17 5 16 2 4 12_continued 16_continued 5_continued 6-10-11-12-13 6-7 6-7-8 6-7-8-9 6_continued"
for s in $ALL_SUBJECTS; do
    if [ -d "$ASL_ROOT/raw/subject $s" ]; then continue; fi
    z=$(ls "$ASL_ROOT"/*__subject_"$s".zip 2>/dev/null | head -1)
    [ -n "$z" ] || { echo "FATAL: archive for subject $s not found in $ASL_ROOT"; exit 1; }
    echo "--- unpacking $(basename "$z") ---"
    unzip -n -q "$z" '*/77ghz/*' -d "$ASL_ROOT/raw/" || [ $? -le 1 ]
done
if [ ! -d "$ASL_LABELS" ]; then
    echo "--- unpacking labels ---"
    unzip -q "$ASL_ROOT/processed_data.zip" -d "$ASL_ROOT/processed"
fi
echo "raw subject dirs: $(ls -d "$ASL_ROOT"/raw/subject\ * | wc -l) (expected 21)"

# The manifest is rebuilt from the raw tree and must hash to the pinned digest
# of the shipped manifest (d684d5a9125a4bd1, 1496 units); otherwise the
# partition differs from the paper's and nothing downstream is comparable.
python -m frozenradar.data.ci4r_splits --dataset ci4r \
    --raw-root "$ASL_ROOT/raw" --labels-root "$ASL_LABELS" \
    --out "$ASL_ROOT/ci4r_asl_manifest.json" --expected-digest d684d5a9125a4bd1
cmp <(python -c "import json;print(json.load(open('$ASL_ROOT/ci4r_asl_manifest.json'))['order_digest'])") \
    <(python -c "import json;print(json.load(open('splits/ci4r_asl_manifest.json'))['order_digest'])")

# Train-assigned repetitions only, 120 frames per repetition.
if [ ! -d "$ASL_STORE" ]; then
    python -m frozenradar.data.extract_ci4r_reps --dataset ci4r \
        --manifest splits/ci4r_asl_manifest.json --raw-root "$ASL_ROOT/raw" \
        --out-root "$ASL_STORE" --frames-per-rep 120
fi
python -m frozenradar.data.verify_pretrain_stores \
    --ci4r-store "$ASL_STORE" --ci4r-manifest splits/ci4r_asl_manifest.json
echo "CI4R-ASL ready: raw=$ASL_ROOT/raw store=$ASL_STORE"
