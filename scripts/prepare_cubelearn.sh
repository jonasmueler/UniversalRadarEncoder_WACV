#!/usr/bin/env bash
# CubeLearn: download and unpack the three task directories.
#
#   scripts/prepare_cubelearn.sh
source "$(dirname "$0")/_env.sh"
python -m frozenradar.data.download_cubelearn --output-dir "$CUBELEARN_ROOT"
if [ ! -d "$CUBELEARN_ROOT/activity_organized" ]; then
    unzip -q "$CUBELEARN_ROOT/HAR_data.zip" -d "$CUBELEARN_ROOT"
fi
for d in activity_organized hand_organized arm_organized; do
    printf '%-20s %6d files\n' "$d" "$(ls "$CUBELEARN_ROOT/$d" | wc -l)"
done
