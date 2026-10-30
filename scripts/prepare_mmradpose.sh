#!/usr/bin/env bash
# mmRadPose: download the raw ADC data and convert it to per-frame .npy files.
#
#   scripts/prepare_mmradpose.sh
source "$(dirname "$0")/_env.sh"
python -c "from frozenradar.core.download import download; download('mmradpose', data_dir='$FR_DATA', variant='raw_adc')"
python -m frozenradar.core.preprocess_mmradpose --root "$MMRADPOSE_ROOT"
echo "mmRadPose ready at $MMRADPOSE_ROOT"
