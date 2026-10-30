# Shared environment for the run scripts. Source this file; do not execute it.
#
#   FR_DATA   root holding the four datasets (see README, "Data layout")
#   FR_CACHE  where embeddings and sequence caches are written (default: ./cache)
#   FR_LOGS   where the per-run logs the aggregator parses are written (default: ./logs)
#   FR_WORKERS number of DataLoader workers (default: 8)
#   FR_EVAL_ONLY 1 = evaluate the heads already in <encoder dir>/classification/ (e.g. the
#              released ones) instead of training them (default: 0)
set -euo pipefail
: "${FR_DATA:?set FR_DATA to the directory holding the datasets (see README)}"
FR_CACHE=${FR_CACHE:-cache}
FR_LOGS=${FR_LOGS:-logs}
FR_WORKERS=${FR_WORKERS:-8}
FR_EVAL_ONLY=${FR_EVAL_ONLY:-0}
if [ "$FR_EVAL_ONLY" = 1 ]; then EVAL_ONLY=true; else EVAL_ONLY=false; fi
mkdir -p "$FR_CACHE" "$FR_LOGS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1

MMRADPOSE_ROOT=${MMRADPOSE_ROOT:-$FR_DATA/mmRadPose_rawdata}
CUBELEARN_ROOT=${CUBELEARN_ROOT:-$FR_DATA/cubelearn}
ASL_ROOT=${ASL_ROOT:-$FR_DATA/ci4r_asl}          # raw/, processed/, *.zip
GAIT_ROOT=${GAIT_ROOT:-$FR_DATA/ci4r_gait}       # raw/<NN>_<activity>/*.bin
ASL_STORE=${ASL_STORE:-$FR_DATA/ci4r_asl_store}
GAIT_STORE=${GAIT_STORE:-$FR_DATA/ci4r_gait_store}
ASL_LABELS="$ASL_ROOT/processed/ASL Sequential Dataset"

# Embedding width follows the encoder config; a content hash of the checkpoint
# keys every cache so two encoders can never share one.
embed_dim_of() { python -c "from frozenradar.utils.config import load_config; print(load_config('$1').model.latent_dim)"; }
ckpt_hash_of() { sha256sum "$1" | cut -c1-16; }
cfg_tag_of() { basename "$1" .yaml; }
