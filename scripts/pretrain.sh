#!/usr/bin/env bash
# Pretrain one encoder; <pool> is one of mcag mca cag mag mcg (configs/pretrain).
#
#   scripts/pretrain.sh <pool> [--section.key value ...]
#   scripts/pretrain.sh mcag --training.resume auto   # continue an interrupted run
source "$(dirname "$0")/_env.sh"
POOL=${1:?"usage: scripts/pretrain.sh <mcag|mca|cag|mag|mcg> [overrides...]"}
CFG=configs/pretrain/pool_${POOL}.yaml
[ -f "$CFG" ] || { echo "FATAL: $CFG does not exist"; exit 1; }

SOURCES=$(python -c "from frozenradar.utils.config import load_config; print(' '.join(load_config('$CFG').dataset.sources))")
echo "=== pretraining pool_$POOL: $SOURCES ==="
VERIFY=()
case " $SOURCES " in *" ci4r "*)   VERIFY+=(--ci4r-store "$ASL_STORE" --ci4r-manifest splits/ci4r_asl_manifest.json) ;; esac
case " $SOURCES " in *" multi3 "*) VERIFY+=(--multi3-store "$GAIT_STORE" --multi3-manifest splits/ci4r_gait_manifest.json) ;; esac
if [ ${#VERIFY[@]} -gt 0 ]; then
    python -m frozenradar.data.verify_pretrain_stores "${VERIFY[@]}"
fi

python -m frozenradar.train --config "$CFG" \
    --dataset.root "$MMRADPOSE_ROOT" \
    --dataset.cubelearn_root "$CUBELEARN_ROOT/activity_organized" \
    --dataset.cubelearn_hgr_root "$CUBELEARN_ROOT/hand_organized" \
    --dataset.cubelearn_agr_root "$CUBELEARN_ROOT/arm_organized" \
    --dataset.ci4r_root "$ASL_STORE" \
    --dataset.multi3_root "$GAIT_STORE" \
    --training.num_workers "$FR_WORKERS" \
    "${@:2}"
