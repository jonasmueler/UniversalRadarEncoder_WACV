#!/usr/bin/env bash
# Mean +- std over seeds for every run in $FR_LOGS (expects seeds 0 1 2).
#   scripts/aggregate.sh [--latex]
source "$(dirname "$0")/_env.sh"
python -m frozenradar.aggregate_seeds --log-dirs "$FR_LOGS" --expected-seeds 3 "$@"
