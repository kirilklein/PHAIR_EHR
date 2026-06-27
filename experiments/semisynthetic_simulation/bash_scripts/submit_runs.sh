#!/usr/bin/env bash
# Submit N outer runs of the semi-synthetic study as parallel Azure jobs, on a
# FIXED shared cohort. Each job = one outer simulation (own seed) + K refits.
#
# Usage:
#   ./submit_runs.sh                       # Phase 1: 1 run, K=1 (single fit + analytic CI)
#   ./submit_runs.sh --baseline-only       # Phase 1, baseline only (CPU, fast)
#   ./submit_runs.sh -n 5 -k 10            # full: 5 outer runs, 10 bootstrap refits each
#
# Anything after the known flags is forwarded to the runner (e.g. --bert-only).
# Override defaults via env: POOL=... EXPERIMENT=... TEMPLATE=... ./submit_runs.sh ...
set -euo pipefail

POOL="${POOL:-CPU-20-LP}"
EXPERIMENT="${EXPERIMENT:-semisynthetic_study}"
TEMPLATE="${TEMPLATE:-experiments/semisynthetic_simulation/job_config_template.yaml}"
GENERATED_DIR="${GENERATED_DIR:-experiments/semisynthetic_simulation/generated_job_configs}"

N_RUNS=1
INNER_RUNS=1
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) N_RUNS="$2"; shift 2 ;;
    -k) INNER_RUNS="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -f "$TEMPLATE" ]]; then
  echo "Template not found: $TEMPLATE" >&2; exit 1
fi
mkdir -p "$GENERATED_DIR"

for ((i = 1; i <= N_RUNS; i++)); do
  RUN=$(printf "run_%02d" "$i")
  CFG="$GENERATED_DIR/$RUN.yaml"
  sed "s|__RUN__|$RUN|g" "$TEMPLATE" > "$CFG"

  BASH_ARGS="--run-id $RUN --inner-runs $INNER_RUNS ${EXTRA_ARGS[*]:-}"

  echo "Submitting $RUN  (inner_runs=$INNER_RUNS) ${EXTRA_ARGS[*]:-}"
  python -m corebehrt.azure job run_semisynthetic_study "$POOL" \
    -e "$EXPERIMENT" \
    -c "$CFG" \
    --bash-args "$BASH_ARGS"
done
