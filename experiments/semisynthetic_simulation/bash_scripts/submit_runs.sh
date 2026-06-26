#!/usr/bin/env bash
# Submit N independent outer runs of the semi-synthetic study as parallel Azure jobs.
# Each job runs one outer simulation (own seed) + K inner reshuffle fits.
#
# Usage:
#   ./submit_runs.sh                 # 1 run, K=2, no sampling   (full population)
#   ./submit_runs.sh -n 2 -k 2 -f 0.1   # SMOKE TEST: 2 runs, 2 inner fits, 10% sampled
#   ./submit_runs.sh -n 10 -k 10        # full study: 10 runs, 10 inner fits each
#
# Override defaults via env: POOL=... EXPERIMENT=... TEMPLATE=... ./submit_runs.sh ...
set -euo pipefail

POOL="${POOL:-CPU-20-LP}"
EXPERIMENT="${EXPERIMENT:-semisynthetic_study}"
TEMPLATE="${TEMPLATE:-experiments/semisynthetic_simulation/job_config_template.yaml}"
GENERATED_DIR="${GENERATED_DIR:-experiments/semisynthetic_simulation/generated_job_configs}"

N_RUNS=1
INNER_RUNS=2
SAMPLE_FRACTION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) N_RUNS="$2"; shift 2 ;;
    -k) INNER_RUNS="$2"; shift 2 ;;
    -f) SAMPLE_FRACTION="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
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

  BASH_ARGS="--run-id $RUN --inner-runs $INNER_RUNS"
  [[ -n "$SAMPLE_FRACTION" ]] && BASH_ARGS="$BASH_ARGS --sample-fraction $SAMPLE_FRACTION"

  echo "Submitting $RUN  (inner_runs=$INNER_RUNS, sample_fraction=${SAMPLE_FRACTION:-none})"
  python -m corebehrt.azure job run_semisynthetic_study "$POOL" \
    -e "$EXPERIMENT" \
    -c "$CFG" \
    --bash-args "$BASH_ARGS"
done
