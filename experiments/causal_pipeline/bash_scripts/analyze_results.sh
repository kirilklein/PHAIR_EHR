#!/bin/bash
# A safer way to write Bash scripts by exiting on errors and undefined variables.
set -euo pipefail

# ========================================
# Analyze Causal Pipeline Experiment Results
# ========================================

show_help() {
    echo "CAUSAL PIPELINE RESULTS ANALYZER"
    echo "================================="
    echo ""
    echo "DESCRIPTION:"
    echo "  Analyzes results from causal pipeline experiments. Can process all experiments"
    echo "  or specific ones, with options to filter by run and customize input/output directories."
    echo ""
    echo "USAGE:"
    echo "  ./analyze_results.sh [OPTIONS] [EXPERIMENTS...]"
    echo ""
    echo "OPTIONS:"
    echo "  --run_id RUN_ID           Analyze results from specific run only (e.g., run_01, run_02)"
    echo "  --results_dir DIR         Directory containing experiment results"
    echo "  --results-dir DIR         (alternative: use hyphen instead of underscore)"
    echo "                              Default: ../../../outputs/causal/sim_study/runs"
    echo "  --output_dir DIR          Directory to save analysis outputs"
    echo "  --output-dir DIR          (alternative: use hyphen instead of underscore)"
    echo "                              Default: ../../../outputs/causal/sim_study/analysis"
    echo "  --outcomes \"O1 O2\"        Filter analysis to specific outcomes only (must be in quotes)"
    echo "  --max-subplots N          Maximum subplots per figure (e.g., 4 creates 2x2 grids)"
    echo "  --min-points N            Minimum data points required to generate a plot (default: 2)"
    echo "  --estimator TYPE          Which estimator(s) to analyze: baseline, bert, or both (default: both)"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "EXPERIMENTS:"
    echo "  all                       Analyze all available experiments (default if none specified)"
    echo "  experiment_name           Specific experiment to analyze (e.g., ce0_cy0_y0_i0)"
    echo "  exp1 exp2 exp3            Multiple specific experiments"
    echo ""
    echo "EXAMPLES:"
    echo "  ./analyze_results.sh"
    echo "    → Analyze all experiments for both estimators, aggregated across all runs."
    echo ""
    echo "  ./analyze_results.sh ce0_cy0_y0_i0 ce1_cy1_y0_i0"
    echo "    → Analyze specific experiments for both estimators."
    echo ""
    echo "  ./analyze_results.sh --run_id run_01 all"
    echo "    → Analyze all experiments from run_01 only."
    echo ""
    echo "  ./analyze_results.sh --estimator baseline"
    echo "    → Analyze only baseline estimator results."
    echo ""
    echo "  ./analyze_results.sh --outcomes \"OUTCOME_1 OUTCOME_2\""
    echo "    → Filter for specific outcomes (note the quotes)."
    echo ""
    echo "NOTES:"
    echo "  • Results are organized by estimator: output_dir/baseline/ and output_dir/bert/"
    echo "  • By default, results are aggregated across all runs (run_01, run_02, etc.)."
    echo "  • Use --run_id to analyze results from a specific run only."
    echo "  • The script will create the output directory if it doesn't exist."
    echo ""
}

# --- 1. SET DEFAULTS ---
results_dir="../../../outputs/causal/sim_study/runs"
output_dir="../../../outputs/causal/sim_study/analysis"
run_id=""
max_subplots=""
min_points="2"
# Use arrays for multi-value arguments. Default is to run both estimators.
estimators_to_run=("baseline" "bert")
outcomes=()
experiments=()

# --- 2. PARSE ARGUMENTS WITH GETOPT ---
# This provides robust parsing for both long and short options.
ARGS=$(getopt -o "h" \
  -l "help,run_id:,run-id:,results_dir:,results-dir:,output_dir:,output-dir:,outcomes:,max-subplots:,min-points:,estimator:" \
  -n "$0" -- "$@")

if [ $? -ne 0 ]; then
    echo "Error parsing arguments. Use --help for usage." >&2
    exit 1
fi

eval set -- "$ARGS"

while true; do
    case "$1" in
        --run_id|--run-id)
            run_id="$2"
            shift 2
            ;;
        --results_dir|--results-dir)
            results_dir="$2"
            shift 2
            ;;
        --output_dir|--output-dir)
            output_dir="$2"
            shift 2
            ;;
        --outcomes)
            # This expects a single quoted string for multiple outcomes, e.g., --outcomes "O1 O2"
            outcomes=($2)
            shift 2
            ;;
        --max-subplots)
            max_subplots="$2"
            shift 2
            ;;
        --min-points)
            min_points="$2"
            shift 2
            ;;
        --estimator)
            # This logic explicitly handles the user's choice.
            if [[ "$2" == "both" ]]; then
                estimators_to_run=("baseline" "bert")
            elif [[ "$2" == "baseline" || "$2" == "bert" ]]; then
                estimators_to_run=("$2")
            else
                echo "Error: Invalid estimator '$2'. Must be 'baseline', 'bert', or 'both'." >&2
                exit 1
            fi
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --)
            shift # Discard the '--' separator
            break # End of options
            ;;
        *)
            echo "Internal error!" >&2
            exit 1
            ;;
    esac
done

# Any remaining arguments are treated as positional arguments (experiments)
experiments=("$@")


# --- 3. PREPARE & EXECUTE ---
# If a specific run ID is specified, append it to the directory paths.
if [ -n "$run_id" ]; then
    results_dir="$results_dir/$run_id"
    output_dir="$output_dir/$run_id"
fi

echo "========================================"
echo "Analyzing Experiment Results"
echo "========================================"
echo "Results Directory: $results_dir"
echo "Output Directory:  $output_dir"
echo "Estimator(s):      ${estimators_to_run[*]}"
[ -n "$run_id" ] && echo "Run ID:            $run_id"
[ ${#outcomes[@]} -gt 0 ] && echo "Outcomes:          ${outcomes[*]}"
[ -n "$max_subplots" ] && echo "Max Subplots:      $max_subplots"
echo "Min Points:        $min_points"

run_all_experiments=false
if [ ${#experiments[@]} -eq 0 ] || [[ " ${experiments[*]} " =~ " all " ]]; then
    run_all_experiments=true
    echo "Analyzing ALL experiments..."
else
    echo "Analyzing specific experiments: ${experiments[*]}"
fi
echo ""

# Build the command in a Bash array to handle spaces and special characters safely.
# This is the modern, secure replacement for building a command string with `eval`.
python_cmd=(
    "python"
    "../python_scripts/analyze_experiment_results.py"
    "--results_dir" "$results_dir"
    "--output_dir" "$output_dir"
    "--min-points" "$min_points"
    # This correctly passes one or more estimators to the Python script.
    "--estimator" "${estimators_to_run[@]}"
)

# Conditionally add other optional arguments to the command array.
if [ -n "$max_subplots" ]; then
    python_cmd+=("--max-subplots" "$max_subplots")
fi
if [ ${#outcomes[@]} -gt 0 ]; then
    python_cmd+=("--outcomes" "${outcomes[@]}")
fi
if [ "$run_all_experiments" = "false" ]; then
    # This assumes your Python script can handle an --experiments flag.
    python_cmd+=("--experiments" "${experiments[@]}")
fi


# --- 4. RUN THE ANALYSIS ---
echo "Executing command:"
# The 'printf' command helps visualize how the arguments are being passed.
printf "%q " "${python_cmd[@]}"
echo -e "\n"

# Execute the command safely
"${python_cmd[@]}"

# The '$?' variable holds the exit code of the most recently executed command.
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Analysis script failed with a non-zero exit code." >&2
    exit 1
fi

echo ""
echo "========================================"
echo "Analysis completed successfully!"
echo "Results saved in: $output_dir"
echo "========================================"