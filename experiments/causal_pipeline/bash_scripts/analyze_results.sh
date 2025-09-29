#!/bin/bash

# ========================================
# Analyze Causal Pipeline Experiment Results
# ========================================

if [ $# -eq 0 ]; then
    echo "Usage: ./analyze_results.sh [OPTIONS] [experiment1 experiment2 ...]"
    echo ""
    echo "OPTIONS:"
    echo "  --run_id run_XX    Analyze results from specific run only"
    echo "  --help, -h         Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  ./analyze_results.sh all                         (analyze all experiments, aggregate across runs)"
    echo "  ./analyze_results.sh ce0_cy0_y0_i0 ce1_cy1_y0_i0  (analyze specific experiments, aggregate across runs)"
    echo "  ./analyze_results.sh --run_id run_01 all         (analyze all experiments from run_01 only)"
    echo "  ./analyze_results.sh --run_id run_02 ce0_cy0_y0_i0 (analyze specific experiment from run_02 only)"
    echo ""
    echo "NOTES:"
    echo "  - By default, results are aggregated across all runs (run_01, run_02, etc.)"
    echo "  - Use --run_id to analyze results from a specific run only"
    echo "  - Results are saved to: ../../../outputs/causal/sim_study/analysis/"
    echo ""
    echo "Running analysis of ALL experiments since no arguments provided..."
    RUN_ALL=true
    RUN_ID=""
else
    RUN_ALL=false
    RUN_ID=""
fi

# Parse arguments (only if arguments were provided)
if [ "$RUN_ALL" = "false" ]; then
    EXPERIMENTS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --run_id)
            RUN_ID="$2"
            shift 2
            ;;
        -h|--help)
            # Show help (already shown above if no args)
            exit 0
            ;;
        *)
            # Add to experiments list
            if [ -z "$EXPERIMENTS" ]; then
                EXPERIMENTS="$1"
            else
                EXPERIMENTS="$EXPERIMENTS $1"
            fi
            shift
            ;;
    esac
done
fi

# Set directories
RESULTS_DIR="../../../outputs/causal/sim_study/runs"
OUTPUT_DIR="../../../outputs/causal/sim_study/analysis"

# If specific run ID specified, point to that run
if [ -n "$RUN_ID" ]; then
    RESULTS_DIR="$RESULTS_DIR/$RUN_ID"
    OUTPUT_DIR="$OUTPUT_DIR/$RUN_ID"
fi

echo "========================================"
echo "Analyzing Experiment Results"
echo "========================================"
echo ""

if [ -n "$RUN_ID" ]; then
    echo "Analyzing results from specific run: $RUN_ID"
else
    echo "Analyzing results aggregated across all runs"
fi

if [ "$RUN_ALL" = "true" ]; then
    echo "Analyzing ALL experiments in $RESULTS_DIR"
    python ../python_scripts/analyze_experiment_results.py --results_dir "$RESULTS_DIR" --output_dir "$OUTPUT_DIR"
elif [ "$EXPERIMENTS" = "all" ]; then
    echo "Analyzing ALL experiments in $RESULTS_DIR"
    python ../python_scripts/analyze_experiment_results.py --results_dir "$RESULTS_DIR" --output_dir "$OUTPUT_DIR"
elif [ -z "$EXPERIMENTS" ]; then
    echo "Analyzing ALL experiments in $RESULTS_DIR"
    python ../python_scripts/analyze_experiment_results.py --results_dir "$RESULTS_DIR" --output_dir "$OUTPUT_DIR"
else
    echo "Analyzing specific experiments: $EXPERIMENTS"
    python ../python_scripts/analyze_experiment_results.py --results_dir "$RESULTS_DIR" --output_dir "$OUTPUT_DIR" --experiments $EXPERIMENTS
fi

if [ $? -ne 0 ]; then
    echo "ERROR: Analysis failed"
    read -p "Press Enter to continue..."
    exit 1
fi

echo ""
echo "========================================"
echo "Analysis completed successfully!"
echo "Results saved in: $OUTPUT_DIR"
echo "========================================"
read -p "Press Enter to continue..."
