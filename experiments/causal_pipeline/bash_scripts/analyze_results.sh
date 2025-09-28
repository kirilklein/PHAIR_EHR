#!/bin/bash

# ========================================
# Analyze Causal Pipeline Experiment Results
# ========================================

if [ $# -eq 0 ]; then
    echo "Usage: ./analyze_results.sh [experiment1 experiment2 ...]"
    echo ""
    echo "Examples:"
    echo "  ./analyze_results.sh all                         (analyze all experiments)"
    echo "  ./analyze_results.sh ce0_cy0_y0_i0 ce1_cy1_y0_i0  (analyze specific experiments)"
    echo ""
    read -p "Press Enter to continue..."
    exit 1
fi

RESULTS_DIR="../../../outputs/causal/sim_study/runs"
OUTPUT_DIR="../../../outputs/causal/sim_study/analysis"

echo "========================================"
echo "Analyzing Experiment Results"
echo "========================================"
echo ""

if [ "$1" = "all" ]; then
    echo "Analyzing ALL experiments in $RESULTS_DIR"
    python ../python_scripts/analyze_experiment_results.py --results_dir "$RESULTS_DIR" --output_dir "$OUTPUT_DIR"
else
    echo "Analyzing specific experiments: $*"
    python ../python_scripts/analyze_experiment_results.py --results_dir "$RESULTS_DIR" --output_dir "$OUTPUT_DIR" --experiments "$@"
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
