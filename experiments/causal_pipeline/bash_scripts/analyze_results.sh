#!/bin/bash

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
    echo "                            Default: ../../../outputs/causal/sim_study/runs"
    echo "  --output_dir DIR          Directory to save analysis outputs"
    echo "  --output-dir DIR          (alternative: use hyphen instead of underscore)"
    echo "                            Default: ../../../outputs/causal/sim_study/analysis"
    echo "  --outcomes OUTCOME1 OUTCOME2  Filter analysis to specific outcomes only"
    echo "                            (e.g., --outcomes OUTCOME_1 OUTCOME_2)"
    echo "  --max-subplots N          Maximum subplots per figure (creates multiple figures if exceeded)"
    echo "                            (e.g., --max-subplots 4 creates 2x2 grids)"
    echo "  --help, -h                Show this help message"
    echo ""
    echo "EXPERIMENTS:"
    echo "  all                       Analyze all available experiments (default if none specified)"
    echo "  experiment_name           Specific experiment to analyze (e.g., ce0_cy0_y0_i0)"
    echo "  exp1 exp2 exp3           Multiple specific experiments"
    echo ""
    echo "EXAMPLES:"
    echo "  ./analyze_results.sh"
    echo "    → Analyze all experiments, aggregate across all runs"
    echo ""
    echo "  ./analyze_results.sh all"
    echo "    → Same as above (explicit 'all')"
    echo ""
    echo "  ./analyze_results.sh ce0_cy0_y0_i0 ce1_cy1_y0_i0"
    echo "    → Analyze specific experiments, aggregate across all runs"
    echo ""
    echo "  ./analyze_results.sh --run_id run_01 all"
    echo "    → Analyze all experiments from run_01 only"
    echo ""
    echo "  ./analyze_results.sh --run_id run_02 ce0_cy0_y0_i0"
    echo "    → Analyze specific experiment from run_02 only"
    echo ""
    echo "  ./analyze_results.sh --results_dir /custom/path --output_dir /custom/output"
    echo "    → Use custom input and output directories"
    echo ""
    echo "  ./analyze_results.sh --run_id run_01 --results_dir /custom/path ce0_cy0_y0_i0"
    echo "    → Combine multiple options"
    echo ""
    echo "  ./analyze_results.sh --outcomes OUTCOME_1 OUTCOME_2"
    echo "    → Analyze all experiments but only for specific outcomes"
    echo ""
    echo "  ./analyze_results.sh --run_id run_01 --outcomes OUTCOME_1"
    echo "    → Combine run filtering with outcome filtering"
    echo ""
    echo "NOTES:"
    echo "  • By default, results are aggregated across all runs (run_01, run_02, etc.)"
    echo "  • Use --run_id to analyze results from a specific run only"
    echo "  • When --run_id is used, it's appended to the results and output directories"
    echo "  • The script will create the output directory if it doesn't exist"
    echo ""
}

if [ $# -eq 0 ]; then
    echo "Running analysis of ALL experiments since no arguments provided..."
    echo "Use './analyze_results.sh --help' for more options."
    echo ""
    RUN_ALL=true
    RUN_ID=""
    RESULTS_DIR="../../../outputs/causal/sim_study/runs"
    OUTPUT_DIR="../../../outputs/causal/sim_study/analysis"
    OUTCOMES=""
    MAX_SUBPLOTS=""
else
    RUN_ALL=false
    RUN_ID=""
    RESULTS_DIR="../../../outputs/causal/sim_study/runs"
    OUTPUT_DIR="../../../outputs/causal/sim_study/analysis"
    OUTCOMES=""
    MAX_SUBPLOTS=""
fi

# Parse arguments (only if arguments were provided)
if [ "$RUN_ALL" = "false" ]; then
    EXPERIMENTS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --run_id|--run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --results_dir|--results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --output_dir|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --outcomes)
            shift  # Skip the --outcomes flag
            # Collect all outcomes until we hit another flag or end of args
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                if [ -z "$OUTCOMES" ]; then
                    OUTCOMES="$1"
                else
                    OUTCOMES="$OUTCOMES $1"
                fi
                shift
            done
            ;;
        --max-subplots)
            MAX_SUBPLOTS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
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

# Directories are now set as defaults above and can be overridden by arguments

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

# Build the Python command with optional outcomes and max-subplots arguments
PYTHON_CMD="python ../python_scripts/analyze_experiment_results.py --results_dir \"$RESULTS_DIR\" --output_dir \"$OUTPUT_DIR\""
if [ -n "$OUTCOMES" ]; then
    PYTHON_CMD="$PYTHON_CMD --outcomes $OUTCOMES"
    echo "Filtering analysis for outcomes: $OUTCOMES"
fi
if [ -n "$MAX_SUBPLOTS" ]; then
    PYTHON_CMD="$PYTHON_CMD --max-subplots $MAX_SUBPLOTS"
    echo "Using maximum $MAX_SUBPLOTS subplots per figure"
fi

if [ "$RUN_ALL" = "true" ]; then
    echo "Analyzing ALL experiments in $RESULTS_DIR"
    eval $PYTHON_CMD
elif [ "$EXPERIMENTS" = "all" ]; then
    echo "Analyzing ALL experiments in $RESULTS_DIR"
    eval $PYTHON_CMD
elif [ -z "$EXPERIMENTS" ]; then
    echo "Analyzing ALL experiments in $RESULTS_DIR"
    eval $PYTHON_CMD
else
    echo "WARNING: Specific experiment selection not supported by Python script. Analyzing ALL experiments in $RESULTS_DIR"
    eval $PYTHON_CMD
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
