#!/bin/bash

# ========================================
# Run Multiple Causal Pipeline Experiments
# ========================================

set -e  # Exit on error

RUN_MODE="both"
EXPERIMENT_LIST=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "========================================"
            echo "Run Multiple Causal Pipeline Experiments"
            echo "========================================"
            echo ""
            echo "Usage: $0 [OPTIONS] <experiment1> <experiment2> ..."
            echo ""
            echo "ARGUMENTS:"
            echo "  experiment1 experiment2 ...    Names of experiments to run"
            echo ""
            echo "OPTIONS:"
            echo "  -h, --help           Show this help message"
            echo "  --baseline-only      Run only baseline (CatBoost) pipeline for all experiments"
            echo "  --bert-only          Run only BERT pipeline for all experiments (requires baseline data)"
            echo "  (no options)         Run both baseline and BERT pipelines for all experiments"
            echo ""
            echo "AVAILABLE EXPERIMENTS:"
            if ls ../experiment_configs/*.yaml 1> /dev/null 2>&1; then
                for file in ../experiment_configs/*.yaml; do
                    if [ -f "$file" ]; then
                        filename=$(basename "$file" .yaml)
                        echo "  - $filename"
                    fi
                done
            else
                echo "  No experiment configs found"
            fi
            echo ""
            echo "EXAMPLES:"
            echo "  $0 ce1_cy1_y0_i0 ce0_cy0_y0_i0"
            echo "    > Runs both baseline and BERT pipelines for the specified experiments"
            echo ""
            echo "  $0 --baseline-only ce1_cy1_y0_i0 ce0_cy0_y0_i0"
            echo "    > Runs only baseline pipeline for the specified experiments"
            echo ""
            echo "  $0 --bert-only ce1_cy1_y0_i0"
            echo "    > Runs only BERT pipeline for the specified experiment (baseline data must exist)"
            echo ""
            echo "NOTES:"
            echo "  - Experiment configs are read from: ../experiment_configs/<experiment_name>.yaml"
            echo "  - Results are saved to: ../../../outputs/causal/sim_study/runs/<experiment_name>/"
            echo "  - Use Ctrl+C to stop the batch run at any time"
            echo ""
            exit 0
            ;;
        --baseline-only)
            RUN_MODE="baseline"
            shift
            ;;
        --bert-only)
            RUN_MODE="bert"
            shift
            ;;
        *)
            # Add experiment to list
            if [ -z "$EXPERIMENT_LIST" ]; then
                EXPERIMENT_LIST="$1"
            else
                EXPERIMENT_LIST="$EXPERIMENT_LIST $1"
            fi
            shift
            ;;
    esac
done

# Check if experiments were specified
if [ -z "$EXPERIMENT_LIST" ]; then
    echo "ERROR: No experiments specified"
    echo ""
    echo "Use -h or --help for usage information"
    exit 1
fi

echo "========================================"
echo "Running Multiple Causal Pipeline Experiments"

case $RUN_MODE in
    "both")
        echo "Mode: BASELINE + BERT"
        ;;
    "baseline")
        echo "Mode: BASELINE ONLY"
        ;;
    "bert")
        echo "Mode: BERT ONLY"
        ;;
esac
echo "========================================"

# Set batch mode to prevent pausing
export BATCH_MODE=true

FAILED_EXPERIMENTS=""
SUCCESS_COUNT=0
TOTAL_COUNT=0

# Count experiments first
for experiment in $EXPERIMENT_LIST; do
    ((TOTAL_COUNT++))
done

echo "Found $TOTAL_COUNT experiments to run: $EXPERIMENT_LIST"
echo ""

CURRENT_COUNT=0

# Run each experiment
for experiment in $EXPERIMENT_LIST; do
    EXPERIMENT_NAME="$experiment"
    ((CURRENT_COUNT++))

    echo ""
    echo "----------------------------------------"
    echo "Running experiment $CURRENT_COUNT of $TOTAL_COUNT: $EXPERIMENT_NAME"
    echo "----------------------------------------"

    # Call experiment with proper argument passing
    echo "DEBUG: Calling run_experiment.sh with experiment: $EXPERIMENT_NAME, mode: $RUN_MODE"
    
    case $RUN_MODE in
        "baseline")
            ./run_experiment.sh "$EXPERIMENT_NAME" --baseline-only
            ;;
        "bert")
            ./run_experiment.sh "$EXPERIMENT_NAME" --bert-only
            ;;
        *)
            ./run_experiment.sh "$EXPERIMENT_NAME"
            ;;
    esac
    
    experiment_result=$?

    if [ $experiment_result -ne 0 ]; then
        echo ""
        echo "ERROR: Experiment $EXPERIMENT_NAME failed with error code $experiment_result"
        echo "Check the output above for detailed error information."
        echo ""
        if [ -z "$FAILED_EXPERIMENTS" ]; then
            FAILED_EXPERIMENTS="$EXPERIMENT_NAME"
        else
            FAILED_EXPERIMENTS="$FAILED_EXPERIMENTS $EXPERIMENT_NAME"
        fi
    else
        echo "SUCCESS: Experiment $EXPERIMENT_NAME completed!"
        ((SUCCESS_COUNT++))
    fi
done

echo ""
echo "========================================"
echo "SUMMARY: Multiple Experiments Completed"
echo "========================================"
echo "Total experiments: $TOTAL_COUNT"
echo "Successful: $SUCCESS_COUNT"
FAILED_COUNT=$((TOTAL_COUNT - SUCCESS_COUNT))
echo "Failed: $FAILED_COUNT"

if [ -n "$FAILED_EXPERIMENTS" ]; then
    echo ""
    echo "Failed experiments: $FAILED_EXPERIMENTS"
    echo ""
    echo "Some experiments failed. Check the output above for details."
    exit 1
else
    echo ""
    echo "All experiments completed successfully!"
    exit 0
fi
