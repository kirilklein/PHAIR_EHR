#!/bin/bash

# ========================================
# Run Experiments in Custom Order
# ========================================

if [ $# -eq 0 ]; then
    echo "Usage: ./run_experiments_ordered.sh experiment1 experiment2 experiment3 ..."
    echo ""
    echo "This script runs experiments in the exact order specified."
    echo "Each experiment will complete (successfully or with failure) before the next starts."
    echo ""
    echo "Available experiments:"
    for file in ../experiment_configs/*.yaml; do
        if [ -f "$file" ]; then
            filename=$(basename "$file" .yaml)
            echo "  - $filename"
        fi
    done
    echo ""
    echo "Examples:"
    echo "  ./run_experiments_ordered.sh ce0_cy0_y0_i0 ce0p5_cy0p5_y0_i0 ce1_cy1_y0_i0"
    echo "  ./run_experiments_ordered.sh ce0_cy0_y0_i5"
    echo ""
    echo "To run ALL experiments: ./run_all_experiments.sh"
    read -p "Press Enter to continue..."
    exit 1
fi

echo "========================================"
echo "Running Experiments in Custom Order"
echo "========================================"
echo ""

# Count experiments and build list
TOTAL_EXPERIMENTS=$#
EXPERIMENT_LIST=""
for arg in "$@"; do
    if [ -z "$EXPERIMENT_LIST" ]; then
        EXPERIMENT_LIST="$arg"
    else
        EXPERIMENT_LIST="$EXPERIMENT_LIST, $arg"
    fi
done

echo "Will run $TOTAL_EXPERIMENTS experiments in order: $EXPERIMENT_LIST"
echo ""

# Initialize tracking
SUCCESSFUL_EXPERIMENTS=0
FAILED_EXPERIMENTS=0
FAILED_LIST=""

# Get timestamp for logging
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="run_ordered_experiments_${timestamp}.log"

echo "Starting ordered run at $(date)"
echo "Logging to: $LOG_FILE"
echo ""

# Set batch mode to prevent pausing
export BATCH_MODE=true

# Run each experiment
CURRENT_EXPERIMENT=0
for experiment_name in "$@"; do
    ((CURRENT_EXPERIMENT++))
    
    echo "========================================"
    echo "Experiment $CURRENT_EXPERIMENT of $TOTAL_EXPERIMENTS: $experiment_name"
    echo "========================================"
    
    # Check if experiment config exists
    if [ ! -f "../experiment_configs/$experiment_name.yaml" ]; then
        echo "ERROR: Experiment config not found: ../experiment_configs/$experiment_name.yaml"
        echo "[$(date +"%H:%M:%S")] ERROR: Config not found for $experiment_name" >> "$LOG_FILE"
        ((FAILED_EXPERIMENTS++))
        if [ -z "$FAILED_LIST" ]; then
            FAILED_LIST="$experiment_name (config not found)"
        else
            FAILED_LIST="$FAILED_LIST, $experiment_name (config not found)"
        fi
        continue
    fi
    
    # Log start
    echo "[$(date +"%H:%M:%S")] Starting experiment: $experiment_name" >> "$LOG_FILE"
    
    # Run experiment
    ./run_experiment.sh "$experiment_name"
    experiment_result=$?
    
    # Log result
    if [ $experiment_result -eq 0 ]; then
        echo "[$(date +"%H:%M:%S")] SUCCESS: $experiment_name" >> "$LOG_FILE"
        echo "SUCCESS: Experiment $experiment_name completed successfully"
        ((SUCCESSFUL_EXPERIMENTS++))
    else
        echo "[$(date +"%H:%M:%S")] FAILED: $experiment_name (error code $experiment_result)" >> "$LOG_FILE"
        echo "FAILED: Experiment $experiment_name failed with error code $experiment_result"
        ((FAILED_EXPERIMENTS++))
        if [ -z "$FAILED_LIST" ]; then
            FAILED_LIST="$experiment_name"
        else
            FAILED_LIST="$FAILED_LIST, $experiment_name"
        fi
    fi
    
    echo ""
    if [ $CURRENT_EXPERIMENT -lt $TOTAL_EXPERIMENTS ]; then
        echo "Continuing to next experiment..."
        echo ""
        sleep 2
    fi
done

echo "========================================"
echo "ORDERED RUN SUMMARY"
echo "========================================"
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Successful: $SUCCESSFUL_EXPERIMENTS"
echo "Failed: $FAILED_EXPERIMENTS"
echo ""

if [ $FAILED_EXPERIMENTS -gt 0 ]; then
    echo "Failed experiments: $FAILED_LIST"
    echo ""
fi

echo "Results logged to: $LOG_FILE"
echo ""

# Log summary
{
    echo "========================================"
    echo "ORDERED RUN SUMMARY"
    echo "========================================"
    echo "Experiment order: $EXPERIMENT_LIST"
    echo "Total experiments: $TOTAL_EXPERIMENTS"
    echo "Successful: $SUCCESSFUL_EXPERIMENTS"
    echo "Failed: $FAILED_EXPERIMENTS"
    if [ $FAILED_EXPERIMENTS -gt 0 ]; then
        echo "Failed experiments: $FAILED_LIST"
    fi
} >> "$LOG_FILE"

if [ $FAILED_EXPERIMENTS -eq 0 ]; then
    echo "All experiments completed successfully!"
    read -p "Press Enter to continue..."
    exit 0
else
    echo "Some experiments failed. Check the log file for details."
    read -p "Press Enter to continue..."
    exit 1
fi
