#!/bin/bash

# ========================================
# Run All Causal Pipeline Experiments Sequentially
# ========================================

echo "========================================"
echo "Running All Causal Pipeline Experiments"
echo "========================================"
echo ""

# Check if any experiment configs exist
if ! ls experiment_configs/*.yaml 1> /dev/null 2>&1; then
    echo "ERROR: No experiment configs found in experiment_configs/"
    echo ""
    echo "Create experiments with: ./create_new_experiment.sh <name>"
    read -p "Press Enter to continue..."
    exit 1
fi

# Initialize tracking variables
TOTAL_EXPERIMENTS=0
SUCCESSFUL_EXPERIMENTS=0
FAILED_EXPERIMENTS=0
FAILED_LIST=""

# Count total experiments
for file in experiment_configs/*.yaml; do
    if [ -f "$file" ]; then
        ((TOTAL_EXPERIMENTS++))
    fi
done

echo "Found $TOTAL_EXPERIMENTS experiments to run"
echo ""

# Get current timestamp for logging
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="run_all_experiments_${timestamp}.log"

echo "Starting batch run at $(date)"
echo "Logging to: $LOG_FILE"
echo ""

# Set batch mode to prevent pausing
export BATCH_MODE=true

# Run each experiment
CURRENT_EXPERIMENT=0
for file in experiment_configs/*.yaml; do
    if [ -f "$file" ]; then
        filename=$(basename "$file" .yaml)
        ((CURRENT_EXPERIMENT++))
        
        echo "========================================"
        echo "Experiment $CURRENT_EXPERIMENT of $TOTAL_EXPERIMENTS: $filename"
        echo "========================================"
        
        # Log start time
        start_time=$(date +"%H:%M:%S")
        echo "[$start_time] Starting experiment: $filename" >> "$LOG_FILE"
        
        # Run the experiment
        ./run_experiment.sh "$filename"
        experiment_result=$?
        
        # Log end time and result
        end_time=$(date +"%H:%M:%S")
        
        if [ $experiment_result -eq 0 ]; then
            echo "[$end_time] SUCCESS: $filename" >> "$LOG_FILE"
            echo "SUCCESS: Experiment $filename completed successfully"
            ((SUCCESSFUL_EXPERIMENTS++))
        else
            echo "[$end_time] FAILED: $filename" >> "$LOG_FILE"
            echo "FAILED: Experiment $filename failed with error code $experiment_result"
            ((FAILED_EXPERIMENTS++))
            if [ -z "$FAILED_LIST" ]; then
                FAILED_LIST="$filename"
            else
                FAILED_LIST="$FAILED_LIST, $filename"
            fi
        fi
        
        echo ""
        echo "Continuing to next experiment..."
        echo ""
        sleep 2
    fi
done

# Final summary
echo "========================================"
echo "BATCH RUN SUMMARY"
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

# Log final summary
{
    echo "========================================"
    echo "BATCH RUN SUMMARY"
    echo "========================================"
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
