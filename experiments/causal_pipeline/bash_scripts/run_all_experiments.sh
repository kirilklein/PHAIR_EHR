#!/bin/bash

# ========================================
# Run All Full Causal Pipeline Experiments (Baseline + BERT)
# ========================================

SKIP_EXISTING=false
RUN_MODE="both"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-existing|-s)
            SKIP_EXISTING=true
            shift
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
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-existing|-s] [--baseline-only|--bert-only]"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Running All Full Causal Pipeline Experiments"
if [ "$SKIP_EXISTING" = "true" ]; then
    echo "Mode: SKIPPING existing experiments"
else
    echo "Mode: Re-running ALL experiments"
fi

case $RUN_MODE in
    "both")
        echo "Pipeline: BASELINE + BERT"
        ;;
    "baseline")
        echo "Pipeline: BASELINE ONLY"
        ;;
    "bert")
        echo "Pipeline: BERT ONLY"
        ;;
esac
echo "========================================"
echo ""

# Check if any experiment configs exist
if ! ls ../experiment_configs/*.yaml 1> /dev/null 2>&1; then
    echo "ERROR: No experiment configs found in ../experiment_configs/"
    echo ""
    echo "Create experiments with: ./create_new_experiment.sh <name>"
    read -p "Press Enter to continue..."
    exit 1
fi

# Initialize tracking variables
TOTAL_EXPERIMENTS=0
SUCCESSFUL_EXPERIMENTS=0
FAILED_EXPERIMENTS=0
SKIPPED_EXPERIMENTS=0
FAILED_LIST=""

# Count total experiments
for file in ../experiment_configs/*.yaml; do
    if [ -f "$file" ]; then
        ((TOTAL_EXPERIMENTS++))
    fi
done

echo "Found $TOTAL_EXPERIMENTS experiments to run"
echo ""

# Get timestamp for logging
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
# Create logs directory if it doesn't exist
mkdir -p "../logs"

LOG_FILE="../logs/run_all_experiments_full_${timestamp}.log"

echo "Starting batch run at $(date)"
echo "Logging to: $LOG_FILE"
echo ""

# Set batch mode to prevent pausing
export BATCH_MODE=true

# Run each experiment
CURRENT_EXPERIMENT=0
for file in ../experiment_configs/*.yaml; do
    if [ -f "$file" ]; then
        filename=$(basename "$file" .yaml)
        ((CURRENT_EXPERIMENT++))
        
        # Check if experiment already exists and should be skipped
        if [ "$SKIP_EXISTING" = "true" ]; then
            SHOULD_SKIP=false
            
            # Check what exists based on run mode
            case $RUN_MODE in
                "baseline")
                    if [ -f "../../outputs/causal/sim_study/runs/$filename/estimate/baseline/estimate_results.csv" ] || 
                       [ -f "../../outputs/causal/sim_study/runs/$filename/estimate/estimate_results.csv" ]; then
                        SHOULD_SKIP=true
                    fi
                    ;;
                "bert")
                    if [ -f "../../outputs/causal/sim_study/runs/$filename/estimate/bert/estimate_results.csv" ]; then
                        SHOULD_SKIP=true
                    fi
                    ;;
                "both")
                    # Skip only if both exist
                    if { [ -f "../../outputs/causal/sim_study/runs/$filename/estimate/baseline/estimate_results.csv" ] || 
                         [ -f "../../outputs/causal/sim_study/runs/$filename/estimate/estimate_results.csv" ]; } &&
                       [ -f "../../outputs/causal/sim_study/runs/$filename/estimate/bert/estimate_results.csv" ]; then
                        SHOULD_SKIP=true
                    fi
                    ;;
            esac
            
            if [ "$SHOULD_SKIP" = "true" ]; then
                echo "========================================"
                echo "Experiment $CURRENT_EXPERIMENT of $TOTAL_EXPERIMENTS: $filename (SKIPPED - already exists)"
                echo "========================================"
                ((SKIPPED_EXPERIMENTS++))
                echo "[$(date +"%H:%M:%S")] SKIPPED: $filename (already exists)" >> "$LOG_FILE"
                continue
            fi
        fi
        
        echo "========================================"
        echo "Experiment $CURRENT_EXPERIMENT of $TOTAL_EXPERIMENTS: $filename"
        echo "========================================"
        
        # Log start time
        start_time=$(date +"%H:%M:%S")
        echo "[$start_time] Starting experiment: $filename" >> "$LOG_FILE"
        
        # Build command arguments
        EXPERIMENT_ARGS="$filename"
        case $RUN_MODE in
            "baseline")
                EXPERIMENT_ARGS="$EXPERIMENT_ARGS --baseline-only"
                ;;
            "bert")
                EXPERIMENT_ARGS="$EXPERIMENT_ARGS --bert-only"
                ;;
        esac
        
        # Run the experiment
        ./run_experiment.sh $EXPERIMENT_ARGS
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
if [ "$SKIP_EXISTING" = "true" ]; then
    echo "Skipped: $SKIPPED_EXPERIMENTS"
fi
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
    if [ "$SKIP_EXISTING" = "true" ]; then
        echo "Skipped: $SKIPPED_EXPERIMENTS"
    fi
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
