#!/bin/bash

# ========================================
# Run All Full Causal Pipeline Experiments (Baseline + BERT)
# ========================================

SKIP_EXISTING=false
RUN_MODE="both"
N_RUNS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "========================================"
            echo "Run All Full Causal Pipeline Experiments"
            echo "========================================"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "OPTIONS:"
            echo "  -h, --help           Show this help message"
            echo "  -s, --skip-existing  Skip experiments that already have results"
            echo "  --n_runs N           Number of runs to execute (default: 1, creates run_01, run_02, etc.)"
            echo "  --baseline-only      Run only baseline (CatBoost) pipeline for all experiments"
            echo "  --bert-only          Run only BERT pipeline for all experiments (requires baseline data)"
            echo "  (no options)         Run both baseline and BERT pipelines for all experiments"
            echo ""
            echo "EXAMPLES:"
            echo "  $0"
            echo "    > Runs all experiments with both baseline and BERT pipelines"
            echo ""
            echo "  $0 --skip-existing"
            echo "    > Runs all experiments, skipping those that already have complete results"
            echo ""
            echo "  $0 --baseline-only -s"
            echo "    > Runs only baseline pipeline for all experiments, skipping existing ones"
            echo ""
            echo "  $0 --bert-only"
            echo "    > Runs only BERT pipeline for all experiments (baseline data must exist)"
            echo ""
            echo "  $0 --n_runs 5"
            echo "    > Runs all experiments 5 times, creating run_01 through run_05"
            echo ""
            echo "  $0 --n_runs 3 --baseline-only -s"
            echo "    > Runs only baseline pipeline 3 times, skipping existing runs"
            echo ""
            echo "NOTES:"
            echo "  - Experiment configs are read from: ../experiment_configs/*.yaml"
            echo "  - Results are saved to: ../../../outputs/causal/sim_study/runs/run_XX/<experiment_name>/"
            echo "  - Log files are saved to: ../logs/run_all_experiments_full_YYYY-MM-DD_HH-MM-SS.log"
            echo "  - Use Ctrl+C to stop the batch run at any time"
            echo ""
            exit 0
            ;;
        --skip-existing|-s)
            SKIP_EXISTING=true
            shift
            ;;
        --n_runs)
            N_RUNS="$2"
            if ! [[ "$N_RUNS" =~ ^[0-9]+$ ]] || [ "$N_RUNS" -lt 1 ]; then
                echo "ERROR: --n_runs must be a positive integer, got: $N_RUNS"
                exit 1
            fi
            shift 2
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
            echo "Usage: $0 [--skip-existing|-s] [--n_runs N] [--baseline-only|--bert-only] [-h|--help]"
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

echo "Number of runs: $N_RUNS"
if [ "$N_RUNS" -gt 1 ]; then
    echo "Will create: run_01 through run_$(printf '%02d' $N_RUNS)"
else
    echo "Will create: run_01"
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
EXPERIMENTS_COUNT=0
for file in ../experiment_configs/*.yaml; do
    if [ -f "$file" ]; then
        ((EXPERIMENTS_COUNT++))
    fi
done

TOTAL_EXPERIMENTS=$((EXPERIMENTS_COUNT * N_RUNS))
echo "Found $EXPERIMENTS_COUNT unique experiments × $N_RUNS runs = $TOTAL_EXPERIMENTS total runs"
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

# Run each experiment for each run
CURRENT_EXPERIMENT=0
for run_number in $(seq 1 $N_RUNS); do
    RUN_ID=$(printf "run_%02d" $run_number)
    echo ""
    echo "========================================="
    echo "STARTING RUN $run_number of $N_RUNS: $RUN_ID"
    echo "========================================="
    echo ""
    
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
                        if [ -f "../../../outputs/causal/sim_study/runs/$RUN_ID/$filename/estimate/baseline/estimate_results.csv" ] || 
                           [ -f "../../../outputs/causal/sim_study/runs/$RUN_ID/$filename/estimate/estimate_results.csv" ]; then
                            SHOULD_SKIP=true
                        fi
                        ;;
                    "bert")
                        if [ -f "../../../outputs/causal/sim_study/runs/$RUN_ID/$filename/estimate/bert/estimate_results.csv" ]; then
                            SHOULD_SKIP=true
                        fi
                        ;;
                    "both")
                        # Skip only if both exist
                        if { [ -f "../../../outputs/causal/sim_study/runs/$RUN_ID/$filename/estimate/baseline/estimate_results.csv" ] || 
                             [ -f "../../../outputs/causal/sim_study/runs/$RUN_ID/$filename/estimate/estimate_results.csv" ]; } &&
                           [ -f "../../../outputs/causal/sim_study/runs/$RUN_ID/$filename/estimate/bert/estimate_results.csv" ]; then
                            SHOULD_SKIP=true
                        fi
                        ;;
                esac
                
                if [ "$SHOULD_SKIP" = "true" ]; then
                    echo "========================================"
                    echo "Experiment $CURRENT_EXPERIMENT of $TOTAL_EXPERIMENTS: $RUN_ID/$filename (SKIPPED - already exists)"
                    echo "========================================"
                    ((SKIPPED_EXPERIMENTS++))
                    echo "[$(date +"%H:%M:%S")] SKIPPED: $RUN_ID/$filename (already exists)" >> "$LOG_FILE"
                    continue
                fi
            fi
            
            echo "========================================"
            echo "Experiment $CURRENT_EXPERIMENT of $TOTAL_EXPERIMENTS: $RUN_ID/$filename"
            echo "========================================"
            
            # Log start time
            start_time=$(date +"%H:%M:%S")
            echo "[$start_time] Starting experiment: $RUN_ID/$filename" >> "$LOG_FILE"
            
            # Build command arguments
            EXPERIMENT_ARGS="$filename --run_id $RUN_ID"
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
                echo "[$end_time] SUCCESS: $RUN_ID/$filename" >> "$LOG_FILE"
                echo "SUCCESS: Experiment $RUN_ID/$filename completed successfully"
                ((SUCCESSFUL_EXPERIMENTS++))
            else
                echo "[$end_time] FAILED: $RUN_ID/$filename" >> "$LOG_FILE"
                echo "FAILED: Experiment $RUN_ID/$filename failed with error code $experiment_result"
                ((FAILED_EXPERIMENTS++))
                if [ -z "$FAILED_LIST" ]; then
                    FAILED_LIST="$RUN_ID/$filename"
                else
                    FAILED_LIST="$FAILED_LIST, $RUN_ID/$filename"
                fi
            fi
            
            echo ""
            echo "Continuing to next experiment..."
            echo ""
            sleep 2
        fi
    done
    
    echo ""
    echo "========================================="
    echo "COMPLETED RUN $run_number of $N_RUNS: $RUN_ID"
    echo "========================================="
    echo ""
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
