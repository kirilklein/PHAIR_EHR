#!/bin/bash

# ========================================
# Run Multiple RESAMPLING Causal Pipeline Experiments
# Each run samples a different subset of the base cohort
# ========================================

# Note: Removed 'set -e' to prevent silent failures during debugging

RUN_MODE="both"
EXPERIMENT_LIST=""
N_RUNS=1
RUN_ID_OVERRIDE=""
OVERWRITE=false  # Safe default: don't overwrite existing results
FAILFAST=false    # Default: continue on failure
EXPERIMENTS_DIR="./outputs/causal/sim_study_sampling/runs"
BASE_CONFIGS_DIR=""  # Empty means use default

# Configurable data paths with defaults
MEDS_DATA="./example_data/synthea_meds_causal"
FEATURES_DATA="./outputs/causal/data/features"
TOKENIZED_DATA="./outputs/causal/data/tokenized"
PRETRAIN_MODEL="./outputs/causal/pretrain/model"

# Resampling-specific parameters
BASE_SEED=42
SAMPLE_FRACTION=""   # Either fraction or size must be provided
SAMPLE_SIZE=""       # Takes precedence over fraction
TIMEOUT_FACTOR=""    # Optional timeout scaling factor

# Parse command line arguments
echo "DEBUG: Starting argument parsing with arguments: $*"
while [[ $# -gt 0 ]]; do
    echo "DEBUG: Processing argument: '$1'"
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
            echo "  -h, --help                Show this help message"
            echo "  --n_runs|-n N             Number of runs to execute (default: 1, creates run_01, run_02, etc.)"
            echo "  --run_id run_XX           Specific run ID to use (overrides --n_runs)"
            echo "  -e, --experiment-dir DIR  Base directory for experiments (default: ./outputs/causal/sim_study_sampling/runs)"
            echo "  --base-configs-dir DIR    Custom base configs directory (default: ../base_configs)"
            echo "  --base-seed N             Base seed for sampling (default: 42). Actual seed = base_seed + run_number"
            echo "  --sample-fraction F       Fraction of patients to sample (0 < F <= 1, mutually exclusive with --sample-size)"
            echo "  --sample-size N           Absolute number of patients to sample (takes precedence, mutually exclusive with --sample-fraction)"
            echo "  --timeout-factor F        Multiply all timeouts by this factor (e.g., 2.0 for 2x longer, default: 1.0)"
            echo "  --meds PATH               Path to MEDS data directory (default: ./example_data/synthea_meds_causal)"
            echo "  --features PATH           Path to features directory (default: ./outputs/causal/data/features)"
            echo "  --tokenized PATH          Path to tokenized data directory (default: ./outputs/causal/data/tokenized)"
            echo "  --pretrain-model PATH     Path to pretrained BERT model (default: ./outputs/causal/pretrain/model)"
            echo "  --baseline-only           Run only baseline (CatBoost) pipeline for all experiments"
            echo "  --bert-only               Run only BERT pipeline for all experiments (requires baseline data)"
            echo "  --overwrite               Force re-run all steps (default: skip completed steps)"
            echo "  --failfast                Stop immediately if any experiment fails (default: continue to next)"
            echo "  (no options)              Run both baseline and BERT pipelines for all experiments"
            echo ""
            echo "NOTE: Multi-word options support both hyphen and underscore (e.g., --timeout-factor or --timeout_factor)"
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
            echo "  $0 --n_runs 3 ce1_cy1_y0_i0 ce0_cy0_y0_i0"
            echo "    > Runs specified experiments 3 times each, creating run_01, run_02, run_03"
            echo ""
            echo "  $0 --run_id run_05 ce1_cy1_y0_i0"
            echo "    > Runs specified experiment in run_05 folder specifically"
            echo ""
            echo "  $0 --n_runs 100 my_experiment"
            echo "    > Runs my_experiment 100 times, automatically resuming from where it left off if interrupted"
            echo ""
            echo "  $0 --n_runs 100 --overwrite my_experiment"
            echo "    > Runs my_experiment 100 times, forcing re-run of all steps"
            echo ""
            echo "  $0 --n_runs 100 --failfast my_experiment"
            echo "    > Runs my_experiment 100 times, stopping immediately if any run fails"
            echo ""
            echo "NOTES:"
            echo "  - This is the RESAMPLING variant: samples from MEDS → simulates → selects cohort for each run"
            echo "  - No pre-built cohort needed; sampling happens from raw MEDS data"
            echo "  - Seed varies per run: base_seed + run_number"
            echo "  - Experiment configs are read from: ../experiment_configs/<experiment_name>.yaml"
            echo "  - Results are saved to: ../../../outputs/causal/sim_study_sampling/runs/run_XX/<experiment_name>/"
            echo "  - Use Ctrl+C to stop the batch run at any time"
            echo ""
            exit 0
            ;;
        --n_runs|-n)
            N_RUNS="$2"
            if ! [[ "$N_RUNS" =~ ^[0-9]+$ ]] || [ "$N_RUNS" -lt 1 ]; then
                echo "ERROR: --n_runs must be a positive integer, got: $N_RUNS"
                exit 1
            fi
            shift 2
            ;;
        --run_id)
            RUN_ID_OVERRIDE="$2"
            if [ -z "$RUN_ID_OVERRIDE" ]; then
                echo "ERROR: --run_id requires a run ID (e.g., run_01)"
                exit 1
            fi
            shift 2
            ;;
        --baseline-only)
            echo "DEBUG: Setting RUN_MODE to baseline"
            RUN_MODE="baseline"
            shift
            ;;
        --bert-only)
            echo "DEBUG: Setting RUN_MODE to bert"
            RUN_MODE="bert"
            shift
            ;;
        --overwrite)
            OVERWRITE=true
            shift
            ;;
        --failfast)
            FAILFAST=true
            shift
            ;;
        -e|--experiment-dir|--experiment_dir)
            EXPERIMENTS_DIR="$2"
            shift 2
            ;;
        --base-configs-dir|--base_configs_dir)
            BASE_CONFIGS_DIR="$2"
            shift 2
            ;;
        --base-seed|--base_seed)
            BASE_SEED="$2"
            shift 2
            ;;
        --sample-fraction|--sample_fraction)
            SAMPLE_FRACTION="$2"
            shift 2
            ;;
        --sample-size|--sample_size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --timeout-factor|--timeout_factor)
            TIMEOUT_FACTOR="$2"
            echo "DEBUG: TIMEOUT_FACTOR set to $TIMEOUT_FACTOR"
            shift 2
            ;;
        --meds)
            MEDS_DATA="$2"
            shift 2
            ;;
        --features)
            FEATURES_DATA="$2"
            shift 2
            ;;
        --tokenized)
            TOKENIZED_DATA="$2"
            shift 2
            ;;
        --pretrain-model|--pretrain_model)
            PRETRAIN_MODEL="$2"
            shift 2
            ;;
        *)
            # Add experiment to list
            echo "DEBUG: Adding '$1' to experiment list"
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

# Set up logging
mkdir -p "$EXPERIMENTS_DIR"
mkdir -p "../logs"  # Create logs directory for per-experiment logs
LOG_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$EXPERIMENTS_DIR/batch_run_${LOG_TIMESTAMP}.log"
echo "Log file: $LOG_FILE"
echo "========================================"

# Redirect all output (stdout and stderr) to both terminal and log file
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo ""
echo "========================================"
echo "Logging initialized at $(date)"
echo "Log file: $LOG_FILE"

# Determine run configuration
if [ -n "$RUN_ID_OVERRIDE" ]; then
    echo "Run mode: Specific run ($RUN_ID_OVERRIDE)"
    N_RUNS=1
else
    echo "Number of runs: $N_RUNS"
    if [ "$N_RUNS" -gt 1 ]; then
        echo "Will create: run_01 through run_$(printf '%02d' $N_RUNS)"
    else
        echo "Will create: run_01"
    fi
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

echo "Experiments directory: $EXPERIMENTS_DIR"
echo "Base seed: $BASE_SEED"
echo "Sample fraction: $SAMPLE_FRACTION"
if [ -n "$TIMEOUT_FACTOR" ]; then
    echo "Timeout factor: $TIMEOUT_FACTOR"
fi
if [ "$OVERWRITE" = "true" ]; then
    echo "Overwrite mode: ENABLED (re-run all steps)"
else
    echo "Overwrite mode: DISABLED (skip completed steps)"
fi
if [ "$FAILFAST" = "true" ]; then
    echo "Failfast mode: ENABLED (stop on first failure)"
else
    echo "Failfast mode: DISABLED (continue on failures)"
fi
echo "========================================"

# Set batch mode to prevent pausing
export BATCH_MODE=true

FAILED_EXPERIMENTS=""
TIMEOUT_EXPERIMENTS=""
SUCCESS_COUNT=0
TIMEOUT_COUNT=0
TOTAL_COUNT=0

# Count experiments first
EXPERIMENT_COUNT=0
echo "DEBUG: EXPERIMENT_LIST='$EXPERIMENT_LIST'"
for experiment in $EXPERIMENT_LIST; do
    echo "DEBUG: Processing experiment: '$experiment'"
    ((EXPERIMENT_COUNT++))
done

TOTAL_COUNT=$((EXPERIMENT_COUNT * N_RUNS))
echo "Found $EXPERIMENT_COUNT experiments × $N_RUNS runs = $TOTAL_COUNT total experiments to run"
echo "Experiments: $EXPERIMENT_LIST"
echo ""

echo "DEBUG: About to start main loop with N_RUNS=$N_RUNS"

CURRENT_COUNT=0

# Run each experiment for each run
for run_number in $(seq 1 $N_RUNS); do
    # Determine run ID
    if [ -n "$RUN_ID_OVERRIDE" ]; then
        RUN_ID="$RUN_ID_OVERRIDE"
    else
        RUN_ID=$(printf "run_%02d" $run_number)
    fi

    echo ""
    echo "========================================="
    echo "STARTING RUN $run_number of $N_RUNS: $RUN_ID"
    echo "========================================="
    echo ""

    for experiment in $EXPERIMENT_LIST; do
        EXPERIMENT_NAME="$experiment"
        ((CURRENT_COUNT++))

        echo ""
        echo "----------------------------------------"
        echo "Running experiment $CURRENT_COUNT of $TOTAL_COUNT: $RUN_ID/$EXPERIMENT_NAME"
        echo "----------------------------------------"

        # Log start time
        start_time=$(date +"%H:%M:%S")
        echo "" >> "$LOG_FILE"
        echo "========================================" >> "$LOG_FILE"
        echo "[$start_time] STARTING: $RUN_ID/$EXPERIMENT_NAME" >> "$LOG_FILE"
        echo "========================================" >> "$LOG_FILE"

        # Call experiment with proper argument passing
        echo "DEBUG: Calling run_experiment.sh with experiment: $EXPERIMENT_NAME, run_id: $RUN_ID, mode: $RUN_MODE"

        # Build command arguments
        EXPERIMENT_ARGS="$EXPERIMENT_NAME --run_id $RUN_ID"

        case $RUN_MODE in
            "baseline")
                EXPERIMENT_ARGS="$EXPERIMENT_ARGS --baseline-only"
                ;;
            "bert")
                EXPERIMENT_ARGS="$EXPERIMENT_ARGS --bert-only"
                ;;
        esac

        # Add experiment directory
        EXPERIMENT_ARGS="$EXPERIMENT_ARGS --experiment-dir \"$EXPERIMENTS_DIR\""

        # Add base configs directory if specified
        if [ -n "$BASE_CONFIGS_DIR" ]; then
            EXPERIMENT_ARGS="$EXPERIMENT_ARGS --base-configs-dir \"$BASE_CONFIGS_DIR\""
        fi

        # Add overwrite flag
        if [ "$OVERWRITE" = "true" ]; then
            EXPERIMENT_ARGS="$EXPERIMENT_ARGS --overwrite"
        fi

        # Add resampling-specific arguments
        EXPERIMENT_ARGS="$EXPERIMENT_ARGS --base-seed $BASE_SEED"
        if [ -n "$SAMPLE_SIZE" ]; then
            EXPERIMENT_ARGS="$EXPERIMENT_ARGS --sample-size $SAMPLE_SIZE"
        elif [ -n "$SAMPLE_FRACTION" ]; then
            EXPERIMENT_ARGS="$EXPERIMENT_ARGS --sample-fraction $SAMPLE_FRACTION"
        fi
        
        # Add timeout factor if specified
        if [ -n "$TIMEOUT_FACTOR" ]; then
            EXPERIMENT_ARGS="$EXPERIMENT_ARGS --timeout-factor $TIMEOUT_FACTOR"
            echo "DEBUG: Adding timeout-factor to args: $TIMEOUT_FACTOR"
        fi

        # Add data path arguments
        EXPERIMENT_ARGS="$EXPERIMENT_ARGS --meds \"$MEDS_DATA\""
        EXPERIMENT_ARGS="$EXPERIMENT_ARGS --features \"$FEATURES_DATA\""
        EXPERIMENT_ARGS="$EXPERIMENT_ARGS --tokenized \"$TOKENIZED_DATA\""
        EXPERIMENT_ARGS="$EXPERIMENT_ARGS --pretrain-model \"$PRETRAIN_MODEL\""

        # Run the experiment and capture output
        EXPERIMENT_LOG_FILE="../logs/experiment_${RUN_ID}_${EXPERIMENT_NAME}_${LOG_TIMESTAMP}.log"
        eval "./run_experiment.sh $EXPERIMENT_ARGS" 2>&1 | tee "$EXPERIMENT_LOG_FILE" | while IFS= read -r line; do
            # Log key steps to main log file
            if [[ "$line" =~ "==== Running" ]] || [[ "$line" =~ "==== Skipping" ]] || [[ "$line" =~ "ERROR" ]] || [[ "$line" =~ "FAILED" ]] || [[ "$line" =~ "timed out" ]] || [[ "$line" =~ "TIMEOUT" ]]; then
                echo "[$(date +"%H:%M:%S")] $line" >> "$LOG_FILE"
            fi
        done
        experiment_result=${PIPESTATUS[0]}

        # Log end time and result
        end_time=$(date +"%H:%M:%S")

        if [ $experiment_result -ne 0 ]; then
            # Check if this was a timeout (exit code 124)
            if [ $experiment_result -eq 124 ]; then
                echo "[$end_time] ⏱ TIMEOUT: $RUN_ID/$EXPERIMENT_NAME (exceeded time limit)" >> "$LOG_FILE"
                echo "[$end_time] Full log: $EXPERIMENT_LOG_FILE" >> "$LOG_FILE"
                echo ""
                echo "========================================"
                echo "⏱ TIMEOUT: Experiment $RUN_ID/$EXPERIMENT_NAME exceeded time limit"
                echo "========================================"
                echo "Exit code: 124 (timeout)"
                echo "Full log available at: $EXPERIMENT_LOG_FILE"
                echo ""
                
                # Track timeouts separately
                ((TIMEOUT_COUNT++))
                if [ -z "$TIMEOUT_EXPERIMENTS" ]; then
                    TIMEOUT_EXPERIMENTS="$RUN_ID/$EXPERIMENT_NAME"
                else
                    TIMEOUT_EXPERIMENTS="$TIMEOUT_EXPERIMENTS, $RUN_ID/$EXPERIMENT_NAME"
                fi
                
                # Also add to failed experiments list for failfast handling
                if [ -z "$FAILED_EXPERIMENTS" ]; then
                    FAILED_EXPERIMENTS="$RUN_ID/$EXPERIMENT_NAME (timeout)"
                else
                    FAILED_EXPERIMENTS="$FAILED_EXPERIMENTS, $RUN_ID/$EXPERIMENT_NAME (timeout)"
                fi
            else
                # Regular failure (non-timeout)
                echo "[$end_time] ✗ FAILED: $RUN_ID/$EXPERIMENT_NAME (exit code: $experiment_result)" >> "$LOG_FILE"
                echo "[$end_time] Full log: $EXPERIMENT_LOG_FILE" >> "$LOG_FILE"
                echo ""
                echo "ERROR: Experiment $RUN_ID/$EXPERIMENT_NAME failed with error code $experiment_result"
                echo "Full log available at: $EXPERIMENT_LOG_FILE"
                echo ""
                
                if [ -z "$FAILED_EXPERIMENTS" ]; then
                    FAILED_EXPERIMENTS="$RUN_ID/$EXPERIMENT_NAME"
                else
                    FAILED_EXPERIMENTS="$FAILED_EXPERIMENTS, $RUN_ID/$EXPERIMENT_NAME"
                fi
            fi

            # Check if failfast is enabled
            if [ "$FAILFAST" = "true" ]; then
                echo ""
                echo "========================================"
                echo "FAILFAST MODE: Stopping due to failure"
                echo "========================================"
                echo "Failed experiment: $RUN_ID/$EXPERIMENT_NAME"
                echo "Error code: $experiment_result"
                if [ $experiment_result -eq 124 ]; then
                    echo "Failure type: TIMEOUT"
                fi
                echo ""
                echo "To resume, run without --failfast or use --overwrite"
                echo ""

                # Print summary and exit
                FAILED_COUNT=$((CURRENT_COUNT - SUCCESS_COUNT))
                echo "========================================"
                echo "SUMMARY: Batch Run Stopped (INCOMPLETE)"
                echo "========================================"
                echo "Total experiments planned: $TOTAL_COUNT"
                echo "Completed: $CURRENT_COUNT"
                echo "Successful: $SUCCESS_COUNT"
                echo "Failed: $FAILED_COUNT"
                if [ $TIMEOUT_COUNT -gt 0 ]; then
                    echo "Timeouts: $TIMEOUT_COUNT"
                fi
                echo ""
                echo "Failed experiments: $FAILED_EXPERIMENTS"
                if [ -n "$TIMEOUT_EXPERIMENTS" ]; then
                    echo "Timed out experiments: $TIMEOUT_EXPERIMENTS"
                fi
                echo ""
                echo "========================================"
                echo "Logging completed at $(date)"
                echo "Full log saved to: $LOG_FILE"
                echo "========================================"
                exit 1
            fi
        else
            echo "[$end_time] ✓ SUCCESS: $RUN_ID/$EXPERIMENT_NAME" >> "$LOG_FILE"
            echo "[$end_time] Full log: $EXPERIMENT_LOG_FILE" >> "$LOG_FILE"
            echo "SUCCESS: Experiment $RUN_ID/$EXPERIMENT_NAME completed!"
            ((SUCCESS_COUNT++))
        fi
    done

    echo ""
    echo "========================================="
    echo "COMPLETED RUN $run_number of $N_RUNS: $RUN_ID"
    echo "========================================="
    echo ""
done

echo ""
echo "========================================"
echo "SUMMARY: Multiple Experiments Completed"
echo "========================================"
echo "Total experiments: $TOTAL_COUNT"
echo "Successful: $SUCCESS_COUNT"
FAILED_COUNT=$((TOTAL_COUNT - SUCCESS_COUNT))
echo "Failed: $FAILED_COUNT"
if [ $TIMEOUT_COUNT -gt 0 ]; then
    echo "  - Timeouts: $TIMEOUT_COUNT"
    echo "  - Other failures: $((FAILED_COUNT - TIMEOUT_COUNT))"
fi

if [ -n "$FAILED_EXPERIMENTS" ]; then
    echo ""
    echo "Failed experiments: $FAILED_EXPERIMENTS"
    if [ -n "$TIMEOUT_EXPERIMENTS" ]; then
        echo ""
        echo "Timed out experiments specifically:"
        echo "  $TIMEOUT_EXPERIMENTS"
    fi
    echo ""
    echo "Some experiments failed. Check the output above for details."
    echo ""
    echo "========================================"
    echo "Logging completed at $(date)"
    echo "Full log saved to: $LOG_FILE"
    echo "========================================"
    exit 1
else
    echo ""
    echo "All experiments completed successfully!"
    echo ""
    echo "========================================"
    echo "Logging completed at $(date)"
    echo "Full log saved to: $LOG_FILE"
    echo "========================================"
    exit 0
fi
