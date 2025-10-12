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
EXPERIMENTS_DIR="./outputs/causal/sim_study_sampling/runs"

# Configurable data paths with defaults
MEDS_DATA="./example_data/synthea_meds_causal"
FEATURES_DATA="./outputs/causal/data/features"
TOKENIZED_DATA="./outputs/causal/data/tokenized"
PRETRAIN_MODEL="./outputs/causal/pretrain/model"

# Resampling-specific parameters
BASE_SEED=42
SAMPLE_FRACTION=0.5

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
            echo "  --base-seed N             Base seed for sampling (default: 42). Actual seed = base_seed + run_number"
            echo "  --sample-fraction F       Fraction of MEDS patients to sample per run (default: 0.5)"
            echo "  --meds PATH               Path to MEDS data directory (default: ./example_data/synthea_meds_causal)"
            echo "  --features PATH           Path to features directory (default: ./outputs/causal/data/features)"
            echo "  --tokenized PATH          Path to tokenized data directory (default: ./outputs/causal/data/tokenized)"
            echo "  --pretrain-model PATH     Path to pretrained BERT model (default: ./outputs/causal/pretrain/model)"
            echo "  --baseline-only           Run only baseline (CatBoost) pipeline for all experiments"
            echo "  --bert-only               Run only BERT pipeline for all experiments (requires baseline data)"
            echo "  --overwrite               Force re-run all steps (default: skip completed steps)"
            echo "  (no options)              Run both baseline and BERT pipelines for all experiments"
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
        -e|--experiment-dir)
            EXPERIMENTS_DIR="$2"
            shift 2
            ;;
        --base-seed)
            BASE_SEED="$2"
            shift 2
            ;;
        --sample-fraction)
            SAMPLE_FRACTION="$2"
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
        --pretrain-model)
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
if [ "$OVERWRITE" = "true" ]; then
    echo "Overwrite mode: ENABLED (re-run all steps)"
else
    echo "Overwrite mode: DISABLED (skip completed steps)"
fi
echo "========================================"

# Set batch mode to prevent pausing
export BATCH_MODE=true

FAILED_EXPERIMENTS=""
SUCCESS_COUNT=0
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

        # Add overwrite flag
        if [ "$OVERWRITE" = "true" ]; then
            EXPERIMENT_ARGS="$EXPERIMENT_ARGS --overwrite"
        fi

        # Add resampling-specific arguments
        EXPERIMENT_ARGS="$EXPERIMENT_ARGS --base-seed $BASE_SEED"
        EXPERIMENT_ARGS="$EXPERIMENT_ARGS --sample-fraction $SAMPLE_FRACTION"

        # Add data path arguments
        EXPERIMENT_ARGS="$EXPERIMENT_ARGS --meds \"$MEDS_DATA\""
        EXPERIMENT_ARGS="$EXPERIMENT_ARGS --features \"$FEATURES_DATA\""
        EXPERIMENT_ARGS="$EXPERIMENT_ARGS --tokenized \"$TOKENIZED_DATA\""
        EXPERIMENT_ARGS="$EXPERIMENT_ARGS --pretrain-model \"$PRETRAIN_MODEL\""

        # Run the experiment
        eval "./run_experiment.sh $EXPERIMENT_ARGS"

        experiment_result=$?

        if [ $experiment_result -ne 0 ]; then
            echo ""
            echo "ERROR: Experiment $RUN_ID/$EXPERIMENT_NAME failed with error code $experiment_result"
            echo "Check the output above for detailed error information."
            echo ""
            if [ -z "$FAILED_EXPERIMENTS" ]; then
                FAILED_EXPERIMENTS="$RUN_ID/$EXPERIMENT_NAME"
            else
                FAILED_EXPERIMENTS="$FAILED_EXPERIMENTS $RUN_ID/$EXPERIMENT_NAME"
            fi
        else
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
