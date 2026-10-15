#!/bin/bash -e
#SBATCH -c 8
#SBATCH -t 6-23
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL
# #SBATCH --mail-user=<your_email@example.com>
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Medium runtime CPU tools: sklearn methods
# No GPU resources requested
#
# Usage: sbatch run_medium_cpu_tools.sh [OPTIONS]
#        bash run_medium_cpu_tools.sh [OPTIONS]  # For testing without SLURM
#
# See benchmark_args_parser.sh or run with --help for options
#
# NOTE: This script was developed for the HUJI SLURM cluster. The lines below
# (module loading via Lmod, conda/mamba env names, SLURM resource flags) may
# need to be adapted for your own cluster.

source ~/.bashrc
. /etc/profile.d/huji-lmod.sh

# Directory containing this script and the benchmarking code
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the shared argument parser
source "$SCRIPT_DIR/benchmark_args_parser.sh"

# Change to script directory for relative paths
cd "$SCRIPT_DIR"

# Tool list for this script
MEDIUM_CPU_TOOLS=("random_forest" "xgboost" "logistic_regression" "svm")

# Parse arguments (pass script name and available tools)
parse_benchmark_args "$(basename "$0")" MEDIUM_CPU_TOOLS "$@"

LOG_ROOT="$SCRIPT_DIR/logs"
for tool in "${SELECTED_TOOLS[@]}"; do
    for task in "${SELECTED_TASKS[@]}"; do
        mkdir -p "$LOG_ROOT/$tool/$task"
    done
done

# Detect array job mode
if [[ -n "${SLURM_ARRAY_TASK_ID}" ]]; then
    ARRAY_MODE=true
    # In array mode, we must have exactly one tool
    if [[ ${#SELECTED_TOOLS[@]} -ne 1 ]]; then
        echo "Error: Array mode requires exactly one tool (use --tool)" >&2
        exit 1
    fi

    # Select task based on array index
    if [[ ${SLURM_ARRAY_TASK_ID} -ge ${#SELECTED_TASKS[@]} ]]; then
        echo "Error: Array index ${SLURM_ARRAY_TASK_ID} out of range (max: $((${#SELECTED_TASKS[@]} - 1)))" >&2
        exit 1
    fi

    TASK_TO_RUN="${SELECTED_TASKS[${SLURM_ARRAY_TASK_ID}]}"
    SELECTED_TASKS=("$TASK_TO_RUN")

    echo "════════════════════════════════════════════════════════════"
    echo "Array Job Mode: Task Index ${SLURM_ARRAY_TASK_ID}"
    echo "Running Tool: ${SELECTED_TOOLS[0]}"
    echo "Running Task: $TASK_TO_RUN"
    echo "════════════════════════════════════════════════════════════"
else
    ARRAY_MODE=false
    # Print configuration
    print_config
fi

# Constants
# EDIT THIS: path to the directory containing benchmark dataset CSVs
# (download from https://campp.yassourlab.com/), with files named
# <task>_train.csv / <task>_test.csv / <task>_test_gt.csv
BENCHMARK_DATASETS_DIR="/path/to/benchmark_datasets/"

# Helper function to log results to MLflow
log_to_mlflow() {
    local model_name=$1
    echo "Logging $model_name results to MLflow..."
    python -u log_results_to_mlflow.py "$task_name" "$model_name" default "$BENCHMARK_DATASETS_DIR"
    python -u log_results_to_mlflow.py "$task_name" "$model_name" optimized "$BENCHMARK_DATASETS_DIR"
}

# Helper function to check if tool is selected
is_tool_selected() {
    local tool=$1
    for selected in "${SELECTED_TOOLS[@]}"; do
        if [[ "$selected" == "$tool" ]]; then
            return 0
        fi
    done
    return 1
}

# Main execution loop over tasks
echo "════════════════════════════════════════════════════════════"
echo "Starting Medium CPU Benchmarking Tools"
echo "════════════════════════════════════════════════════════════"

for task_name in "${SELECTED_TASKS[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "TASK: $task_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Track if we need to activate/deactivate sklearn environment
    sklearn_tools_to_run=()
    for tool in "${MEDIUM_CPU_TOOLS[@]}"; do
        if is_tool_selected "$tool"; then
            sklearn_tools_to_run+=("$tool")
        fi
    done

    # Run sklearn methods if any are selected
    if [[ ${#sklearn_tools_to_run[@]} -gt 0 ]]; then
        echo ""
        echo "┌─────────────────────────────────────────────────────────┐"
        echo "│ Running sklearn methods                                 │"
        echo "└─────────────────────────────────────────────────────────┘"

        for tool in "${sklearn_tools_to_run[@]}"; do
            LOG_DIR="$LOG_ROOT/$tool/$task_name"
            LOG_FILE="$LOG_DIR/${tool}_${task_name}_${SLURM_JOB_ID}.out"
            ERR_FILE="$LOG_DIR/${tool}_${task_name}_${SLURM_JOB_ID}.err"
            (
                mamba activate benchmark_jupyter
                echo "→ Running $tool..."
                python -u sklearn_methods.py "$task_name" "$BENCHMARK_DATASETS_DIR" "$tool"
                log_to_mlflow "$tool"
                mamba deactivate
                echo "✓ $tool completed"
            ) >>"$LOG_FILE" 2>>"$ERR_FILE"
        done

        echo "✓ sklearn methods completed"
    fi

    echo ""
    echo "✓ Task $task_name completed"
done
