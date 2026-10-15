#!/bin/bash -e
#SBATCH -c 1
#SBATCH -t 6-23:59:00
#SBATCH --mem=8G
#SBATCH --mail-type=FAIL
# #SBATCH --mail-user=<your_email@example.com>
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Long runtime CPU tools: SIAMCAT + DEBIAS-M
# No GPU resources requested
#
# Usage: sbatch run_long_cpu_tools.sh [OPTIONS]
#        bash run_long_cpu_tools.sh [OPTIONS]  # For testing without SLURM
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
LONG_CPU_TOOLS=("siamcat" "debias_m")

# Parse arguments (pass script name and available tools)
parse_benchmark_args "$(basename "$0")" LONG_CPU_TOOLS "$@"

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
LOG_ROOT="$SCRIPT_DIR/logs"

# SLURM settings for per-configuration jobs
CONFIG_JOB_CPUS=1
CONFIG_JOB_MEM="4G"
CONFIG_JOB_TIME="3-00:00:00"
# Add your own node exclusions here if needed, e.g. CONFIG_JOB_EXCLUDE="node01,node02"
CONFIG_JOB_EXCLUDE=""
EXTRA_SBATCH_ARGS=()
if [[ -n "$CONFIG_JOB_EXCLUDE" ]]; then
    EXTRA_SBATCH_ARGS+=(--exclude="$CONFIG_JOB_EXCLUDE")
fi

# Helper function to log results to MLflow
log_to_mlflow() {
    local model_name=$1
    local run_type=$2
    echo "Logging $model_name ($run_type) results to MLflow..."
    python -u log_results_to_mlflow.py "$task_name" "$model_name" "$run_type" "$BENCHMARK_DATASETS_DIR"
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

get_siamcat_config_count() {
    local run_type=$1
    module load R4/4.4.1
    Rscript siamcat_train_and_predict_flow.R "$task_name" "$BENCHMARK_DATASETS_DIR" \
        --mode list-configs --run-type "$run_type" | wc -l
}

get_debiasm_config_count() {
    local run_type=$1
    mamba run -n benchmark_jupyter \
        python -u debiasm_method.py "$task_name" "$BENCHMARK_DATASETS_DIR" \
        --mode list-configs --run-type "$run_type" | wc -l
}

submit_siamcat_jobs() {
    local run_type=$1
    local config_count=$2
    local array_range="0-$((config_count - 1))"

    local array_job_id
    array_job_id=$(sbatch --parsable \
        --array="$array_range" \
        --cpus-per-task="$CONFIG_JOB_CPUS" \
        --mem="$CONFIG_JOB_MEM" \
        --time="$CONFIG_JOB_TIME" \
        "${EXTRA_SBATCH_ARGS[@]}" \
        --output="$LOG_DIR/siamcat_${task_name}_${run_type}_%A_%a.out" \
        --error="$LOG_DIR/siamcat_${task_name}_${run_type}_%A_%a.err" \
        --job-name="siamcat_${task_name}_${run_type}" \
        --wrap ". ~/.bashrc; . /etc/profile.d/huji-lmod.sh; cd '$SCRIPT_DIR'; module load R4/4.4.1; Rscript siamcat_train_and_predict_flow.R '$task_name' '$BENCHMARK_DATASETS_DIR' --mode run-config --run-type '$run_type' --config-index \$SLURM_ARRAY_TASK_ID")

    echo "Submitted SIAMCAT $run_type array: $array_job_id (configs=$config_count, range=$array_range)"

    sbatch --dependency=afterok:"$array_job_id" \
        --cpus-per-task="$CONFIG_JOB_CPUS" \
        --mem="$CONFIG_JOB_MEM" \
        --time="$CONFIG_JOB_TIME" \
        "${EXTRA_SBATCH_ARGS[@]}" \
        --output="$LOG_DIR/siamcat_${task_name}_${run_type}_agg_%A.out" \
        --error="$LOG_DIR/siamcat_${task_name}_${run_type}_agg_%A.err" \
        --job-name="siamcat_${task_name}_${run_type}_agg" \
        --wrap ". ~/.bashrc; . /etc/profile.d/huji-lmod.sh; cd '$SCRIPT_DIR'; module load R4/4.4.1; Rscript siamcat_train_and_predict_flow.R '$task_name' '$BENCHMARK_DATASETS_DIR' --mode aggregate --run-type '$run_type'; mamba run -n benchmark_jupyter python -u log_results_to_mlflow.py '$task_name' siamcat '$run_type' '$BENCHMARK_DATASETS_DIR'"
}

submit_siamcat_aggregate_only_job() {
    local run_type=$1
    sbatch \
        --cpus-per-task="$CONFIG_JOB_CPUS" \
        --mem="$CONFIG_JOB_MEM" \
        --time="$CONFIG_JOB_TIME" \
        "${EXTRA_SBATCH_ARGS[@]}" \
        --output="$LOG_DIR/siamcat_${task_name}_${run_type}_agg_%A.out" \
        --error="$LOG_DIR/siamcat_${task_name}_${run_type}_agg_%A.err" \
        --job-name="siamcat_${task_name}_${run_type}_agg" \
        --wrap ". ~/.bashrc; . /etc/profile.d/huji-lmod.sh; cd '$SCRIPT_DIR'; module load R4/4.4.1; Rscript siamcat_train_and_predict_flow.R '$task_name' '$BENCHMARK_DATASETS_DIR' --mode aggregate --run-type '$run_type'; mamba run -n benchmark_jupyter python -u log_results_to_mlflow.py '$task_name' siamcat '$run_type' '$BENCHMARK_DATASETS_DIR'"
}

submit_debiasm_jobs() {
    local run_type=$1
    local config_count=$2
    local array_range="0-$((config_count - 1))"

    local array_job_id
    array_job_id=$(sbatch --parsable \
        --array="$array_range" \
        --cpus-per-task="$CONFIG_JOB_CPUS" \
        --mem="$CONFIG_JOB_MEM" \
        --time="$CONFIG_JOB_TIME" \
        "${EXTRA_SBATCH_ARGS[@]}" \
        --output="$LOG_DIR/debias_m_${task_name}_${run_type}_%A_%a.out" \
        --error="$LOG_DIR/debias_m_${task_name}_${run_type}_%A_%a.err" \
        --job-name="debias_m_${task_name}_${run_type}" \
        --wrap ". ~/.bashrc; cd '$SCRIPT_DIR'; mamba run -n benchmark_jupyter python -u debiasm_method.py '$task_name' '$BENCHMARK_DATASETS_DIR' --mode run-config --run-type '$run_type' --config-index \$SLURM_ARRAY_TASK_ID")

    echo "Submitted DEBIAS-M $run_type array: $array_job_id (configs=$config_count, range=$array_range)"

    sbatch --dependency=afterok:"$array_job_id" \
        --cpus-per-task="$CONFIG_JOB_CPUS" \
        --mem="$CONFIG_JOB_MEM" \
        --time="$CONFIG_JOB_TIME" \
        "${EXTRA_SBATCH_ARGS[@]}" \
        --output="$LOG_DIR/debias_m_${task_name}_${run_type}_agg_%A.out" \
        --error="$LOG_DIR/debias_m_${task_name}_${run_type}_agg_%A.err" \
        --job-name="debias_m_${task_name}_${run_type}_agg" \
        --wrap ". ~/.bashrc; cd '$SCRIPT_DIR'; mamba run -n benchmark_jupyter python -u debiasm_method.py '$task_name' '$BENCHMARK_DATASETS_DIR' --mode aggregate --run-type '$run_type'; mamba run -n benchmark_jupyter python -u log_results_to_mlflow.py '$task_name' debias_m '$run_type' '$BENCHMARK_DATASETS_DIR'"
}

submit_debiasm_aggregate_only_job() {
    local run_type=$1
    sbatch \
        --cpus-per-task="$CONFIG_JOB_CPUS" \
        --mem="$CONFIG_JOB_MEM" \
        --time="$CONFIG_JOB_TIME" \
        "${EXTRA_SBATCH_ARGS[@]}" \
        --output="$LOG_DIR/debias_m_${task_name}_${run_type}_agg_%A.out" \
        --error="$LOG_DIR/debias_m_${task_name}_${run_type}_agg_%A.err" \
        --job-name="debias_m_${task_name}_${run_type}_agg" \
        --wrap ". ~/.bashrc; cd '$SCRIPT_DIR'; mamba run -n benchmark_jupyter python -u debiasm_method.py '$task_name' '$BENCHMARK_DATASETS_DIR' --mode aggregate --run-type '$run_type'; mamba run -n benchmark_jupyter python -u log_results_to_mlflow.py '$task_name' debias_m '$run_type' '$BENCHMARK_DATASETS_DIR'"
}

# Main execution loop over tasks
echo "════════════════════════════════════════════════════════════"
echo "Starting Long CPU Benchmarking Tools"
echo "════════════════════════════════════════════════════════════"

for task_name in "${SELECTED_TASKS[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "TASK: $task_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Run SIAMCAT if selected
    if is_tool_selected "siamcat"; then
        echo ""
        echo "┌─────────────────────────────────────────────────────────┐"
        echo "│ Running SIAMCAT                                         │"
        echo "└─────────────────────────────────────────────────────────┘"

        LOG_DIR="$LOG_ROOT/siamcat/$task_name"
        {
            for run_type in default optimized; do
                if [[ "$AGGREGATE_ONLY" == "true" ]]; then
                    submit_siamcat_aggregate_only_job "$run_type"
                else
                    config_count=$(get_siamcat_config_count "$run_type")
                    if [[ "$config_count" -le 0 ]]; then
                        echo "Error: No SIAMCAT configs found for $run_type" >&2
                        exit 1
                    fi
                    submit_siamcat_jobs "$run_type" "$config_count"
                fi
            done

            if [[ "$AGGREGATE_ONLY" == "true" ]]; then
                echo "✓ SIAMCAT aggregation-only jobs submitted"
            else
                echo "✓ SIAMCAT jobs submitted"
            fi
        } >>"$LOG_DIR/siamcat_${task_name}_launcher.out" 2>>"$LOG_DIR/siamcat_${task_name}_launcher.err"
    fi

    # Run DEBIAS-M if selected
    if is_tool_selected "debias_m"; then
        echo ""
        echo "┌─────────────────────────────────────────────────────────┐"
        echo "│ Running DEBIAS-M                                        │"
        echo "└─────────────────────────────────────────────────────────┘"

        LOG_DIR="$LOG_ROOT/debias_m/$task_name"
        {
            for run_type in default optimized; do
                if [[ "$AGGREGATE_ONLY" == "true" ]]; then
                    submit_debiasm_aggregate_only_job "$run_type"
                else
                    config_count=$(get_debiasm_config_count "$run_type")
                    if [[ "$config_count" -le 0 ]]; then
                        echo "Error: No DEBIAS-M configs found for $run_type" >&2
                        exit 1
                    fi
                    submit_debiasm_jobs "$run_type" "$config_count"
                fi
            done

            if [[ "$AGGREGATE_ONLY" == "true" ]]; then
                echo "✓ DEBIAS-M aggregation-only jobs submitted"
            else
                echo "✓ DEBIAS-M jobs submitted"
            fi
        } >>"$LOG_DIR/debias_m_${task_name}_launcher.out" 2>>"$LOG_DIR/debias_m_${task_name}_launcher.err"
    fi

    echo ""
    echo "✓ Task $task_name completed"
done
