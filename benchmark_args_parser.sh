#!/bin/bash
# Shared library for parsing benchmark tool/task selection arguments
# Source this file in run_cpu_tools.sh, run_gpu_tools.sh, and run_all_benchmarks.sh

# Define all available tasks and tools
ALL_TASKS=("scz" "ghs" "crc" "ibd" "dm90" "dm7")
CPU_TOOLS=("random_forest" "xgboost" "logistic_regression" "svm" "siamcat" "debias_m")
GPU_TOOLS=("deep_micro" "fully_connected" "tabpfn")

# Global variables to be set after parsing
SELECTED_TASKS=()
SELECTED_TOOLS=()
AGGREGATE_ONLY=false

# Print usage information
print_usage() {
    local script_name=$1
    local available_tools=$2
    cat << EOF
Usage: $script_name [OPTIONS]

Options:
    --benchmark-tasks TASKS    Comma-separated list of tasks to run
                              Available: ${ALL_TASKS[*]}
                              Default: all tasks

    --aggregate-only          Run aggregation + MLflow logging only
                              (skip per-configuration jobs)
    
    --tool TOOL                Single tool to run (for array job mode)
                              Available: $available_tools
                              Mutually exclusive with --tools
    
    --tools TOOLS              Comma-separated list of tools to run (legacy mode)
                              Available: $available_tools
                              Default: all available tools for this script
    
    --exclude-tasks TASKS      Comma-separated list of tasks to exclude
                              
    --exclude-tools TOOLS      Comma-separated list of tools to exclude
    
    -h, --help                Show this help message

Examples:
    # Array job mode: Run single tool (task determined by SLURM_ARRAY_TASK_ID)
    $script_name --tool random_forest --benchmark-tasks "crc,ibd"

    # Legacy mode: Run all tools on all tasks (default)
    $script_name

    # Run only on CRC and IBD tasks
    $script_name --benchmark-tasks "crc,ibd"

    # Run only Random Forest on all tasks
    $script_name --tools "random_forest"

    # Exclude SIAMCAT and GHS task
    $script_name --exclude-tools "siamcat" --exclude-tasks "ghs"

EOF
}

# Convert comma-separated string to array
string_to_array() {
    local input="$1"
    local -n output_array=$2
    IFS=',' read -ra output_array <<< "$input"
    # Trim whitespace from each element
    for i in "${!output_array[@]}"; do
        output_array[$i]=$(echo "${output_array[$i]}" | xargs)
    done
}

# Check if array contains element
array_contains() {
    local element="$1"
    shift
    local array=("$@")
    for item in "${array[@]}"; do
        if [[ "$item" == "$element" ]]; then
            return 0
        fi
    done
    return 1
}

# Validate that all elements in test_array exist in valid_array
validate_items() {
    local item_type="$1"  # "task" or "tool"
    shift
    local -n test_arr=$1
    shift
    local valid_arr=("$@")
    
    local invalid=()
    for item in "${test_arr[@]}"; do
        if ! array_contains "$item" "${valid_arr[@]}"; then
            invalid+=("$item")
        fi
    done
    
    if [[ ${#invalid[@]} -gt 0 ]]; then
        echo "Error: Invalid ${item_type}(s): ${invalid[*]}" >&2
        echo "Valid ${item_type}s: ${valid_arr[*]}" >&2
        return 1
    fi
    return 0
}

# Remove excluded items from array
remove_excluded() {
    local -n source_arr=$1
    local -n exclude_arr=$2
    local result=()
    
    for item in "${source_arr[@]}"; do
        if ! array_contains "$item" "${exclude_arr[@]}"; then
            result+=("$item")
        fi
    done
    
    source_arr=("${result[@]}")
}

# Parse command-line arguments
# Args: $1 = script name, $2 = available tools array name (e.g., "CPU_TOOLS" or "GPU_TOOLS")
parse_benchmark_args() {
    local script_name="$1"
    local -n available_tools=$2
    
    local include_tasks=""
    local include_tool=""
    local include_tools=""
    local exclude_tasks=""
    local exclude_tools=""
    AGGREGATE_ONLY=false
    
    # Parse arguments
    while [[ $# -gt 2 ]]; do
        case ${3} in
            --benchmark-tasks)
                include_tasks="$4"
                shift 2
                ;;
            --tool)
                include_tool="$4"
                shift 2
                ;;
            --tools)
                include_tools="$4"
                shift 2
                ;;
            --exclude-tasks)
                exclude_tasks="$4"
                shift 2
                ;;
            --exclude-tools)
                exclude_tools="$4"
                shift 2
                ;;
            --aggregate-only)
                AGGREGATE_ONLY=true
                shift 1
                ;;
            -h|--help)
                print_usage "$script_name" "${available_tools[*]}"
                exit 0
                ;;
            *)
                echo "Error: Unknown option: ${3}" >&2
                print_usage "$script_name" "${available_tools[*]}"
                exit 1
                ;;
        esac
    done
    
    # Validate mutually exclusive options
    if [[ -n "$include_tool" && -n "$include_tools" ]]; then
        echo "Error: --tool and --tools are mutually exclusive" >&2
        exit 1
    fi
    
    # Build task list
    if [[ -n "$include_tasks" ]]; then
        string_to_array "$include_tasks" SELECTED_TASKS
        validate_items "task" SELECTED_TASKS "${ALL_TASKS[@]}" || exit 1
    else
        SELECTED_TASKS=("${ALL_TASKS[@]}")
    fi
    
    # Apply task exclusions
    if [[ -n "$exclude_tasks" ]]; then
        local exclude_tasks_arr
        string_to_array "$exclude_tasks" exclude_tasks_arr
        validate_items "task" exclude_tasks_arr "${ALL_TASKS[@]}" || exit 1
        remove_excluded SELECTED_TASKS exclude_tasks_arr
    fi
    
    # Build tool list
    if [[ -n "$include_tool" ]]; then
        # Single tool mode (for array jobs)
        SELECTED_TOOLS=("$include_tool")
        validate_items "tool" SELECTED_TOOLS "${available_tools[@]}" || exit 1
    elif [[ -n "$include_tools" ]]; then
        # Multiple tools mode (legacy)
        string_to_array "$include_tools" SELECTED_TOOLS
        validate_items "tool" SELECTED_TOOLS "${available_tools[@]}" || exit 1
    else
        # Default: all available tools
        SELECTED_TOOLS=("${available_tools[@]}")
    fi
    
    # Apply tool exclusions (only if not in single-tool mode)
    if [[ -z "$include_tool" && -n "$exclude_tools" ]]; then
        local exclude_tools_arr
        string_to_array "$exclude_tools" exclude_tools_arr
        validate_items "tool" exclude_tools_arr "${available_tools[@]}" || exit 1
        remove_excluded SELECTED_TOOLS exclude_tools_arr
    fi
    
    # Validate we have at least one task and one tool
    if [[ ${#SELECTED_TASKS[@]} -eq 0 ]]; then
        echo "Error: No tasks selected after applying filters" >&2
        exit 1
    fi
    
    if [[ ${#SELECTED_TOOLS[@]} -eq 0 ]]; then
        echo "Error: No tools selected after applying filters" >&2
        exit 1
    fi
}

# Print selected configuration
print_config() {
    echo "════════════════════════════════════════════════════════════"
    echo "Selected Tasks: ${SELECTED_TASKS[*]}"
    echo "Selected Tools: ${SELECTED_TOOLS[*]}"
    echo "Aggregate-only mode: $AGGREGATE_ONLY"
    echo "Number of task-tool combinations: $((${#SELECTED_TASKS[@]} * ${#SELECTED_TOOLS[@]}))"
    echo "════════════════════════════════════════════════════════════"
}
