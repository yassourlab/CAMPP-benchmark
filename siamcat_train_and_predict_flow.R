# Install required packages (one-time setup):
# install.packages("tidyverse")
# install.packages("dplyr")
# if (!requireNamespace("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# BiocManager::install("SIAMCAT")
# install.packages("ranger")
library(jsonlite)
library(rlang)
library(tidyverse)
library(dplyr)
library(SIAMCAT)

set.seed(17)

options(error = traceback)
############# Helper function for training and testing ############

train_and_test_model <- function(features,
                                 meta,
                                 test_features,
                                 num_folds,
                                 num_repeats,
                                 inseparable_var = "subject_id",
                                 filter_method = "abundance",
                                 filter_cutoff = 0.001,
                                 norm = "log.std",
                                 ml = "lasso") {
  start_time <- Sys.time()

  siamcat_obj <- siamcat(
    feat = features,
    meta = meta,
    label = "label",
    case = "1",
    verbose = 1
  )

  # Feature filtering
  siamcat_obj <- filter.features(
    siamcat_obj,
    filter.method = filter_method,
    cutoff = filter_cutoff,
    verbose = 0
  )

  siamcat_obj <- normalize.features(siamcat_obj, norm.method = norm)

  siamcat_obj <- create.data.split(
    siamcat_obj,
    num.folds = num_folds,
    num.resample = num_repeats,
    stratify = TRUE,
    inseparable = inseparable_var,
    verbose = 0
  )

  # Train model
  siamcat_obj <- train.model(
    siamcat_obj,
    method = ml,
    measure = "classif.auc",
    verbose = 1,
  )

  # Make predictions on training data for CV evaluation
  siamcat_obj <- make.predictions(siamcat_obj)
  siamcat_obj <- evaluate.predictions(siamcat_obj, verbose = 0)

  # Get model evaluation
  eval <- eval_data(siamcat_obj)

  # Create SIAMCAT object for test data with dummy metadata
  # This avoids the warning about missing label information
  test_meta <- data.frame(
    sample_id = colnames(test_features),
    row.names = colnames(test_features)
  )

  siamcat_test <- siamcat(
    feat = test_features,
    meta = test_meta,
    verbose = 0
  )

  # Make predictions on test data with frozen normalization
  # The make.predictions function will handle normalization internally
  siamcat_test <- make.predictions(
    siamcat = siamcat_obj,
    siamcat.holdout = siamcat_test,
    normalize.holdout = TRUE,
    verbose = 0
  )

  # Extract test predictions
  test_pred_matrix <- pred_matrix(siamcat_test)

  time_elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

  current_result <- data.frame(
    filter_method = filter_method,
    filter_cutoff = filter_cutoff,
    norm = norm,
    ml = ml,
    mean_test_score = mean(eval$auroc)
  )

  return(list(
    current_result = current_result,
    test_predictions = test_pred_matrix,
    time_elapsed = time_elapsed,
    siamcat_obj = siamcat_obj,
    eval = eval
  ))
}

############# Helper functions for output processing ############

extract_predictions <- function(test_pred_matrix) {
  # Get predictions (average across CV repeats if multiple columns)
  if (is.matrix(test_pred_matrix)) {
    pred_vector <- rowMeans(test_pred_matrix)
  } else {
    pred_vector <- test_pred_matrix
  }
  return(pred_vector)
}

save_outputs <- function(pred_vector, time_elapsed, output_dir) {
  # Create output directory
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  # Create predictions dataframe
  predictions_df <- data.frame(
    sample_id = names(pred_vector),
    prediction = as.numeric(pred_vector)
  )

  # Save predictions
  write.csv(predictions_df,
    file = file.path(output_dir, "predictions.csv"),
    row.names = FALSE
  )

  # Save training time
  write(as.character(time_elapsed),
    file = file.path(output_dir, "train_time.txt")
  )
}

############# run_with_params: Shared function for training with parameter grid ############

run_with_params <- function(features, meta, test_features, test_sample_ids,
                           filter_cutoffs, norm_methods, ml_methods,
                           run_type, partial_output_dir) {
  cat(sprintf("Training with parameter search (%s)...\n", run_type))

  num_folds <- 5
  num_repeats <- 1

  # Create output directory
  full_output_dir <- file.path(partial_output_dir, run_type)
  dir.create(full_output_dir, recursive = TRUE, showWarnings = FALSE)

  # Create results storage
  all_results <- data.frame()
  counter <- 0
  total_combinations <- length(filter_cutoffs) * length(ml_methods) * length(norm_methods)

  # Grid search
  for (filter_cutoff in filter_cutoffs) {
    for (ml in ml_methods) {
      for (norm in norm_methods) {
        counter <- counter + 1

        cat(sprintf(
          "Combination %d/%d: filter_cutoff=%s, ml=%s, norm=%s\n",
          counter, total_combinations, filter_cutoff, ml, norm
        ))

        result <- tryCatch({
          train_and_test_model(
            features = features,
            meta = meta,
            test_features = test_features,
            num_folds = num_folds,
            num_repeats = num_repeats,
            filter_method = "abundance",
            filter_cutoff = filter_cutoff,
            norm = norm,
            ml = ml
          )
        }, error = function(e) {
          cat(sprintf("ERROR: Failed with filter_cutoff=%s, ml=%s, norm=%s\n",
                     filter_cutoff, ml, norm))
          cat(sprintf("Error message: %s\n", e$message))
          return(NULL)
        })

        # Skip this combination if it failed
        if (is.null(result)) {
          cat("Skipping this combination and continuing...\n\n")
          next
        }

        # Create result row
        result_row <- result$current_result
        result_row$mean_fit_time <- result$time_elapsed / num_folds

        # Extract fold-level AUC scores from evaluation data
        fold_aucs <- result$eval$auroc
        for (fold_idx in 1:length(fold_aucs)) {
          result_row[[paste0("fold_", fold_idx, "_auc")]] <- fold_aucs[fold_idx]
        }

        # Add std_test_score
        result_row$std_test_score <- sd(fold_aucs)

        # Store results
        all_results <- rbind(all_results, result_row)
      }
    }
  }

  # Save parameter search results
  write.csv(all_results,
    file = file.path(full_output_dir, "params_search.csv"),
    row.names = FALSE
  )

  # Get best parameters
  best_idx <- which.max(all_results$mean_test_score)
  best_params <- all_results[best_idx, ]

  cat(sprintf(
    "\nBest parameters: filter_cutoff=%s, ml=%s, norm=%s, AUC=%.4f\n",
    best_params$filter_cutoff, best_params$ml, best_params$norm,
    best_params$mean_test_score
  ))

  # Train final model with best parameters
  cat("Training final model with best parameters...\n")
  final_result <- train_and_test_model(
    features = features,
    meta = meta,
    test_features = test_features,
    num_folds = 5,
    num_repeats = 1,
    filter_method = "abundance",
    filter_cutoff = best_params$filter_cutoff,
    norm = best_params$norm,
    ml = best_params$ml
  )

  # Extract and save predictions
  pred_vector <- extract_predictions(final_result$test_predictions)
  save_outputs(pred_vector, final_result$time_elapsed, full_output_dir)

  return(full_output_dir)
}

############# Config helpers for per-configuration runs ############

build_param_configs <- function(run_type) {
  if (run_type == "default") {
    filter_cutoffs <- c(0.001)
    norm_methods <- c("log.std")
    ml_methods <- c("lasso")
  } else if (run_type == "optimized") {
    filter_cutoffs <- c(0.001, 0.005, 0.01)
    norm_methods <- c("rank.unit", "rank.std", "log.std", "log.unit", "log.clr", "std")
    ml_methods <- c("lasso", "enet", "ridge", "lasso_ll", "ridge_ll", "randomForest")
  } else {
    stop(sprintf("Invalid run_type: %s", run_type))
  }

  configs <- list()
  counter <- 0
  for (filter_cutoff in filter_cutoffs) {
    for (ml in ml_methods) {
      for (norm in norm_methods) {
        counter <- counter + 1
        configs[[counter]] <- list(
          filter_cutoff = filter_cutoff,
          ml = ml,
          norm = norm
        )
      }
    }
  }

  return(configs)
}

config_output_dir <- function(partial_output_dir, run_type, config_index) {
  return(file.path(partial_output_dir, run_type, "configs", sprintf("config_%04d", config_index)))
}

run_single_config <- function(features, meta, test_features, test_sample_ids,
                              run_type, partial_output_dir, config_index) {
  configs <- build_param_configs(run_type)

  if (config_index < 0 || config_index >= length(configs)) {
    stop(sprintf("Config index %d out of range (max %d)", config_index, length(configs) - 1))
  }

  config <- configs[[config_index + 1]]
  cat(sprintf(
    "Running config %d: filter_cutoff=%s, ml=%s, norm=%s\n",
    config_index, config$filter_cutoff, config$ml, config$norm
  ))

  num_folds <- 5
  num_repeats <- 1

  result <- train_and_test_model(
    features = features,
    meta = meta,
    test_features = test_features,
    num_folds = num_folds,
    num_repeats = num_repeats,
    filter_method = "abundance",
    filter_cutoff = config$filter_cutoff,
    norm = config$norm,
    ml = config$ml
  )

  result_row <- result$current_result
  result_row$mean_fit_time <- result$time_elapsed / num_folds

  fold_aucs <- result$eval$auroc
  for (fold_idx in 1:length(fold_aucs)) {
    result_row[[paste0("fold_", fold_idx, "_auc")]] <- fold_aucs[fold_idx]
  }
  result_row$std_test_score <- sd(fold_aucs)

  output_dir <- config_output_dir(partial_output_dir, run_type, config_index)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(result_row,
    file = file.path(output_dir, "params_search.csv"),
    row.names = FALSE
  )

  return(output_dir)
}

aggregate_configs <- function(features, meta, test_features, test_sample_ids,
                              run_type, partial_output_dir) {
  configs <- build_param_configs(run_type)

  all_results <- data.frame()
  missing <- c()
  for (config_index in 0:(length(configs) - 1)) {
    output_dir <- config_output_dir(partial_output_dir, run_type, config_index)
    params_path <- file.path(output_dir, "params_search.csv")
    if (!file.exists(params_path)) {
      missing <- c(missing, output_dir)
      next
    }
    result_row <- read.csv(params_path)
    all_results <- rbind(all_results, result_row)
  }

  if (length(missing) > 0) {
    missing_list <- paste(missing, collapse = "\n")
    stop(sprintf("Missing config outputs:\n%s", missing_list))
  }

  run_output_dir <- file.path(partial_output_dir, run_type)
  dir.create(run_output_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(all_results,
    file = file.path(run_output_dir, "params_search.csv"),
    row.names = FALSE
  )

  best_idx <- which.max(all_results$mean_test_score)
  best_params <- all_results[best_idx, ]

  cat(sprintf(
    "\nBest parameters: filter_cutoff=%s, ml=%s, norm=%s, AUC=%.4f\n",
    best_params$filter_cutoff, best_params$ml, best_params$norm,
    best_params$mean_test_score
  ))

  final_result <- train_and_test_model(
    features = features,
    meta = meta,
    test_features = test_features,
    num_folds = 5,
    num_repeats = 1,
    filter_method = "abundance",
    filter_cutoff = best_params$filter_cutoff,
    norm = best_params$norm,
    ml = best_params$ml
  )

  pred_vector <- extract_predictions(final_result$test_predictions)
  save_outputs(pred_vector, final_result$time_elapsed, run_output_dir)

  return(run_output_dir)
}

############# run_default: Train with default parameters ############

run_default <- function(features, meta, test_features, test_sample_ids, partial_output_dir) {
  cat("Training with default parameters...\n")

  # Single parameter set (default)
  filter_cutoffs <- c(0.001)
  norm_methods <- c("log.std")
  ml_methods <- c("lasso")

  return(run_with_params(features, meta, test_features, test_sample_ids,
                        filter_cutoffs, norm_methods, ml_methods,
                        "default", partial_output_dir))
}

############# run_optimized: Train with hyperparameter search ############

run_optimized <- function(features, meta, test_features, test_sample_ids, partial_output_dir) {
  cat("Training with parameter search...\n")

  # Define parameter grid
  filter_cutoffs <- c(0.001, 0.005, 0.01)
  norm_methods <- c("rank.unit", "rank.std", "log.std", "log.unit", "log.clr", "std")
  ml_methods <- c("lasso", "enet", "ridge", "lasso_ll", "ridge_ll", "randomForest")



  return(run_with_params(features, meta, test_features, test_sample_ids,
                        filter_cutoffs, norm_methods, ml_methods,
                        "optimized", partial_output_dir))
}


############# Main execution ############

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)
task_name <- args[1]
benchmark_datasets <- args[2]

mode <- "full"
run_type <- NA
config_index <- NA

if (length(args) > 2) {
  i <- 3
  while (i <= length(args)) {
    if (args[i] == "--mode") {
      mode <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--run-type") {
      run_type <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--config-index") {
      config_index <- as.integer(args[i + 1])
      i <- i + 2
    } else {
      stop(sprintf("Unknown argument: %s", args[i]))
    }
  }
}

if (mode == "list-configs") {
  if (is.na(run_type)) {
    stop("--run-type is required for list-configs mode")
  }
  configs <- build_param_configs(run_type)
  for (idx in seq_along(configs)) {
    cat(sprintf("%d\t%s\n", idx - 1, toJSON(configs[[idx]], auto_unbox = TRUE)))
  }
  quit(save = "no", status = 0)
}

cat(sprintf("Task name: %s\n", task_name))
cat(sprintf("Benchmark datasets dir: %s\n", benchmark_datasets))

# Load train and test data
train_path <- sprintf("%s/%s_train.csv", benchmark_datasets, task_name)
test_x_path <- sprintf("%s/%s_test.csv", benchmark_datasets, task_name)

train_data <- data.frame(read_csv(train_path, col_types = cols(label = col_character())))
rownames(train_data) <- train_data$sample_id

features <- t(train_data %>% select(-label, -sample_id, -subject_id, -study_id))
meta <- train_data %>% select(sample_id, subject_id, label)

# Read test data
test_data <- data.frame(read_csv(test_x_path))
rownames(test_data) <- test_data$sample_id
test_features <- t(test_data %>% select(-sample_id))

# Define output directory
output_dir <- sprintf("benchmarking_outputs/%s_siamcat", task_name)

if (mode == "run-config") {
  if (is.na(run_type) || is.na(config_index)) {
    stop("--run-type and --config-index are required for run-config mode")
  }
  run_single_config(
    features,
    meta,
    test_features,
    test_data$sample_id,
    run_type,
    output_dir,
    config_index
  )
  quit(save = "no", status = 0)
}

if (mode == "aggregate") {
  if (is.na(run_type)) {
    stop("--run-type is required for aggregate mode")
  }
  aggregate_configs(
    features,
    meta,
    test_features,
    test_data$sample_id,
    run_type,
    output_dir
  )
  quit(save = "no", status = 0)
}

# Full mode: run default and optimized in one process (legacy behavior)
cat("\n========== Running Default Model ==========\n")
default_output_dir <- run_default(
  features,
  meta,
  test_features,
  test_data$sample_id,
  output_dir
)

cat("\n========== Running Optimized Model ==========\n")
optimized_output_dir <- run_optimized(
  features,
  meta,
  test_features,
  test_data$sample_id,
  output_dir
)

cat("\n========== SIAMCAT Training Complete ==========\n")
cat(sprintf("Default outputs: %s\n", default_output_dir))
cat(sprintf("Optimized outputs: %s\n", optimized_output_dir))
