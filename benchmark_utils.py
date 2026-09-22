"""
Shared utilities for benchmark experiments (non-MLflow functions).
"""
import pandas as pd

# Random seed for reproducibility
SEED = 17

# Feature scaling options
FEATURE_SCALES = {
    'sum1': 1.0,
}

# Data loading configuration
TRAIN_COLS_TO_DROP = ['sample_id', 'subject_id', 'study_id']
TEST_COLS_TO_DROP = ['sample_id']

# Available tasks and tools
ALL_TASKS = ['scz', 'ghs', 'crc', 'ibd', 'dm90', 'dm7']
CPU_TOOLS = ['random_forest', 'xgboost', 'logistic_regression', 'svm', 'siamcat', 'debias_m']
GPU_TOOLS = ['deep_micro', 'fully_connected', 'tabpfn']
ALL_TOOLS = CPU_TOOLS + GPU_TOOLS


def load_benchmark_data(
    benchmark_datasets_dir,
    task_name,
    get_batch_info=False,
    return_subject_ids=False,
):
    """Load train and test datasets for a benchmark task.

    Args:
        benchmark_datasets_dir: Directory containing benchmark dataset CSVs.
        task_name: Name of the benchmark task (e.g., 'crc', 'ibd').
        get_batch_info: If False (default), returns (train_df, test_x_df) with
            standard metadata columns dropped. If True, returns DEBIAS-M-ready
            arrays (X_train_with_batch, y_train, X_test_with_batch, test_sample_ids)
            where the first column of X is the batch (study_id for train; a new
            unseen batch ID for test).
        return_subject_ids: If True, appends training subject_id values to the
            returned tuple for grouped cross-validation.
    """
    import numpy as np

    train_path = f"{benchmark_datasets_dir}/{task_name}_train.csv"
    test_x_path = f"{benchmark_datasets_dir}/{task_name}_test.csv"

    if not get_batch_info:
        train_df = pd.read_csv(train_path)
        train_subject_ids = train_df['subject_id'].values
        train = train_df.drop(columns=TRAIN_COLS_TO_DROP)
        test_x = pd.read_csv(test_x_path)
        if return_subject_ids:
            return train, test_x, train_subject_ids
        return train, test_x

    # get_batch_info=True: return batch-prepended arrays for DEBIAS-M
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_x_path)

    train_batches = train_df['study_id'].values.astype(int)
    train_subject_ids = train_df['subject_id'].values
    y_train = train_df['label'].values
    train_features = train_df.drop(columns=TRAIN_COLS_TO_DROP + ['label']).values

    test_sample_ids = test_df['sample_id']
    test_features = test_df.drop(columns=TEST_COLS_TO_DROP).values

    # Prepend batch column: DEBIAS-M expects [batch_id | features]
    X_train_with_batch = np.hstack([train_batches[:, np.newaxis], train_features])

    # Assign all test samples a single new batch ID (unseen during training)
    test_batch_id = train_batches.max() + 1
    test_batches = np.full((test_features.shape[0], 1), test_batch_id, dtype=int)
    X_test_with_batch = np.hstack([test_batches, test_features])

    if return_subject_ids:
        return X_train_with_batch, y_train, X_test_with_batch, test_sample_ids, train_subject_ids
    return X_train_with_batch, y_train, X_test_with_batch, test_sample_ids


def scale_train_test_frames(train_df, test_x_df, feature_scale):
    """Scale feature columns for train/test DataFrames based on feature_scale.

    Only feature columns are scaled (label/sample_id are preserved).
    """
    if feature_scale not in FEATURE_SCALES:
        raise ValueError(
            f"Unknown feature_scale '{feature_scale}'. "
            f"Expected one of {list(FEATURE_SCALES.keys())}."
        )

    scale_factor = FEATURE_SCALES[feature_scale]
    if scale_factor == 1.0:
        return train_df, test_x_df

    train_scaled = train_df.copy()
    train_feature_cols = [col for col in train_scaled.columns if col != 'label']
    train_scaled[train_feature_cols] = train_scaled[train_feature_cols] * scale_factor

    test_scaled = test_x_df.copy()
    test_feature_cols = [col for col in test_scaled.columns if col != 'sample_id']
    test_scaled[test_feature_cols] = test_scaled[test_feature_cols] * scale_factor

    return train_scaled, test_scaled
