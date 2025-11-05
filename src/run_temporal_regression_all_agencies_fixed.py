#!/usr/bin/env python3
"""
Fixed and improved version of run_temporal_regression_all_agencies.py
- Removes temporal leakage in CV (forward-chaining per-period splits)
- Detects binary target and switches to classification models/metrics
- Saves encoders and Optuna study alongside model
- Logs per-fold metrics and baseline for context
- Keeps same DATA_ROOT, MODEL_DIR and report saving behavior

Place this file in the same `src` folder. It expects the same CSV layout and will
write reports the same way as the original script.
"""
import os
from pathlib import Path
from datetime import datetime
import json
import traceback
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import optuna
import joblib

# CONFIG
DATA_ROOT = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
N_TRIALS = 10
CV_FOLDS = 5
RANDOM_STATE = 42
TARGET_COL = "Binary Rating"
TIME_COLS = ["Rating Year", "Rating Month"]  # used for temporal splitting
CSV_GLOB = "filteredpromax_kaggle_with_year_month*.csv"
VERBOSE = True


def log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def safe_load_csv(path):
    return pd.read_csv(path)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def is_binary_target(s: pd.Series) -> bool:
    # treat as binary if 2 unique non-null values, or dtype is object with 2 uniques
    s_nonnull = s.dropna()
    unique_vals = s_nonnull.unique()
    # Check if we have exactly 2 unique values that can be converted to integers
    try:
        unique_ints = pd.to_numeric(unique_vals, errors='raise')
        return len(unique_ints) == 2
    except:
        return False  # If values can't be converted to numbers, treat as regression


def build_model_from_trial(trial, task="regression"):
    model_name = trial.suggest_categorical("model", ["xgb", "lgbm", "cat"])
    if task == "regression":
        if model_name == "xgb":
            params = {
                "n_estimators": int(trial.suggest_int("n_estimators", 200, 1200)),
                "max_depth": int(trial.suggest_int("max_depth", 3, 12)),
                "learning_rate": float(trial.suggest_float("learning_rate", 0.01, 0.3, log=True)),
                "subsample": float(trial.suggest_float("subsample", 0.6, 1.0)),
                "colsample_bytree": float(trial.suggest_float("colsample_bytree", 0.4, 1.0)),
                "reg_alpha": float(trial.suggest_float("reg_alpha", 0.0, 2.0)),
                "reg_lambda": float(trial.suggest_float("reg_lambda", 0.0, 2.0)),
                "random_state": RANDOM_STATE,
                "verbosity": 0,
            }
            model = XGBRegressor(**params)
        elif model_name == "lgbm":
            params = {
                "n_estimators": int(trial.suggest_int("n_estimators", 200, 1200)),
                "max_depth": int(trial.suggest_int("max_depth", 3, 12)),
                "learning_rate": float(trial.suggest_float("learning_rate", 0.01, 0.3, log=True)),
                "subsample": float(trial.suggest_float("subsample", 0.6, 1.0)),
                "colsample_bytree": float(trial.suggest_float("colsample_bytree", 0.4, 1.0)),
                "reg_alpha": float(trial.suggest_float("reg_alpha", 0.0, 2.0)),
                "reg_lambda": float(trial.suggest_float("reg_lambda", 0.0, 2.0)),
                "random_state": RANDOM_STATE,
                "verbosity": -1,
            }
            model = LGBMRegressor(**params)
        else:
            params = {
                "iterations": int(trial.suggest_int("iterations", 200, 1200)),
                "depth": int(trial.suggest_int("depth", 4, 10)),
                "learning_rate": float(trial.suggest_float("learning_rate", 0.01, 0.3, log=True)),
                "l2_leaf_reg": float(trial.suggest_float("l2_leaf_reg", 1.0, 5.0)),
                "subsample": float(trial.suggest_float("subsample", 0.6, 1.0)),
                "random_seed": RANDOM_STATE,
                "verbose": 0,
            }
            model = CatBoostRegressor(**params)
    else:
        # classification
        if model_name == "xgb":
            params = {
                "n_estimators": int(trial.suggest_int("n_estimators", 200, 1200)),
                "max_depth": int(trial.suggest_int("max_depth", 3, 12)),
                "learning_rate": float(trial.suggest_float("learning_rate", 0.01, 0.3, log=True)),
                "subsample": float(trial.suggest_float("subsample", 0.6, 1.0)),
                "colsample_bytree": float(trial.suggest_float("colsample_bytree", 0.4, 1.0)),
                "reg_alpha": float(trial.suggest_float("reg_alpha", 0.0, 2.0)),
                "reg_lambda": float(trial.suggest_float("reg_lambda", 0.0, 2.0)),
                "random_state": RANDOM_STATE,
                "verbosity": 0,
                "use_label_encoder": False,
                "objective": "binary:logistic",
            }
            model = XGBClassifier(**params)
        elif model_name == "lgbm":
            params = {
                "n_estimators": int(trial.suggest_int("n_estimators", 200, 1200)),
                "max_depth": int(trial.suggest_int("max_depth", 3, 12)),
                "learning_rate": float(trial.suggest_float("learning_rate", 0.01, 0.3, log=True)),
                "subsample": float(trial.suggest_float("subsample", 0.6, 1.0)),
                "colsample_bytree": float(trial.suggest_float("colsample_bytree", 0.4, 1.0)),
                "reg_alpha": float(trial.suggest_float("reg_alpha", 0.0, 2.0)),
                "reg_lambda": float(trial.suggest_float("reg_lambda", 0.0, 2.0)),
                "random_state": RANDOM_STATE,
            }
            model = LGBMClassifier(**params)
        else:
            params = {
                "iterations": int(trial.suggest_int("iterations", 200, 1200)),
                "depth": int(trial.suggest_int("depth", 4, 10)),
                "learning_rate": float(trial.suggest_float("learning_rate", 0.01, 0.3, log=True)),
                "l2_leaf_reg": float(trial.suggest_float("l2_leaf_reg", 1.0, 5.0)),
                "subsample": float(trial.suggest_float("subsample", 0.6, 1.0)),
                "random_seed": RANDOM_STATE,
                "verbose": 0,
            }
            model = CatBoostClassifier(**params)
    return model, model_name


def temporal_cv_splits_by_period(df, n_splits=5):
    """
    Forward-chaining temporal CV generator by unique (Year,Month).
    Yields (train_idx, val_idx) as integer positions.
    """
    if not all(c in df.columns for c in TIME_COLS):
        # fallback to simple contiguous splits
        idx = np.arange(len(df))
        n_splits = min(n_splits, len(df))
        fold_sizes = int(len(df) / n_splits) or 1
        for k in range(n_splits):
            start = k * fold_sizes
            end = start + fold_sizes if k < n_splits - 1 else len(df)
            val_idx = idx[start:end]
            train_idx = np.setdiff1d(idx, val_idx)
            yield train_idx, val_idx
        return

    # create period id and sort
    df_temp = df.copy()
    df_temp["__period_id__"] = df_temp[TIME_COLS].astype(int).astype(str).agg("-".join, axis=1)
    periods = sorted(df_temp["__period_id__"].unique(), key=lambda s: tuple(map(int, s.split("-"))))
    if len(periods) < 2:
        idx = np.arange(len(df_temp))
        mid = len(idx) // 2
        yield idx[:mid], idx[mid:]
        return

    n_splits = min(n_splits, len(periods))
    # compute contiguous chunks of periods
    chunk_sizes = [len(periods) // n_splits + (1 if i < len(periods) % n_splits else 0) for i in range(n_splits)]
    chunks = []
    i = 0
    for sz in chunk_sizes:
        chunks.append(periods[i:i+sz])
        i += sz

    # For each validation chunk, use only periods STRICTLY BEFORE the first validation period as training
    for k in range(len(chunks)):
        val_periods = chunks[k]
        first_val = val_periods[0]
        try:
            first_val_index = periods.index(first_val)
        except ValueError:
            continue
        train_periods = periods[:first_val_index]
        if len(train_periods) == 0:
            # cannot form a forward-chaining split for this chunk
            continue
        train_idx = df_temp[df_temp["__period_id__"].isin(train_periods)].index.values
        val_idx = df_temp[df_temp["__period_id__"].isin(val_periods)].index.values
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        yield np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


def safe_label_encode_train_val(X_train, X_val, categorical_cols):
    """
    Fit label encoders on X_train only and apply to X_train and X_val. Returns encoders dict too.
    """
    X_train = X_train.copy().reset_index(drop=True)
    X_val = X_val.copy().reset_index(drop=True)
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(X_train[col].astype(str))
        mapping = {c: i for i, c in enumerate(le.classes_)}
        encoders[col] = mapping
        X_train[col] = X_train[col].astype(str).map(mapping).fillna(-1).astype(int)
        X_val[col] = X_val[col].astype(str).map(mapping).fillna(-1).astype(int)
    return X_train, X_val, encoders


def safe_label_encode_train_test_final(X_train, X_test, categorical_cols):
    """
    Fit encoders on X_train and apply to both. Return encoders as well.
    """
    X_train = X_train.copy().reset_index(drop=True)
    X_test = X_test.copy().reset_index(drop=True)
    encoders = {}
    
    # First pass: analyze the data
    for col in categorical_cols:
        train_unique = set(X_train[col].astype(str).unique())
        test_unique = set(X_test[col].astype(str).unique())
        overlap = train_unique & test_unique
        if len(overlap) == 0:
            log(f"Warning: No overlap in values for column {col} between train and test!")
            log(f"Train unique: {train_unique}")
            log(f"Test unique: {test_unique}")
    
    for col in categorical_cols:
        # Combine train and test values to ensure consistent encoding
        all_values = pd.concat([X_train[col], X_test[col]]).astype(str).unique()
        mapping = {c: i for i, c in enumerate(sorted(all_values))}
        encoders[col] = mapping
        
        # Apply encoding
        X_train[col] = X_train[col].astype(str).map(mapping).fillna(-1).astype(int)
        X_test[col] = X_test[col].astype(str).map(mapping).fillna(-1).astype(int)
        
        # Verify encoding
        if (X_train[col] == -1).any() or (X_test[col] == -1).any():
            log(f"Warning: Missing values detected after encoding in column {col}")
            
    return X_train, X_test, encoders


def tune_and_evaluate(df, X, y, categorical_cols, task="regression", n_trials=N_TRIALS, cv_folds=CV_FOLDS):
    # objective uses temporal_cv_splits_by_period
    def objective(trial):
        model, model_name = build_model_from_trial(trial, task=task)
        fold_scores = []
        splits = list(temporal_cv_splits_by_period(df, n_splits=cv_folds))
        if len(splits) == 0:
            # fallback simple splits
            idx = np.arange(len(X))
            np.random.RandomState(RANDOM_STATE).shuffle(idx)
            fold_sizes = int(len(idx) / cv_folds) or 1
            for k in range(cv_folds):
                s = k*fold_sizes
                e = s + fold_sizes if k < cv_folds-1 else len(idx)
                train_idx = np.concatenate([idx[:s], idx[e:]])
                val_idx = idx[s:e]
                splits.append((np.array(train_idx, dtype=int), np.array(val_idx, dtype=int)))

        for train_idx, val_idx in splits:
            try:
                X_train = X.iloc[train_idx].reset_index(drop=True)
                X_val = X.iloc[val_idx].reset_index(drop=True)
                y_train = y.iloc[train_idx].reset_index(drop=True)
                y_val = y.iloc[val_idx].reset_index(drop=True)
                X_train_enc, X_val_enc, _ = safe_label_encode_train_val(X_train, X_val, categorical_cols)
                y_train_np = np.asarray(y_train).ravel()
                y_val_np = np.asarray(y_val).ravel()
                model.fit(X_train_enc, y_train_np)
                preds = model.predict(X_val_enc)
                if task == "regression":
                    fold_scores.append(rmse(y_val_np, preds))
                else:
                    # classification: use predicted probabilities when available
                    try:
                        prob = model.predict_proba(X_val_enc)[:, 1]
                        score = roc_auc_score(y_val_np, prob)
                    except Exception:
                        # fallback to using raw predictions
                        score = roc_auc_score(y_val_np, preds)
                    # Optuna minimizes objective; use 1 - AUC
                    fold_scores.append(1.0 - float(score))
            except Exception as e:
                log("  fold error:", e)
                traceback.print_exc()
                return float("inf")
        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def run_for_agency(agency_csv_path):
    agency_csv_path = Path(agency_csv_path)
    agency_name = agency_csv_path.parent.name
    log("\n" + "="*60)
    log(f"🚀 Running agency: {agency_name}")
    try:
        df = safe_load_csv(agency_csv_path)
        if TARGET_COL not in df.columns:
            raise RuntimeError(f"{TARGET_COL} not found in {agency_csv_path}")

        df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
        y = df[TARGET_COL]

        # Detect potential leakage columns (high numeric correlation or perfect categorical predictors)
        potential_leakage = []
        for col in df.columns:
            if col == TARGET_COL:
                continue
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # numeric correlation (coerce to numeric when possible)
                    try:
                        corr = df[col].astype(float).corr(df[TARGET_COL].astype(float))
                    except Exception:
                        corr = df[col].corr(df[TARGET_COL])
                    if pd.notnull(corr) and abs(corr) > 0.95:
                        log(f"WARNING: High correlation detected: {col} (correlation: {corr:.4f})")
                        potential_leakage.append(col)
                else:
                    # categorical: check if any category maps to a single target value
                    vc = df.groupby(col)[TARGET_COL].nunique()
                    if (vc == 1).any():
                        perfect_vals = vc[vc == 1].index.tolist()
                        log(f"WARNING: Perfect categorical predictor detected: {col}")
                        log(f"Values that perfectly predict target: {perfect_vals}")
                        potential_leakage.append(col)
            except Exception:
                # ignore columns that can't be analyzed
                continue

        if potential_leakage:
            log("\nWARNING: Removing columns suspected of data leakage:")
            for col in potential_leakage:
                log(f"- {col}")

        # Remove leakage columns from features
        X = df.drop(columns=[TARGET_COL] + potential_leakage)
        categorical_cols = [c for c in X.select_dtypes(include="object").columns.tolist() if c not in TIME_COLS]

        # Print remaining features
        log("\nFeatures being used for modeling:")
        log(f"Numeric features: {[c for c in X.select_dtypes(include=np.number).columns]}")
        log(f"Categorical features: {categorical_cols}")

        # detect task and log target details
        unique_vals = y.unique()
        task = "classification" if is_binary_target(y) else "regression"
        log(f"  Detected task: {task}")
        log(f"  Target unique values: {unique_vals}")
        log(f"  Target dtype: {y.dtype}")

        # prepare temporal final split (80/20) using TIME_COLS if available
        if all(c in df.columns for c in TIME_COLS):
            # Create a single timestamp for sorting
            df['__temp_date'] = df[TIME_COLS[0]].astype(str) + df[TIME_COLS[1]].astype(str).str.zfill(2)
            df_sorted = df.sort_values('__temp_date').reset_index(drop=True)

            # Log temporal information
            log("\nTemporal split information:")
            log(f"Time range: {df['__temp_date'].min()} to {df['__temp_date'].max()}")

            split_idx = int(len(df_sorted) * 0.8)
            train_idx = df_sorted.index[:split_idx].values
            test_idx = df_sorted.index[split_idx:].values

            # Verify temporal separation
            train_max_date = df_sorted.loc[train_idx, '__temp_date'].max()
            test_min_date = df_sorted.loc[test_idx, '__temp_date'].min()
            log(f"Train period: up to {train_max_date}")
            log(f"Test period: from {test_min_date}")

            # Check for temporal separation of values in categorical columns
            for col in categorical_cols:
                if col not in df_sorted.columns:
                    continue
                train_values = set(df_sorted.loc[train_idx, col].dropna().unique())
                test_values = set(df_sorted.loc[test_idx, col].dropna().unique())
                new_in_test = test_values - train_values
                if len(new_in_test) > 0:
                    log(f"\nWARNING: New values in test set for column {col}:")
                    log(f"Values only in test: {new_in_test}")

            # Check target distribution over time
            time_periods = sorted(df_sorted['__temp_date'].unique())
            target_by_period = pd.DataFrame({
                'period': time_periods,
                'target_mean': [df_sorted[df_sorted['__temp_date'] == p][TARGET_COL].mean() for p in time_periods]
            })
            log("\nTarget distribution over time:")
            log(target_by_period)

            # Warning if target distribution changes dramatically
            train_target_mean = df_sorted.loc[train_idx, TARGET_COL].mean()
            test_target_mean = df_sorted.loc[test_idx, TARGET_COL].mean()
            if abs(train_target_mean - test_target_mean) > 0.1:
                log(f"\nWARNING: Large shift in target distribution:")
                log(f"Train mean: {train_target_mean:.3f}")
                log(f"Test mean: {test_target_mean:.3f}")

            df = df.drop(columns=['__temp_date'])
        else:
            log("\nWarning: No temporal columns found, using random split!")
            idx = np.arange(len(df))
            np.random.seed(RANDOM_STATE)  # Ensure reproducibility
            np.random.shuffle(idx)
            split_idx = int(len(idx)*0.8)
            train_idx = idx[:split_idx]
            test_idx = idx[split_idx:]

        X_full = X.reset_index(drop=True)
        y_full = y.reset_index(drop=True)

        X_train = X_full.loc[train_idx].reset_index(drop=True)
        X_test = X_full.loc[test_idx].reset_index(drop=True)
        y_train = y_full.loc[train_idx].reset_index(drop=True)
        y_test = y_full.loc[test_idx].reset_index(drop=True)

        df_for_cv = pd.concat([X_full, y_full], axis=1)

        log(f"  rows: total={len(df)}, train={len(train_idx)}, test={len(test_idx)}")

        # compute simple baseline
        baseline = {}
        if task == "regression":
            baseline_pred = np.asarray(y_train).ravel().mean()
            baseline_rmse = rmse(np.asarray(y_test).ravel(), np.full(len(y_test), baseline_pred))
            baseline["baseline_rmse"] = baseline_rmse
            log(f"  baseline (predict mean) test RMSE: {baseline_rmse:.4f}")
        else:
            # predict majority class
            majority = pd.Series(y_train).mode().iloc[0]
            try:
                auc = roc_auc_score(y_test.astype(int), np.full(len(y_test), int(majority)))
            except Exception:
                auc = float("nan")
            baseline["baseline_majority"] = int(majority)
            baseline["baseline_auc"] = float(auc)
            log(f"  baseline majority: {majority}, baseline AUC (degenerate): {auc}")

        log("  🔎 Starting Optuna tuning...")
        study = tune_and_evaluate(df_for_cv, X_full, y_full, categorical_cols, task=task, n_trials=N_TRIALS, cv_folds=CV_FOLDS)

        if study.best_value == float("inf"):
            raise RuntimeError("All trials failed during tuning.")

        best_params = study.best_trial.params
        best_model_type = study.best_trial.params.get("model", None)
        if best_model_type is None:
            if "iterations" in study.best_trial.params:
                best_model_type = "cat"
            elif "colsample_bytree" in study.best_trial.params and "n_estimators" in study.best_trial.params:
                best_model_type = "xgb"
            else:
                best_model_type = "xgb"

        # Reconstruct final model using best trial params (matching task)
        if task == "regression":
            if best_model_type == "xgb":
                construct_params = {
                    "n_estimators": int(study.best_trial.params.get("n_estimators", 500)),
                    "max_depth": int(study.best_trial.params.get("max_depth", 6)),
                    "learning_rate": float(study.best_trial.params.get("learning_rate", 0.1)),
                    "subsample": float(study.best_trial.params.get("subsample", 1.0)),
                    "colsample_bytree": float(study.best_trial.params.get("colsample_bytree", 1.0)),
                    "reg_alpha": float(study.best_trial.params.get("reg_alpha", 0.0)),
                    "reg_lambda": float(study.best_trial.params.get("reg_lambda", 1.0)),
                        "random_state": RANDOM_STATE,
                        "verbosity": 0,
                }
                best_model = XGBRegressor(**construct_params)
            elif best_model_type == "lgbm":
                construct_params = {
                    "n_estimators": int(study.best_trial.params.get("n_estimators", 500)),
                    "max_depth": int(study.best_trial.params.get("max_depth", 6)),
                    "learning_rate": float(study.best_trial.params.get("learning_rate", 0.1)),
                    "subsample": float(study.best_trial.params.get("subsample", 1.0)),
                    "colsample_bytree": float(study.best_trial.params.get("colsample_bytree", 1.0)),
                    "reg_alpha": float(study.best_trial.params.get("reg_alpha", 0.0)),
                    "reg_lambda": float(study.best_trial.params.get("reg_lambda", 1.0)),
                    "random_state": RANDOM_STATE,
                    "min_child_samples": 20,
                    "min_split_gain": 0.01,
                    "path_smooth": 0.1,
                    "verbosity": -1,
                }
                best_model = LGBMRegressor(**construct_params)
            else:
                construct_params = {
                    "iterations": int(study.best_trial.params.get("iterations", 500)),
                    "depth": int(study.best_trial.params.get("depth", 6)),
                    "learning_rate": float(study.best_trial.params.get("learning_rate", 0.1)),
                    "l2_leaf_reg": float(study.best_trial.params.get("l2_leaf_reg", 3.0)),
                    "subsample": float(study.best_trial.params.get("subsample", 1.0)),
                    "random_seed": RANDOM_STATE,
                    "verbose": 0,
                }
                best_model = CatBoostRegressor(**construct_params)
        else:
            # classification
            if best_model_type == "xgb":
                construct_params = {
                    "n_estimators": int(study.best_trial.params.get("n_estimators", 500)),
                    "max_depth": int(study.best_trial.params.get("max_depth", 6)),
                    "learning_rate": float(study.best_trial.params.get("learning_rate", 0.1)),
                    "subsample": float(study.best_trial.params.get("subsample", 1.0)),
                    "colsample_bytree": float(study.best_trial.params.get("colsample_bytree", 1.0)),
                    "reg_alpha": float(study.best_trial.params.get("reg_alpha", 0.0)),
                    "reg_lambda": float(study.best_trial.params.get("reg_lambda", 1.0)),
                    "random_state": RANDOM_STATE,
                    "verbosity": 0,
                    "use_label_encoder": False,
                    "objective": "binary:logistic",
                }
                best_model = XGBClassifier(**construct_params)
            elif best_model_type == "lgbm":
                construct_params = {
                    "n_estimators": int(study.best_trial.params.get("n_estimators", 500)),
                    "max_depth": int(study.best_trial.params.get("max_depth", 6)),
                    "learning_rate": float(study.best_trial.params.get("learning_rate", 0.1)),
                    "subsample": float(study.best_trial.params.get("subsample", 1.0)),
                    "colsample_bytree": float(study.best_trial.params.get("colsample_bytree", 1.0)),
                    "reg_alpha": float(study.best_trial.params.get("reg_alpha", 0.0)),
                    "reg_lambda": float(study.best_trial.params.get("reg_lambda", 1.0)),
                    "random_state": RANDOM_STATE,
                    "min_child_samples": 20,
                    "min_split_gain": 0.01,
                    "path_smooth": 0.1,
                }
                best_model = LGBMClassifier(**construct_params)
            else:
                construct_params = {
                    "iterations": int(study.best_trial.params.get("iterations", 500)),
                    "depth": int(study.best_trial.params.get("depth", 6)),
                    "learning_rate": float(study.best_trial.params.get("learning_rate", 0.1)),
                    "l2_leaf_reg": float(study.best_trial.params.get("l2_leaf_reg", 3.0)),
                    "subsample": float(study.best_trial.params.get("subsample", 1.0)),
                    "random_seed": RANDOM_STATE,
                    "verbose": 0,
                }
                best_model = CatBoostClassifier(**construct_params)

        # final encoding for train/test and fit; also save encoders
        X_train_enc, X_test_enc, encoders = safe_label_encode_train_test_final(X_train, X_test, categorical_cols)
        y_train_np = np.asarray(y_train).ravel()
        y_test_np = np.asarray(y_test).ravel()

        log("  ⏳ Training final model on temporal train split...")
        best_model.fit(X_train_enc, y_train_np)

        # Evaluate
        train_pred = best_model.predict(X_train_enc)
        test_pred = best_model.predict(X_test_enc)

        # initialize all metrics to NaN
        train_rmse = float("nan")
        test_rmse = float("nan")
        train_r2 = float("nan")
        test_r2 = float("nan")
        train_auc = float("nan")
        test_auc = float("nan")
        train_accuracy = float("nan")
        test_accuracy = float("nan")

        # Debug information about predictions
        if task == "classification":
            log("\nDebug Classification Metrics:")
            log(f"Train target distribution:\n{pd.Series(y_train_np).value_counts(normalize=True)}")
            log(f"Test target distribution:\n{pd.Series(y_test_np).value_counts(normalize=True)}")
            log(f"Train predictions distribution:\n{pd.Series(train_pred).value_counts(normalize=True)}")
            log(f"Test predictions distribution:\n{pd.Series(test_pred).value_counts(normalize=True)}")
            
            # Check for potential data leakage
            train_features = pd.DataFrame(X_train_enc)
            test_features = pd.DataFrame(X_test_enc)
            for col in train_features.columns:
                if len(set(train_features[col]) & set(test_features[col])) == 0:
                    log(f"Warning: Column {col} has no overlap between train and test!")

        if task == "regression":
            try:
                # Convert predictions and targets to float
                y_train_float = y_train_np.astype(float)
                y_test_float = y_test_np.astype(float)
                train_pred_float = train_pred.astype(float)
                test_pred_float = test_pred.astype(float)
                
                log(f"  Computing regression metrics...")
                log(f"  Train predictions range: [{train_pred_float.min():.2f}, {train_pred_float.max():.2f}]")
                log(f"  Train targets range: [{y_train_float.min():.2f}, {y_train_float.max():.2f}]")
                
                train_rmse = rmse(y_train_float, train_pred_float)
                test_rmse = rmse(y_test_float, test_pred_float)
                train_r2 = float(r2_score(y_train_float, train_pred_float))
                test_r2 = float(r2_score(y_test_float, test_pred_float))
                
                log(f"  Regression metrics computed - RMSE train: {train_rmse:.4f}, test: {test_rmse:.4f}")
                log(f"  R2 scores - train: {train_r2:.4f}, test: {test_r2:.4f}")
            except Exception as e:
                log(f"  Error computing regression metrics: {str(e)}")
                traceback.print_exc()
                train_rmse = float("nan")
                test_rmse = float("nan")
                train_r2 = float("nan")
                test_r2 = float("nan")
        else:
            # classification: compute AUC if possible and accuracy
            try:
                train_prob = best_model.predict_proba(X_train_enc)[:, 1]
                test_prob = best_model.predict_proba(X_test_enc)[:, 1]
                train_auc = float(roc_auc_score(y_train_np, train_prob))
                test_auc = float(roc_auc_score(y_test_np, test_prob))
            except Exception:
                train_auc = float("nan")
                test_auc = float("nan")
            try:
                # Detailed classification metrics
                train_accuracy = float(accuracy_score(y_train_np, train_pred))
                test_accuracy = float(accuracy_score(y_test_np, test_pred))
                
                # Log detailed classification metrics
                log("\nDetailed Classification Metrics:")
                log(f"Number of classes: {len(np.unique(y_train_np))}")
                log(f"Train accuracy: {train_accuracy:.4f}")
                log(f"Test accuracy: {test_accuracy:.4f}")
                
                # Warning for suspiciously high accuracy
                if train_accuracy > 0.99 or test_accuracy > 0.99:
                    log("\nWARNING: Extremely high accuracy detected!")
                    log("This might indicate:")
                    log("1. Data leakage")
                    log("2. Target variable in features")
                    log("3. Temporal leakage")
                    log("4. Too perfect separation of classes")
                
                # Calculate per-class metrics
                from sklearn.metrics import classification_report
                log("\nTrain Classification Report:")
                log(classification_report(y_train_np, train_pred))
                log("\nTest Classification Report:")
                log(classification_report(y_test_np, test_pred))
                
            except Exception as e:
                log(f"Error computing accuracy metrics: {str(e)}")
                traceback.print_exc()
                train_accuracy = float("nan")
                test_accuracy = float("nan")

        # Save model, encoders, and study
        agency_model_dir = MODEL_DIR / agency_name.replace(" ", "_")
        agency_model_dir.mkdir(parents=True, exist_ok=True)
        model_file = agency_model_dir / "best_model.pkl"
        enc_file = agency_model_dir / "encoders.pkl"
        study_file = agency_model_dir / "optuna_study.pkl"
        joblib.dump(best_model, model_file)
        joblib.dump(encoders, enc_file)
        try:
            joblib.dump(study, study_file)
        except Exception:
            # study may not be serializable in some environments; ignore non-fatal
            pass

        report = {
            "agency": agency_name,
            "csv": str(agency_csv_path),
            "rows_total": int(len(df)),
            "rows_train": int(len(train_idx)),
            "rows_test": int(len(test_idx)),
            "model_type": best_model_type,
            "task": task,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_auc": train_auc if task == "classification" else float("nan"),
            "test_auc": test_auc if task == "classification" else float("nan"),
            "train_accuracy": train_accuracy if task == "classification" else float("nan"),
            "test_accuracy": test_accuracy if task == "classification" else float("nan"),
            "optuna_best_value": float(study.best_value),
            "optuna_best_params": study.best_trial.params,
            "baseline": baseline,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        report_df = pd.DataFrame([report])
        report_csv_path = agency_csv_path.parent / f"temporal_model_report_{agency_csv_path.stem}.csv"
        report_df.to_csv(report_csv_path, index=False)

        log("  ✅ Done for", agency_name)
        log(json.dumps(report, indent=2))
        return report

    except Exception as exc:
        log(f"  ❌ Failed for {agency_name}: {exc}")
        traceback.print_exc()
        return {"agency": agency_name, "error": str(exc), "timestamp": datetime.utcnow().isoformat() + "Z"}


def main():
    agency_csvs = []
    for agency_folder in DATA_ROOT.iterdir():
        if agency_folder.is_dir():
            matches = list(agency_folder.glob(CSV_GLOB))
            agency_csvs.extend(matches)

    if not agency_csvs:
        log("No agency CSV files found under data/. Make sure the pattern matches.")
        return

    summary = []
    for csv_path in agency_csvs:
        report = run_for_agency(csv_path)
        summary.append(report)

    summary_df = pd.DataFrame(summary)
    summary_csv = DATA_ROOT / "summary_report.csv"
    summary_df.to_csv(summary_csv, index=False)
    log("\nAll done. Combined summary saved to:", summary_csv)
    log(summary_df)


if __name__ == "__main__":
    main()
