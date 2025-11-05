import os
import sys
import glob
import traceback
from pathlib import Path
import logging
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
import optuna
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
OUT_SUMMARY = DATA_DIR / 'summary_report.csv'
N_TRIALS = 10
N_FOLDS = 3
RANDOM_STATE = 42


def find_csv_files(data_dir):
    pattern = str(data_dir / '**' / 'filteredpromax_kaggle_with_year_month_*.csv')
    return sorted(glob.glob(pattern, recursive=True))


def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    df = df.copy()
    if 'label' in df.columns:
        y_col = 'label'
    else:
        possible = [c for c in df.columns if c.lower() in ('rating','ratings','rating_label')]
        y_col = possible[0] if possible else df.columns[-1]
    df = df.reset_index(drop=True)
    df = df.drop_duplicates()
    df = df.dropna(axis=0, how='any')
    X = df.drop(columns=[y_col], errors='ignore')
    y = df[y_col]
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    if len(X) != len(y):
        minlen = min(len(X), len(y))
        X = X.iloc[:minlen].reset_index(drop=True)
        y = y.iloc[:minlen].reset_index(drop=True)
    X = X.select_dtypes(include=[np.number])
    if X.shape[1] == 0:
        X = df.drop(columns=[y_col], errors='ignore')
        X = pd.get_dummies(X, drop_first=True)
    return X, y


def make_groups(df):
    if 'year_month' in df.columns:
        grp = df['year_month'].astype(str)
        le = LabelEncoder()
        return le.fit_transform(grp)
    for c in ('year','month','date'):
        if c in df.columns:
            grp = df[c].astype(str)
            le = LabelEncoder()
            return le.fit_transform(grp)
    return np.arange(len(df))


def safe_fit(model, X_t, y_t, X_v=None, y_v=None):
    X_t = np.ascontiguousarray(X_t.values if hasattr(X_t, 'values') else X_t)
    y_t = np.asarray(y_t).ravel()
    if X_v is not None and y_v is not None:
        X_v = np.ascontiguousarray(X_v.values if hasattr(X_v, 'values') else X_v)
        y_v = np.asarray(y_v).ravel()
    try:
        if X_v is not None and y_v is not None:
            try:
                model.fit(X_t, y_t, eval_set=[(X_v, y_v)], early_stopping_rounds=30, verbose=False)
            except TypeError:
                model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
        else:
            model.fit(X_t, y_t)
        return model
    except Exception as e:
        logging.debug('first-fit failed: %s', e)
        try:
            model.fit(X_t, y_t)
            return model
        except Exception:
            raise


def objective_builder(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 64, 512),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.5, log=True),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'use_label_encoder': False,
            'eval_metric': 'mlogloss'
        }
        model = xgb.XGBClassifier(**params)
        safe_fit(model, X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        return acc
    return objective


def run_for_file(csv_path):
    basename = os.path.basename(csv_path)
    logging.info('Processing %s', csv_path)
    X, y = load_and_clean(csv_path)
    if X.shape[0] < 10:
        logging.warning('Too few rows, skipping: %s', csv_path)
        return None
    y = LabelEncoder().fit_transform(y.astype(str))
    df = pd.read_csv(csv_path).reset_index(drop=True)
    groups = make_groups(df)
    if len(np.unique(groups)) < 2:
        logging.warning('Not enough groups for GroupKFold, using simple split for %s', basename)
        split_idx = int(0.8 * len(y))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        objective = objective_builder(X_train, y_train, X_val, y_val)
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(objective, n_trials=N_TRIALS)
        best = study.best_params
        model = xgb.XGBClassifier(**best, use_label_encoder=False, eval_metric='mlogloss')
        safe_fit(model, X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        return {'file': basename, 'rows': len(y), 'val_acc': float(acc), 'best_params': best}

    gkf = GroupKFold(n_splits=min(N_FOLDS, len(np.unique(groups))))
    fold = 0
    fold_results = []
    for tr_idx, val_idx in gkf.split(X, y, groups=groups):
        X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_train, y_val = y[tr_idx], y[val_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 1:
            logging.warning('Insufficient class diversity in fold %s for %s, skipping fold', fold, basename)
            continue
        objective = objective_builder(X_train, y_train, X_val, y_val)
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        try:
            study.optimize(objective, n_trials=N_TRIALS)
        except Exception as e:
            logging.warning('Optuna error on %s fold %d: %s', basename, fold, e)
            fold += 1
            continue
        best = study.best_params
        model = xgb.XGBClassifier(**best, use_label_encoder=False, eval_metric='mlogloss')
        safe_fit(model, X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        fold_results.append({'fold': fold, 'val_acc': float(acc), 'best_params': best, 'val_size': len(y_val)})
        fold += 1
    if len(fold_results) == 0:
        logging.warning('No successful folds for %s', basename)
        return None
    avg_acc = float(np.mean([f['val_acc'] for f in fold_results]))
    return {'file': basename, 'rows': len(y), 'avg_val_acc': avg_acc, 'folds': fold_results}


def main():
    files = find_csv_files(DATA_DIR)
    logging.info('Found %d csv files', len(files))
    results = []
    for f in files:
        try:
            res = run_for_file(f)
            if res is not None:
                results.append(res)
        except Exception as e:
            logging.error('Fatal error processing %s: %s', f, traceback.format_exc())
    if len(results) > 0:
        df = pd.json_normalize(results)
        df.to_csv(OUT_SUMMARY, index=False)
        logging.info('Summary saved to: %s', OUT_SUMMARY)
    else:
        logging.warning('No results to save')

if __name__ == '__main__':
    main()
