from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier


RANDOM_STATE = 42
DATA_DIR = Path(".")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
OUTPUT_PATH = Path("expected_submission.csv")
EXPECTED_TEST_ROWS = 339
N_FOLDS = 5

FEATURE_COLS = [f"X{i}" for i in range(1, 50)]

STAGE_GROUPS = {
    "entry": ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9"],
    "furnace": ["X10", "X11", "X12", "X13", "X14", "X15", "X16"],
    "mill": ["X17", "X18", "X19", "X20", "X21", "X22"],
    "cool": [
        "X23",
        "X24",
        "X25",
        "X26",
        "X27",
        "X28",
        "X29",
        "X30",
        "X31",
        "X32",
        "X33",
    ],
    "downcoiler": ["X34", "X35", "X36", "X37", "X38", "X39", "X40"],
    "tail": ["X41", "X42", "X43", "X44", "X45", "X46", "X47", "X48", "X49"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a calibrated high-recall classifier for Tata Steel hot rolling defects."
    )
    parser.add_argument("--train", type=Path, default=TRAIN_PATH, help="Path to train.csv")
    parser.add_argument("--test", type=Path, default=TEST_PATH, help="Path to test.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path for expected_submission.csv",
    )
    return parser.parse_args()


def _configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def load_data(
    train_path: Path = TRAIN_PATH, test_path: Path = TEST_PATH
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    required_train_cols = {"CoilID", "Y", *FEATURE_COLS}
    required_test_cols = {"CoilID", *FEATURE_COLS}
    missing_train = sorted(required_train_cols.difference(train.columns))
    missing_test = sorted(required_test_cols.difference(test.columns))
    if missing_train:
        raise ValueError(f"Training data is missing required columns: {missing_train}")
    if missing_test:
        raise ValueError(f"Test data is missing required columns: {missing_test}")

    return train, test


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / (denominator.replace(0, np.nan) + 1)


def make_features(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    base = df.loc[:, feature_cols].copy()
    for col in feature_cols:
        base[col] = pd.to_numeric(base[col], errors="coerce")

    engineered: Dict[str, pd.Series] = {}

    for stage_name, cols in STAGE_GROUPS.items():
        missing_cols = [col for col in cols if col not in base.columns]
        if missing_cols:
            raise ValueError(f"Stage group {stage_name!r} references missing columns: {missing_cols}")

        stage = base.loc[:, cols]
        engineered[f"{stage_name}_mean"] = stage.mean(axis=1)
        engineered[f"{stage_name}_std"] = stage.std(axis=1)
        engineered[f"{stage_name}_min"] = stage.min(axis=1)
        engineered[f"{stage_name}_max"] = stage.max(axis=1)
        engineered[f"{stage_name}_range"] = engineered[f"{stage_name}_max"] - engineered[f"{stage_name}_min"]
        engineered[f"{stage_name}_median"] = stage.median(axis=1)
        engineered[f"{stage_name}_first_last_delta"] = base[cols[-1]] - base[cols[0]]

        for prev_col, curr_col in zip(cols[:-1], cols[1:]):
            engineered[f"{stage_name}_{curr_col}_minus_{prev_col}"] = base[curr_col] - base[prev_col]
            engineered[f"{stage_name}_{curr_col}_to_{prev_col}_ratio"] = _safe_ratio(
                base[curr_col], base[prev_col]
            )

    engineered["log_X35"] = np.log1p(base["X35"].clip(lower=0))
    engineered["log_X34"] = np.log1p(base["X34"].clip(lower=0))

    engineered["X13_X35_ratio"] = _safe_ratio(base["X13"], base["X35"])
    engineered["X13_X10_product"] = base["X13"] * base["X10"]
    engineered["X13_minus_X10"] = base["X13"] - base["X10"]
    engineered["X35_X34_ratio"] = _safe_ratio(base["X35"], base["X34"])
    engineered["X13_log_X35"] = base["X13"] * np.log1p(base["X35"].clip(lower=0))
    engineered["X10_log_X35"] = base["X10"] * np.log1p(base["X35"].clip(lower=0))
    engineered["cool_furnace_interaction"] = base["X30"] * base["X13"]
    engineered["X35_zero_flag"] = (base["X35"] == 0).astype(float)
    engineered["X34_zero_flag"] = (base["X34"] == 0).astype(float)

    for col in feature_cols:
        filled = base[col].fillna(0)
        engineered[f"{col}_slog"] = np.sign(filled) * np.log1p(np.abs(filled))
        engineered[f"{col}_isna"] = base[col].isna().astype(float)

    feature_df = pd.concat([base, pd.DataFrame(engineered, index=base.index)], axis=1)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    return feature_df


def get_models(pos_weight: float) -> Dict[str, object]:
    return {
        "XGBoost": XGBClassifier(
            n_estimators=600,
            max_depth=3,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=2,
            reg_lambda=5.0,
            scale_pos_weight=pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=600,
            learning_rate=0.02,
            num_leaves=15,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=5.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
        "XGBoost_SMOTE": XGBClassifier(
            n_estimators=400,
            max_depth=2,
            learning_rate=0.025,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=1,
            reg_lambda=3.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=1000,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=1000,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def _make_calibrator(model: object) -> CalibratedClassifierCV:
    estimator = clone(model)
    try:
        return CalibratedClassifierCV(estimator=estimator, method="isotonic", cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method="isotonic", cv=3)


def _predict_positive_probability(model: object, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-scores))


def _fit_calibrated_or_fallback(model: object, X_train: np.ndarray, y_train: np.ndarray) -> object:
    calibrated = _make_calibrator(model)
    try:
        calibrated.fit(X_train, y_train)
        return calibrated
    except Exception as exc:
        warnings.warn(
            f"Calibration failed for {model.__class__.__name__}: {exc}. "
            "Falling back to uncalibrated predict_proba.",
            RuntimeWarning,
        )
        fallback = clone(model)
        fallback.fit(X_train, y_train)
        return fallback


def _preprocess_train_valid(
    X_train: pd.DataFrame, X_valid: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, SimpleImputer, RobustScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_train_imp = imputer.fit_transform(X_train)
    X_valid_imp = imputer.transform(X_valid)
    X_train_pp = scaler.fit_transform(X_train_imp)
    X_valid_pp = scaler.transform(X_valid_imp)
    return X_train_pp, X_valid_pp, imputer, scaler


def _maybe_apply_smote(
    model_name: str, X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if "SMOTE" not in model_name:
        return X_train, y_train

    minority_count = int(np.sum(y_train == 1))
    if minority_count < 2:
        warnings.warn(
            f"Skipping SMOTE for {model_name}: only {minority_count} positive sample(s).",
            RuntimeWarning,
        )
        return X_train, y_train

    k_neighbors = max(1, min(5, minority_count - 1))
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
    try:
        return smote.fit_resample(X_train, y_train)
    except Exception as exc:
        warnings.warn(
            f"SMOTE failed for {model_name}: {exc}. Training without SMOTE for this fit.",
            RuntimeWarning,
        )
        return X_train, y_train


def run_cv(X: pd.DataFrame, y: np.ndarray, models: Dict[str, object]) -> Dict[str, np.ndarray]:
    y = np.asarray(y).astype(int)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_scores: Dict[str, np.ndarray] = {}

    for name, base_model in models.items():
        print(f"\nTraining CV model: {name}")
        oof = np.zeros(len(y), dtype=float)

        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
            X_train = X.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            y_train = y[train_idx].copy()

            X_train_pp, X_valid_pp, _, _ = _preprocess_train_valid(X_train, X_valid)
            X_train_fit, y_train_fit = _maybe_apply_smote(name, X_train_pp, y_train)

            fitted = _fit_calibrated_or_fallback(base_model, X_train_fit, y_train_fit)
            oof[valid_idx] = _predict_positive_probability(fitted, X_valid_pp)
            print(f"  fold {fold}/{N_FOLDS} complete")

        oof_scores[name] = np.clip(oof, 0.0, 1.0)

    return oof_scores


def _metric_counts(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> Tuple[int, int, int, float, float]:
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return tp, fp, fn, recall, precision


def select_ensemble(
    y: np.ndarray, oof_scores: Dict[str, np.ndarray]
) -> Tuple[Tuple[str, ...], float, float]:
    y = np.asarray(y).astype(int)
    names = list(oof_scores.keys())
    best_precision = -1.0
    best_combo: Tuple[str, ...] = tuple()
    best_threshold = 0.0

    for size in range(1, len(names) + 1):
        for combo in itertools.combinations(names, size):
            avg_scores = np.mean([oof_scores[name] for name in combo], axis=0)
            positive_scores = avg_scores[y == 1]
            if len(positive_scores) == 0:
                continue

            threshold = float(np.min(positive_scores))
            _, _, fn, _, precision = _metric_counts(y, avg_scores, threshold)
            if fn > 0:
                continue

            is_better = precision > best_precision + 1e-12
            is_tie_with_smaller_combo = (
                abs(precision - best_precision) <= 1e-12
                and best_combo
                and len(combo) < len(best_combo)
            )
            if is_better or is_tie_with_smaller_combo:
                best_precision = precision
                best_combo = combo
                best_threshold = threshold

    if not best_combo:
        raise RuntimeError("No ensemble could achieve 100% recall on OOF scores.")

    return best_combo, best_threshold, best_precision


def _fit_one_final_model(
    name: str, base_model: object, X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()

    X_imp = imputer.fit_transform(X)
    X_test_imp = imputer.transform(X_test)
    X_pp = scaler.fit_transform(X_imp)
    X_test_pp = scaler.transform(X_test_imp)

    X_fit, y_fit = _maybe_apply_smote(name, X_pp, y)
    fitted = _fit_calibrated_or_fallback(base_model, X_fit, y_fit)

    train_scores = _predict_positive_probability(fitted, X_pp)
    test_scores = _predict_positive_probability(fitted, X_test_pp)
    return np.clip(train_scores, 0.0, 1.0), np.clip(test_scores, 0.0, 1.0)


def fit_final(
    X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, selected_names: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(int)
    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))
    pos_weight = neg_count / max(pos_count, 1)
    models = get_models(pos_weight)

    train_score_parts: List[np.ndarray] = []
    test_score_parts: List[np.ndarray] = []

    for name in selected_names:
        if name not in models:
            raise KeyError(f"Selected model {name!r} is not available in get_models().")
        print(f"\nFitting final model: {name}")
        train_scores, test_scores = _fit_one_final_model(name, models[name], X, y, X_test)
        train_score_parts.append(train_scores)
        test_score_parts.append(test_scores)

    train_ensemble = np.mean(train_score_parts, axis=0)
    test_ensemble = np.mean(test_score_parts, axis=0)
    return np.clip(train_ensemble, 0.0, 1.0), np.clip(test_ensemble, 0.0, 1.0)


def _format_names(names: Iterable[str]) -> str:
    return "[" + ", ".join(names) + "]"


def main() -> None:
    _configure_output_encoding()
    args = parse_args()

    print("Loading data...")
    train_df, test_df = load_data(args.train, args.test)
    y = train_df["Y"].astype(int).to_numpy()

    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))
    pos_weight = neg_count / max(pos_count, 1)

    print("Engineering features...")
    X = make_features(train_df, FEATURE_COLS)
    X_test = make_features(test_df, FEATURE_COLS)
    train_feature_columns = X.columns.tolist()
    X_test = X_test.reindex(columns=train_feature_columns)

    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")
    print(f"Feature columns after engineering: {X.shape[1]}")
    print(f"Positive class weight: {pos_weight:.4f}")

    models = get_models(pos_weight)
    oof_scores = run_cv(X, y, models)

    selected_names, oof_threshold, selected_precision = select_ensemble(y, oof_scores)
    oof_ensemble = np.mean([oof_scores[name] for name in selected_names], axis=0)
    oof_tp, oof_fp, oof_fn, oof_recall, oof_precision = _metric_counts(
        y, oof_ensemble, oof_threshold
    )

    print("\nSelected ensemble from OOF scores:")
    print(f"  Models: {_format_names(selected_names)}")
    print(f"  OOF zero-FN threshold: {oof_threshold:.6f}")
    print(f"  OOF precision at selection threshold: {selected_precision:.3f}")

    train_ensemble, test_ensemble = fit_final(X, y, X_test, selected_names)

    oof_zero_fn_threshold = float(np.min(oof_ensemble[y == 1]))
    full_fit_zero_fn_threshold = float(np.min(train_ensemble[y == 1]))
    final_threshold = min(oof_zero_fn_threshold, full_fit_zero_fn_threshold) * 0.98
    final_threshold = float(max(0.0, min(1.0, final_threshold)))

    final_tp, final_fp, final_fn, final_recall, final_precision = _metric_counts(
        y, train_ensemble, final_threshold
    )
    oof_final_tp, oof_final_fp, oof_final_fn, oof_final_recall, oof_final_precision = _metric_counts(
        y, oof_ensemble, final_threshold
    )

    assert final_fn == 0, f"FAIL: {final_fn} false negatives - zero FN required"
    assert final_recall == 1.0, "FAIL: recall < 100%"

    if oof_precision < 0.90:
        print(
            f"WARNING: OOF precision={oof_precision:.3f} < 0.90 - "
            "submission may not score 100 points"
        )
        print(f"Flagging {oof_tp + oof_fp} OOF train coils ({oof_fp} FP). Ideal: <=73 total.")

    if final_precision < 0.90:
        print(
            f"WARNING: final-fit train precision={final_precision:.3f} < 0.90 - "
            "submission may not score 100 points"
        )
        print(
            f"Flagging {final_tp + final_fp} final-fit train coils ({final_fp} FP). "
            "Ideal: <=73 total."
        )
    else:
        print(f"PASS: recall=100% precision={final_precision:.3f} - meets all requirements")

    test_preds = (test_ensemble >= final_threshold).astype(int)
    submission = pd.DataFrame({"CoilID": test_df["CoilID"], "Y": test_preds})

    if submission.shape != (EXPECTED_TEST_ROWS, 2):
        raise AssertionError(
            f"Submission shape must be ({EXPECTED_TEST_ROWS}, 2), got {submission.shape}."
        )
    if list(submission.columns) != ["CoilID", "Y"]:
        raise AssertionError(f"Submission columns must be ['CoilID', 'Y'], got {list(submission.columns)}.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)

    print("\n=== VALIDATION SUMMARY ===")
    print(
        f"OOF ensemble: recall={oof_recall:.3f}, precision={oof_precision:.3f}, "
        f"FP={oof_fp}, FN={oof_fn}"
    )
    print(
        f"OOF at final threshold: recall={oof_final_recall:.3f}, "
        f"precision={oof_final_precision:.3f}, FP={oof_final_fp}, FN={oof_final_fn}"
    )
    print(
        f"Final-fit train: recall={final_recall:.3f}, precision={final_precision:.3f}, "
        f"FP={final_fp}, FN={final_fn}"
    )
    print(f"Final threshold: {final_threshold:.6f}")
    print(f"Selected models: {_format_names(selected_names)}")
    print(f"Test predictions: {int(test_preds.sum())} positives out of {len(test_preds)}")
    print(f"Submission saved: {args.output} ✓")


if __name__ == "__main__":
    main()
