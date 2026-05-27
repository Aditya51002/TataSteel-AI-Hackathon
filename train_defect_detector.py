from __future__ import annotations

import argparse
import itertools
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency fallback
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional dependency fallback
    LGBMClassifier = None


RANDOM_STATE = 42
DEFAULT_DATA_DIR = Path(r"c:\Users\adity\Downloads\169df72b552611f1\dataset")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: BaseEstimator | None = None
    use_smote: bool = False
    is_isolation_forest: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a high-recall Tata Steel Alpha defect detector."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_DATA_DIR / "train.csv")
    parser.add_argument("--test", type=Path, default=DEFAULT_DATA_DIR / "test.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("expected_submission.csv"),
        help="Submission CSV path.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for plots and validation reports.",
    )
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def metric_block(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "avg_precision": float(average_precision_score(y_true, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "predicted_positive": int(y_pred.sum()),
    }


def full_recall_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    positive_scores = scores[y_true == 1]
    if len(positive_scores) == 0:
        raise ValueError("No positive labels were found; cannot tune recall threshold.")
    threshold = float(np.nanmin(positive_scores))
    if not np.isfinite(threshold):
        raise ValueError("Non-finite model scores encountered.")
    return max(0.0, min(1.0, threshold))


def build_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            # add_indicator keeps missingness as a signal while median-imputing values.
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", RobustScaler()),
        ]
    )


def build_supervised_pipeline(spec: ModelSpec, y_train_fold: np.ndarray) -> ImbPipeline:
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", RobustScaler()),
    ]
    if spec.use_smote:
        minority_count = int(np.sum(y_train_fold == 1))
        k_neighbors = max(1, min(5, minority_count - 1))
        steps.append(
            (
                "smote",
                SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors),
            )
        )
    steps.append(("model", clone(spec.estimator)))
    return ImbPipeline(steps=steps)


def normalize_anomaly_scores(
    train_scores: np.ndarray, target_scores: np.ndarray
) -> np.ndarray:
    scaler = MinMaxScaler()
    scaler.fit(train_scores.reshape(-1, 1))
    return np.clip(scaler.transform(target_scores.reshape(-1, 1)).ravel(), 0.0, 1.0)


def cv_isolation_forest(
    X: pd.DataFrame, y: np.ndarray, folds: StratifiedKFold
) -> tuple[np.ndarray, list[dict[str, float]]]:
    oof = np.zeros(len(y), dtype=float)
    fold_rows: list[dict[str, float]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(X, y), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_valid = y[valid_idx]

        preprocessor = build_preprocessor()
        X_train_pp = preprocessor.fit_transform(X_train)
        X_valid_pp = preprocessor.transform(X_valid)

        model = IsolationForest(
            n_estimators=500,
            contamination=float(np.mean(y[train_idx])),
            random_state=RANDOM_STATE + fold_idx,
            n_jobs=-1,
        )
        model.fit(X_train_pp)

        train_anomaly = -model.decision_function(X_train_pp)
        valid_anomaly = -model.decision_function(X_valid_pp)
        valid_scores = normalize_anomaly_scores(train_anomaly, valid_anomaly)
        oof[valid_idx] = valid_scores

        fold_rows.append(
            {
                "model": "IsolationForest",
                "fold": fold_idx,
                "roc_auc": roc_auc_score(y_valid, valid_scores),
                "avg_precision": average_precision_score(y_valid, valid_scores),
            }
        )

    return oof, fold_rows


def fit_predict_isolation_forest(
    X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    preprocessor = build_preprocessor()
    X_train_pp = preprocessor.fit_transform(X_train)
    X_test_pp = preprocessor.transform(X_test)

    model = IsolationForest(
        n_estimators=500,
        contamination=float(np.mean(y_train)),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_pp)

    train_anomaly = -model.decision_function(X_train_pp)
    test_anomaly = -model.decision_function(X_test_pp)
    train_scores = normalize_anomaly_scores(train_anomaly, train_anomaly)
    test_scores = normalize_anomaly_scores(train_anomaly, test_anomaly)
    return train_scores, test_scores


def get_model_specs(pos_weight: float) -> list[ModelSpec]:
    specs: list[ModelSpec] = []

    if XGBClassifier is not None:
        specs.append(
            ModelSpec(
                name="XGBoost",
                estimator=XGBClassifier(
                    n_estimators=500,
                    max_depth=3,
                    learning_rate=0.025,
                    subsample=0.85,
                    colsample_bytree=0.80,
                    min_child_weight=1,
                    reg_lambda=3.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=pos_weight,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                use_smote=False,
            )
        )
        specs.append(
            ModelSpec(
                name="XGBoost_SMOTE",
                estimator=XGBClassifier(
                    n_estimators=350,
                    max_depth=2,
                    learning_rate=0.03,
                    subsample=0.85,
                    colsample_bytree=0.80,
                    min_child_weight=1,
                    reg_lambda=3.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                use_smote=True,
            )
        )

    specs.extend(
        [
            ModelSpec(
                name="RandomForest",
                estimator=RandomForestClassifier(
                    n_estimators=800,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                use_smote=False,
            ),
            ModelSpec(
                name="ExtraTrees",
                estimator=ExtraTreesClassifier(
                    n_estimators=800,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                use_smote=False,
            ),
            ModelSpec(
                name="LogisticRegression",
                estimator=LogisticRegression(
                    class_weight="balanced",
                    C=0.20,
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
                use_smote=False,
            ),
            ModelSpec(
                name="GradientBoosting_SMOTE",
                estimator=GradientBoostingClassifier(
                    n_estimators=400,
                    learning_rate=0.03,
                    max_depth=2,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                ),
                use_smote=True,
            ),
            ModelSpec(
                name="AdaBoost_SMOTE",
                estimator=AdaBoostClassifier(
                    n_estimators=350,
                    learning_rate=0.04,
                    random_state=RANDOM_STATE,
                ),
                use_smote=True,
            ),
            ModelSpec(
                name="BalancedRandomForest",
                estimator=BalancedRandomForestClassifier(
                    n_estimators=800,
                    max_features="sqrt",
                    replacement=True,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                use_smote=False,
            ),
            ModelSpec(
                name="EasyEnsemble",
                estimator=EasyEnsembleClassifier(
                    n_estimators=25,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                use_smote=False,
            ),
            ModelSpec(name="IsolationForest", is_isolation_forest=True),
        ]
    )

    if LGBMClassifier is not None:
        specs.insert(
            1,
            ModelSpec(
                name="LightGBM",
                estimator=LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.025,
                    num_leaves=15,
                    max_depth=-1,
                    subsample=0.85,
                    subsample_freq=1,
                    colsample_bytree=0.80,
                    reg_lambda=3.0,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1,
                ),
                use_smote=False,
            ),
        )

    return specs


def plot_correlations(train: pd.DataFrame, feature_cols: list[str], artifacts_dir: Path) -> None:
    correlations = (
        train[feature_cols + ["Y"]]
        .corr(numeric_only=True)["Y"]
        .drop("Y")
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    correlations.to_csv(artifacts_dir / "feature_correlations_with_y.csv")

    plt.figure(figsize=(12, 8))
    sns.barplot(x=correlations.values, y=correlations.index, hue=correlations.values, palette="vlag", legend=False)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Feature Correlation With Defect Label Y")
    plt.xlabel("Pearson correlation")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(artifacts_dir / "feature_correlations_with_y.png", dpi=180)
    plt.close()


def plot_precision_recall(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    artifacts_dir: Path,
    name: str,
) -> None:
    display = PrecisionRecallDisplay.from_predictions(y, scores, name=name)
    y_pred = (scores >= threshold).astype(int)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    plt.scatter([recall], [precision], color="red", label=f"threshold={threshold:.6f}")
    plt.title(f"Precision-Recall Curve: {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifacts_dir / f"precision_recall_{name}.png", dpi=180)
    plt.close()


def print_eda(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> None:
    print("\n=== DATA LOADING & EDA ===")
    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")
    print("\nClass distribution:")
    print(train["Y"].value_counts().sort_index().to_string())
    print("\nFeature dtypes:")
    print(train[feature_cols].dtypes.value_counts().to_string())
    print("\nMissing values per train column:")
    missing = train.isna().sum()
    print(missing[missing > 0].sort_values(ascending=False).to_string())
    print(f"Total train missing values: {int(train.isna().sum().sum())}")
    print(f"Total test missing values:  {int(test.isna().sum().sum())}")


def make_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Create process-stage features without fitting on the data."""
    out = df[feature_cols].copy()

    for col in feature_cols:
        series = df[col]
        filled = series.fillna(0)
        out[f"{col}_isna"] = series.isna().astype(float)
        out[f"{col}_zero"] = filled.eq(0).astype(float)
        out[f"{col}_slog"] = np.sign(filled) * np.log1p(np.abs(filled))

    groups = {
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
        "downcoiler": ["X34", "X35", "X36", "X37", "X38"],
        "tail": ["X41", "X42", "X43", "X44", "X45", "X46", "X47", "X48", "X49"],
    }

    for group_name, cols in groups.items():
        values = df[cols]
        out[f"{group_name}_mean"] = values.mean(axis=1)
        out[f"{group_name}_std"] = values.std(axis=1)
        out[f"{group_name}_min"] = values.min(axis=1)
        out[f"{group_name}_max"] = values.max(axis=1)
        out[f"{group_name}_range"] = values.max(axis=1) - values.min(axis=1)
        out[f"{group_name}_median"] = values.median(axis=1)
        out[f"{group_name}_first_last_delta"] = df[cols[-1]] - df[cols[0]]

        for left, right in zip(cols[:-1], cols[1:]):
            out[f"{right}_minus_{left}"] = df[right] - df[left]
            out[f"{right}_div_{left}"] = df[right] / df[left].replace(0, np.nan)

    cross_stage_pairs = [
        ("X13", "X10"),
        ("X13", "X1"),
        ("X19", "X17"),
        ("X20", "X22"),
        ("X30", "X24"),
        ("X33", "X23"),
        ("X35", "X34"),
        ("X37", "X36"),
        ("X45", "X41"),
    ]
    for left, right in cross_stage_pairs:
        out[f"{left}_minus_{right}"] = df[left] - df[right]
        out[f"{left}_div_{right}"] = df[left] / df[right].replace(0, np.nan)

    return out.replace([np.inf, -np.inf], np.nan)


def evaluate_candidate_ensembles(
    y: np.ndarray, oof_scores: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    names = list(oof_scores)

    for size in range(1, len(names) + 1):
        for subset in itertools.combinations(names, size):
            stacked = np.vstack([oof_scores[name] for name in subset])
            scores = stacked.mean(axis=0)
            threshold = full_recall_threshold(y, scores)
            row = metric_block(y, scores, threshold)
            row["model"] = "+".join(subset)
            row["n_models"] = size
            rows.append(row)

    return pd.DataFrame(rows).sort_values(
        by=["recall", "precision", "fpr", "roc_auc", "n_models"],
        ascending=[False, False, True, False, False],
    )


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=PerformanceWarning)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    feature_cols = [col for col in train.columns if col.startswith("X")]
    if "CoilID" not in train.columns or "CoilID" not in test.columns:
        raise ValueError("Both train and test must contain CoilID.")
    if "Y" not in train.columns:
        raise ValueError("train.csv must contain Y.")

    y = train["Y"].astype(int).to_numpy()
    print_eda(train, test, feature_cols)
    plot_correlations(train, feature_cols, args.artifacts_dir)

    X = make_feature_matrix(train, feature_cols)
    X_test = make_feature_matrix(test, feature_cols).reindex(columns=X.columns)
    print(f"\nModel feature matrix columns: {X.shape[1]}")

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    pos_weight = n_neg / n_pos
    print(f"\nscale_pos_weight: {pos_weight:.4f} ({n_neg}/{n_pos})")

    folds = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=RANDOM_STATE
    )
    specs = get_model_specs(pos_weight=pos_weight)

    oof_scores: dict[str, np.ndarray] = {}
    fold_rows: list[dict[str, float]] = []
    model_summary_rows: list[dict[str, float]] = []

    print("\n=== CROSS-VALIDATION ===")
    for spec in specs:
        print(f"\nTraining {spec.name}...")

        if spec.is_isolation_forest:
            scores, model_fold_rows = cv_isolation_forest(X, y, folds)
            oof_scores[spec.name] = scores
            fold_rows.extend(model_fold_rows)
        else:
            scores = np.zeros(len(y), dtype=float)
            for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(X, y), start=1):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y[train_idx], y[valid_idx]

                pipeline = build_supervised_pipeline(spec, y_train)
                pipeline.fit(X_train, y_train)
                fold_scores = pipeline.predict_proba(X_valid)[:, 1]
                scores[valid_idx] = fold_scores

                fold_rows.append(
                    {
                        "model": spec.name,
                        "fold": fold_idx,
                        "roc_auc": roc_auc_score(y_valid, fold_scores),
                        "avg_precision": average_precision_score(y_valid, fold_scores),
                    }
                )

            oof_scores[spec.name] = scores

        threshold = full_recall_threshold(y, oof_scores[spec.name])
        metrics = metric_block(y, oof_scores[spec.name], threshold)
        metrics["model"] = spec.name
        metrics["n_models"] = 1
        model_summary_rows.append(metrics)
        print(
            "OOF tuned threshold={threshold:.6f} | recall={recall:.4f} | "
            "precision={precision:.4f} | f1={f1:.4f} | fpr={fpr:.4f} | "
            "cm=[[{tn}, {fp}], [{fn}, {tp}]]".format(**metrics)
        )

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(args.artifacts_dir / "fold_metrics.csv", index=False)

    model_metrics = pd.DataFrame(model_summary_rows).sort_values(
        by=["recall", "precision", "fpr", "roc_auc"],
        ascending=[False, False, True, False],
    )
    model_metrics.to_csv(args.artifacts_dir / "model_oof_metrics.csv", index=False)

    candidate_metrics = evaluate_candidate_ensembles(y, oof_scores)
    candidate_metrics.to_csv(
        args.artifacts_dir / "candidate_ensemble_metrics.csv", index=False
    )

    multi_model_candidates = candidate_metrics[candidate_metrics["n_models"] >= 2]
    selected = multi_model_candidates.iloc[0]
    selected_names = str(selected["model"]).split("+")
    oof_ensemble = np.vstack([oof_scores[name] for name in selected_names]).mean(axis=0)
    oof_threshold = full_recall_threshold(y, oof_ensemble)
    selected_metrics = metric_block(y, oof_ensemble, oof_threshold)

    print("\n=== SELECTED SOFT-VOTING ENSEMBLE ===")
    print(f"Models: {', '.join(selected_names)}")
    print(
        "OOF threshold={threshold:.6f} | recall={recall:.4f} | "
        "precision={precision:.4f} | f1={f1:.4f} | fpr={fpr:.4f} | "
        "cm=[[{tn}, {fp}], [{fn}, {tp}]]".format(**selected_metrics)
    )
    if selected_metrics["recall"] < 1.0:
        raise RuntimeError("Selected validation threshold failed the zero-FN rule.")
    if selected_metrics["precision"] < 0.90:
        print(
            "WARNING: No selected validation ensemble achieved precision >= 0.90 "
            "while preserving 100% recall. Submission still prioritizes zero FN."
        )
    if selected_metrics["fpr"] > 0.10:
        print(
            "WARNING: Selected validation ensemble has FPR > 10% at the zero-FN threshold."
        )

    plot_precision_recall(
        y,
        oof_ensemble,
        oof_threshold,
        args.artifacts_dir,
        "selected_ensemble",
    )

    print("\n=== FINAL FIT ON FULL TRAINING DATA ===")
    train_full_scores: dict[str, np.ndarray] = {}
    test_scores: dict[str, np.ndarray] = {}

    for spec in specs:
        if spec.name not in selected_names:
            continue
        print(f"Fitting {spec.name} on all training rows...")

        if spec.is_isolation_forest:
            train_scores, holdout_scores = fit_predict_isolation_forest(X, y, X_test)
        else:
            pipeline = build_supervised_pipeline(spec, y)
            pipeline.fit(X, y)
            train_scores = pipeline.predict_proba(X)[:, 1]
            holdout_scores = pipeline.predict_proba(X_test)[:, 1]

        train_full_scores[spec.name] = train_scores
        test_scores[spec.name] = holdout_scores

    train_full_ensemble = np.vstack(
        [train_full_scores[name] for name in selected_names]
    ).mean(axis=0)
    test_ensemble = np.vstack([test_scores[name] for name in selected_names]).mean(axis=0)

    # Keep the final threshold no higher than both the OOF zero-FN threshold and
    # the full-fit minimum positive training score. This is conservative for recall.
    full_fit_threshold = full_recall_threshold(y, train_full_ensemble)
    final_threshold = min(oof_threshold, full_fit_threshold)
    final_threshold = max(0.0, min(1.0, final_threshold))
    final_train_metrics = metric_block(y, train_full_ensemble, final_threshold)

    print(
        "Final threshold={:.6f} (OOF {:.6f}, full-fit {:.6f})".format(
            final_threshold, oof_threshold, full_fit_threshold
        )
    )
    print(
        "Full-train sanity metrics at final threshold: recall={recall:.4f} | "
        "precision={precision:.4f} | fpr={fpr:.4f} | "
        "cm=[[{tn}, {fp}], [{fn}, {tp}]]".format(**final_train_metrics)
    )

    test_pred = (test_ensemble >= final_threshold).astype(int)
    submission = pd.DataFrame({"CoilID": test["CoilID"], "Y": test_pred.astype(int)})
    submission.to_csv(args.output, index=False)

    pd.DataFrame(
        {
            "CoilID": train["CoilID"],
            "Y": y,
            "selected_ensemble_oof_score": oof_ensemble,
            "selected_ensemble_oof_pred": (oof_ensemble >= oof_threshold).astype(int),
        }
    ).to_csv(args.artifacts_dir / "oof_selected_ensemble_scores.csv", index=False)

    pd.DataFrame(
        {
            "CoilID": test["CoilID"],
            "selected_ensemble_score": test_ensemble,
            "Y": test_pred,
        }
    ).to_csv(args.artifacts_dir / "test_scores.csv", index=False)

    print("\n=== SUBMISSION ===")
    print(f"Saved: {args.output.resolve()}")
    print(f"Shape: {submission.shape}")
    print("Prediction distribution:")
    print(submission["Y"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
