"""Phase 9 -- classical baseline models.

This module WILL NOT INVENT LABELS. It requires a real
``data/processed/targets.csv`` built by :mod:`src.ratings` from real human
ratings, and exits with an explanation if one is not there. There is no demo
mode and no synthetic target, because a model trained on fabricated ratings
would produce numbers that look exactly like real results.

Design points that matter for correctness
-----------------------------------------
* SPLITTING is at the participant level, always (see :mod:`src.splits`).
* PREPROCESSING (imputation, scaling) lives INSIDE a scikit-learn Pipeline, so
  it is fitted on training folds only. Imputing before splitting would leak the
  test set's distribution into training.
* SAMPLE WEIGHTS default to the number of ratings behind each target, so a mean
  of 30 ratings counts for more than a mean of 3. Disable with --no-weights.
* THE BASELINE to beat is "always predict the training mean". An R^2 near zero
  means the model has learned nothing, however good the MAE looks.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURES_CSV, MODELS_DIR, OUTPUTS_DIR, TARGETS_CSV, ensure_dirs
from .dataset import ID_COLUMNS, load_feature_table
from .evaluate import RegressionMetrics, evaluate_predictions
from .splits import assert_no_leakage, participant_folds, participant_split

TARGET_COLUMN = "mean_rating"


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401, PLC0415
        return True
    except ImportError:
        return False


def build_model(name: str, seed: int = 42) -> Pipeline:
    """A preprocessing+estimator pipeline. Preprocessing is fitted per fold."""
    if name == "linear":
        estimator: Any = LinearRegression()
        scale = True
    elif name == "ridge":
        from sklearn.linear_model import RidgeCV  # noqa: PLC0415
        estimator = RidgeCV(alphas=np.logspace(-3, 3, 25))
        scale = True
    elif name == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            random_state=seed, n_jobs=-1,
        )
        scale = False
    elif name == "xgboost":
        if not xgboost_available():
            raise ImportError("xgboost is not installed; pip install xgboost")
        from xgboost import XGBRegressor  # noqa: PLC0415
        estimator = XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=3,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=2.0, random_state=seed, n_jobs=-1,
            # Tree models handle NaN natively, but we impute anyway so every
            # model sees identical inputs and their scores stay comparable.
        )
        scale = False
    else:
        raise ValueError(f"unknown model {name!r}")

    steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


MODELS = ("linear", "ridge", "random_forest", "xgboost")


# --------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------

@dataclass
class TrainingData:
    table: pd.DataFrame
    feature_names: list[str]
    dropped: list[str] = field(default_factory=list)

    @property
    def X(self) -> pd.DataFrame:
        return self.table[self.feature_names]

    @property
    def y(self) -> np.ndarray:
        return self.table[TARGET_COLUMN].to_numpy(dtype=float)

    @property
    def weights(self) -> np.ndarray:
        return self.table["n_ratings"].to_numpy(dtype=float)


def assemble(
    features_path: Path | str = FEATURES_CSV,
    targets_path: Path | str = TARGETS_CSV,
    min_ratings: int = 1,
) -> TrainingData:
    """Join features to consensus targets on ``participant_id``."""
    targets_path = Path(targets_path)
    if not targets_path.exists():
        raise FileNotFoundError(
            f"{targets_path} not found.\n\n"
            "Training needs real human ratings. The pipeline stops here until they exist:\n"
            "  1. Collect ratings into data/labels/ratings.csv\n"
            "     (columns: participant_id,rater_id,rating)\n"
            "  2. Run: python -m src.ratings aggregate\n"
            "  3. Rerun this command.\n\n"
            "No synthetic labels are generated, deliberately -- a model trained on\n"
            "invented ratings produces output indistinguishable from a real result."
        )

    features = load_feature_table(features_path)
    targets = pd.read_csv(targets_path)
    if TARGET_COLUMN not in targets.columns:
        raise ValueError(f"{targets_path.name} has no '{TARGET_COLUMN}' column")

    targets = targets[targets["n_ratings"] >= min_ratings]
    merged = features.merge(
        targets[["participant_id", TARGET_COLUMN, "n_ratings", "std_rating"]],
        on="participant_id", how="inner",
    )
    if merged.empty:
        raise ValueError(
            "No participant has both features and ratings. Check that participant_id "
            "values match exactly between features.csv and targets.csv."
        )

    candidates = [c for c in features.columns if c not in ID_COLUMNS]
    dropped = [c for c in candidates if merged[c].isna().all() or merged[c].nunique(dropna=True) <= 1]
    feature_names = [c for c in candidates if c not in dropped]
    if not feature_names:
        raise ValueError("Every feature column is empty or constant; nothing to train on.")
    return TrainingData(merged, feature_names, dropped)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def cross_validate(
    data: TrainingData, model_name: str, n_splits: int = 5,
    seed: int = 42, use_weights: bool = True,
) -> tuple[RegressionMetrics, np.ndarray]:
    """Grouped CV producing one out-of-fold prediction per row."""
    predictions = np.full(len(data.table), np.nan)
    for train_idx, test_idx in participant_folds(data.table, n_splits):
        pipeline = build_model(model_name, seed)
        fit_kwargs = {}
        if use_weights:
            fit_kwargs["model__sample_weight"] = data.weights[train_idx]
        pipeline.fit(data.X.iloc[train_idx], data.y[train_idx], **fit_kwargs)
        predictions[test_idx] = pipeline.predict(data.X.iloc[test_idx])
    return evaluate_predictions(data.y, predictions, label=f"{model_name} (out-of-fold)"), predictions


def train_final(
    data: TrainingData, model_name: str, test_size: float = 0.2,
    seed: int = 42, use_weights: bool = True,
) -> tuple[Pipeline, RegressionMetrics, RegressionMetrics, pd.DataFrame]:
    """Fit on a training split and score on held-out participants."""
    split = participant_split(data.table, test_size=test_size, seed=seed)
    assert_no_leakage(split.train, split.test)

    pipeline = build_model(model_name, seed)
    fit_kwargs = {"model__sample_weight": split.train["n_ratings"].to_numpy(float)} if use_weights else {}
    pipeline.fit(split.train[data.feature_names], split.train[TARGET_COLUMN], **fit_kwargs)

    train_pred = pipeline.predict(split.train[data.feature_names])
    test_pred = pipeline.predict(split.test[data.feature_names])
    train_metrics = evaluate_predictions(
        split.train[TARGET_COLUMN].to_numpy(float), train_pred, label=f"{model_name} (train)")
    test_metrics = evaluate_predictions(
        split.test[TARGET_COLUMN].to_numpy(float), test_pred, label=f"{model_name} (held-out)")

    predictions = split.test[ID_COLUMNS + [TARGET_COLUMN, "n_ratings"]].copy()
    predictions["predicted"] = test_pred
    predictions["error"] = predictions["predicted"] - predictions[TARGET_COLUMN]
    return pipeline, train_metrics, test_metrics, predictions


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train baseline models on human-rating targets.")
    parser.add_argument("--features", default=str(FEATURES_CSV))
    parser.add_argument("--targets", default=str(TARGETS_CSV))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-ratings", type=int, default=3,
                        help="ignore participants with fewer ratings than this")
    parser.add_argument("--no-weights", action="store_true",
                        help="do not weight samples by the number of ratings behind each target")
    parser.add_argument("--save", action="store_true", help="write the best model to models/")
    args = parser.parse_args(argv)

    ensure_dirs()
    try:
        data = assemble(args.features, args.targets, args.min_ratings)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    n_participants = data.table["participant_id"].nunique()
    print(f"Rows: {len(data.table)}   Participants: {n_participants}   Features: {len(data.feature_names)}")
    if data.dropped:
        print(f"Dropped {len(data.dropped)} empty/constant column(s): {', '.join(data.dropped[:6])}"
              + (" ..." if len(data.dropped) > 6 else ""))

    if n_participants < 30:
        print(f"\nWARNING: {n_participants} participants is a very small cohort. Held-out")
        print("metrics will be dominated by which people landed in the test split, and")
        print("model comparisons at this size are not meaningful. Treat results as a")
        print("pipeline smoke test, not evidence about facial geometry.")
    if n_participants < args.folds:
        print(f"\nCannot run {args.folds}-fold CV with {n_participants} participants.")
        return 1

    use_weights = not args.no_weights
    print(f"Sample weighting: {'n_ratings' if use_weights else 'off'}\n")

    results: dict[str, dict] = {}
    for name in args.models:
        if name == "xgboost" and not xgboost_available():
            print(f"-- {name}: not installed, skipped")
            continue
        print(f"-- {name} " + "-" * (58 - len(name)))
        cv_metrics, _ = cross_validate(data, name, args.folds, args.seed, use_weights)
        print(cv_metrics.summary(indent="   "))
        pipeline, train_metrics, test_metrics, predictions = train_final(
            data, name, args.test_size, args.seed, use_weights)
        print(test_metrics.summary(indent="   "))
        gap = train_metrics.r2 - test_metrics.r2
        if gap > 0.25:
            print(f"   NOTE: train R^2 exceeds held-out R^2 by {gap:.2f} -- overfitting.")
        print()
        results[name] = {
            "cv": cv_metrics.to_dict(),
            "train": train_metrics.to_dict(),
            "test": test_metrics.to_dict(),
        }
        if args.save:
            predictions.to_csv(OUTPUTS_DIR / f"predictions_{name}.csv", index=False)

    if not results:
        print("No model ran.")
        return 1

    best = max(results, key=lambda k: results[k]["cv"]["r2"])
    print("=" * 62)
    print(f"Best by out-of-fold R^2: {best} (R^2 = {results[best]['cv']['r2']:.3f})")
    print("=" * 62)
    print("\nR^2 <= 0 means the model does no better than predicting the mean.")
    print("Before concluding anything, compare against the ICC(2,k) ceiling from")
    print("`python -m src.ratings aggregate` -- no model can beat rater reliability.")

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(data.table)),
        "n_participants": int(n_participants),
        "features": data.feature_names,
        "dropped_features": data.dropped,
        "sample_weighting": "n_ratings" if use_weights else None,
        "seed": args.seed,
        "results": results,
        "best_model": best,
    }
    path = OUTPUTS_DIR / "training_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {path}")

    if args.save:
        import pickle  # noqa: PLC0415

        pipeline, *_ = train_final(data, best, args.test_size, args.seed, use_weights)
        model_path = MODELS_DIR / f"{best}.pkl"
        with model_path.open("wb") as fh:
            pickle.dump({"pipeline": pipeline, "features": data.feature_names, "report": report}, fh)
        print(f"Model written to {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
