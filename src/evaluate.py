"""Phase 9/10 -- regression metrics, fairness auditing, and interpretability.

Metrics
-------
This is a REGRESSION problem, so accuracy is not reported and should not be
requested: it is undefined for continuous targets, and reaching it by bucketing
ratings into "attractive"/"unattractive" would discard most of the signal and
invent a threshold nobody agreed on. The reported metrics are MAE, RMSE, R^2 and
the correlation between predictions and human ratings.

Always read R^2 against two reference points:
  * the "predict the training mean" baseline, reported alongside it here; and
  * the ICC(2,k) rater-reliability ceiling from :mod:`src.ratings`.
A model cannot meaningfully exceed the reliability of its own target.

Fairness
--------
:func:`group_metrics` reports error broken down by a demographic column, to
detect whether the model is WORSE for some groups -- an audit for dataset and
model bias. Demographic attributes are never model inputs, and there is no
per-group calibration: adjusting predictions by group would encode different
standards per group, which is the opposite of the goal.

Interpretability
----------------
:func:`feature_importance` and :func:`shap_analysis` describe WHICH FEATURES THE
MODEL USED. That is a fact about the fitted model on this dataset. It is NOT
evidence that the measured trait causes human ratings. Features in a face are
heavily intercorrelated, so a model will happily lean on one and ignore another
that carries the same information; swapping which one it picks changes the
importance ranking without changing a single prediction. Read importances as
"this is what the model leaned on", never as "this is what makes a face
attractive".
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .config import FEATURES_CSV, OUTPUTS_DIR, TARGETS_CSV, ensure_dirs


# ==========================================================================
# Metrics
# ==========================================================================

@dataclass
class RegressionMetrics:
    label: str
    n: int
    mae: float
    rmse: float
    r2: float
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    baseline_mae: float
    baseline_rmse: float

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def beats_baseline(self) -> bool:
        return self.mae < self.baseline_mae

    def summary(self, indent: str = "") -> str:
        verdict = (
            f"beats the mean-only baseline (MAE {self.baseline_mae:.3f})"
            if self.beats_baseline
            else f"DOES NOT beat the mean-only baseline (MAE {self.baseline_mae:.3f})"
        )
        return "\n".join([
            f"{indent}{self.label}  (n={self.n})",
            f"{indent}  MAE           {self.mae:.4f}",
            f"{indent}  RMSE          {self.rmse:.4f}",
            f"{indent}  R^2           {self.r2:.4f}",
            f"{indent}  Pearson r     {self.pearson_r:.4f}  (p={self.pearson_p:.3g})",
            f"{indent}  Spearman rho  {self.spearman_rho:.4f}  (p={self.spearman_p:.3g})",
            f"{indent}  -> {verdict}",
        ])


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, label: str = "model"
) -> RegressionMetrics:
    """MAE / RMSE / R^2 / correlations, plus a mean-only baseline for reference."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    n = len(y_true)
    if n < 2:
        nan = float("nan")
        return RegressionMetrics(label, n, nan, nan, nan, nan, nan, nan, nan, nan, nan)

    errors = y_pred - y_true
    mae = float(np.abs(errors).mean())
    rmse = float(np.sqrt((errors ** 2).mean()))

    total_ss = float(((y_true - y_true.mean()) ** 2).sum())
    residual_ss = float((errors ** 2).sum())
    r2 = float(1 - residual_ss / total_ss) if total_ss > 0 else float("nan")

    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        pearson_r = pearson_p = spearman_rho = spearman_p = float("nan")
    else:
        pearson_r, pearson_p = (float(v) for v in stats.pearsonr(y_true, y_pred))
        spearman_rho, spearman_p = (float(v) for v in stats.spearmanr(y_true, y_pred))

    baseline = float(y_true.mean())
    return RegressionMetrics(
        label=label, n=n, mae=mae, rmse=rmse, r2=r2,
        pearson_r=pearson_r, pearson_p=pearson_p,
        spearman_rho=spearman_rho, spearman_p=spearman_p,
        baseline_mae=float(np.abs(y_true - baseline).mean()),
        baseline_rmse=float(np.sqrt(((y_true - baseline) ** 2).mean())),
    )


# ==========================================================================
# Fairness auditing
# ==========================================================================

def group_metrics(
    predictions: pd.DataFrame,
    group_column: str,
    truth_column: str = "mean_rating",
    prediction_column: str = "predicted",
    min_group_size: int = 10,
) -> pd.DataFrame:
    """Per-group error, to detect whether the model serves some groups worse.

    Demographic data must be collected with informed consent, and is used here
    ONLY to audit performance. It is never a model input and never adjusts a
    prediction.

    Groups smaller than ``min_group_size`` are reported but flagged: a MAE from
    six people is mostly noise, and acting on it is its own kind of error.
    """
    if group_column not in predictions.columns:
        raise ValueError(f"no '{group_column}' column in the predictions table")

    rows = []
    for name, group in predictions.groupby(group_column):
        metrics = evaluate_predictions(
            group[truth_column].to_numpy(float),
            group[prediction_column].to_numpy(float),
            label=str(name),
        )
        rows.append({
            "group": name,
            "n": metrics.n,
            "mae": metrics.mae,
            "rmse": metrics.rmse,
            "r2": metrics.r2,
            "mean_true": float(group[truth_column].mean()),
            "mean_predicted": float(group[prediction_column].mean()),
            "mean_bias": float((group[prediction_column] - group[truth_column]).mean()),
            "underpowered": metrics.n < min_group_size,
        })
    return pd.DataFrame(rows).sort_values("mae", ascending=False).reset_index(drop=True)


def fairness_report(table: pd.DataFrame) -> str:
    """Narrative summary of a :func:`group_metrics` table."""
    lines = ["Per-group error (audit for dataset/model bias):", ""]
    lines.append(f"  {'group':<18s} {'n':>5s} {'MAE':>8s} {'RMSE':>8s} {'bias':>8s}")
    for _, row in table.iterrows():
        flag = "  (underpowered)" if row["underpowered"] else ""
        lines.append(
            f"  {str(row['group']):<18s} {row['n']:5d} {row['mae']:8.4f} "
            f"{row['rmse']:8.4f} {row['mean_bias']:+8.4f}{flag}"
        )
    solid = table[~table["underpowered"]]
    if len(solid) >= 2:
        spread = float(solid["mae"].max() - solid["mae"].min())
        worst = solid.iloc[0]["group"]
        best = solid.iloc[-1]["group"]
        lines += [
            "",
            f"  MAE spread across adequately-sized groups: {spread:.4f}",
            f"  worst-served: {worst}    best-served: {best}",
            "",
            "  A large spread means the model is less useful for some groups, usually",
            "  because they are underrepresented in training or rated by a panel whose",
            "  composition differs. The fix is data collection and rater-panel design,",
            "  NOT per-group prediction adjustment.",
        ]
    else:
        lines += ["", "  Too few adequately-sized groups to compare."]
    return "\n".join(lines)


# ==========================================================================
# Interpretability
# ==========================================================================

def feature_importance(pipeline, feature_names: Sequence[str]) -> pd.DataFrame:
    """Model-reported importance, if the estimator exposes one.

    Reports the model's internal attribution. See the module docstring: this
    describes the fitted model, not the world.
    """
    estimator = pipeline.named_steps["model"] if hasattr(pipeline, "named_steps") else pipeline
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        kind = "impurity/gain"
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float).ravel())
        kind = "|coefficient| (standardised inputs)"
    else:
        raise TypeError(f"{type(estimator).__name__} exposes no importance attribute")

    table = pd.DataFrame({"feature": list(feature_names), "importance": values})
    table["kind"] = kind
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def permutation_importance_table(
    pipeline, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 20, seed: int = 42
) -> pd.DataFrame:
    """Model-agnostic importance: how much does shuffling a feature hurt?

    More trustworthy than impurity importance, which is biased towards
    high-cardinality features. Still shares the correlated-feature caveat: if two
    features carry the same information, shuffling either one alone barely hurts,
    so both look unimportant.
    """
    from sklearn.inspection import permutation_importance  # noqa: PLC0415

    result = permutation_importance(
        pipeline, X, y, n_repeats=n_repeats, random_state=seed,
        scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    return pd.DataFrame({
        "feature": list(X.columns),
        "importance": result.importances_mean,
        "std": result.importances_std,
        "kind": "permutation (MAE increase)",
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def shap_analysis(pipeline, X: pd.DataFrame, max_samples: int = 500, seed: int = 42):
    """SHAP values for a tree model. Returns ``(shap_values, sampled_X)``."""
    try:
        import shap  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("shap is not installed; pip install shap") from exc

    estimator = pipeline.named_steps["model"] if hasattr(pipeline, "named_steps") else pipeline
    sample = X.sample(min(max_samples, len(X)), random_state=seed) if len(X) > max_samples else X

    # Push the sample through the fitted preprocessing so SHAP sees what the
    # estimator actually saw.
    transformed = sample
    if hasattr(pipeline, "named_steps"):
        for name, step in pipeline.named_steps.items():
            if name == "model":
                break
            transformed = step.transform(transformed)
        transformed = pd.DataFrame(transformed, columns=sample.columns, index=sample.index)

    try:
        explainer = shap.TreeExplainer(estimator)
    except Exception:
        explainer = shap.Explainer(estimator, transformed)
    return explainer(transformed), transformed


CAUSAL_DISCLAIMER = """
READ THIS BEFORE QUOTING ANY IMPORTANCE NUMBER
----------------------------------------------
What an importance ranking tells you:
  * which measurements this particular fitted model leaned on,
  * on this particular dataset,
  * to predict the mean rating of this particular panel of raters.

What it does NOT tell you:
  * that the measured trait CAUSES a higher or lower rating,
  * that changing that trait on a real face would change how people rate it,
  * that the relationship holds for a different population of faces or raters.

Three concrete reasons the ranking can mislead:
  1. COLLINEARITY. Facial measurements are heavily intercorrelated. Where two
     features carry the same information a model picks one arbitrarily; the
     other then looks unimportant despite being equally predictive.
  2. CONFOUNDING. Ratings reflect rater demographics, culture, and exposure.
     A geometric feature can be a proxy for something else entirely -- age, sex,
     ancestry, grooming, or an artefact of how the capture was taken.
  3. TARGET NOISE. The target is itself an estimate with a confidence interval.
     Importances fitted against a noisy target are themselves noisy, and shift
     between reseeds.

Establishing causation would need an intervention -- systematically varying one
geometric feature while holding others fixed and re-rating. This pipeline
performs no such experiment and cannot support causal claims.
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--model", default=None, help="path to a .pkl saved by src.train --save")
    parser.add_argument("--predictions", default=None,
                        help="a predictions CSV from src.train --save")
    parser.add_argument("--demographics", default=None,
                        help="CSV with participant_id plus a demographic column, for the fairness audit")
    parser.add_argument("--group-column", default=None, help="column in --demographics to group by")
    parser.add_argument("--importance", action="store_true", help="print feature importances")
    parser.add_argument("--shap", action="store_true", help="run SHAP analysis (tree models)")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args(argv)
    ensure_dirs()

    if args.predictions:
        predictions = pd.read_csv(args.predictions)
        metrics = evaluate_predictions(
            predictions["mean_rating"].to_numpy(float),
            predictions["predicted"].to_numpy(float),
            label=Path(args.predictions).stem,
        )
        print(metrics.summary())

        if args.demographics and args.group_column:
            demographics = pd.read_csv(args.demographics)
            merged = predictions.merge(demographics, on="participant_id", how="left")
            print()
            print(fairness_report(group_metrics(merged, args.group_column)))

    if args.model:
        import pickle  # noqa: PLC0415

        with open(args.model, "rb") as fh:
            bundle = pickle.load(fh)
        pipeline, feature_names = bundle["pipeline"], bundle["features"]

        if args.importance:
            try:
                table = feature_importance(pipeline, feature_names)
                print(f"\nTop {args.top} features ({table['kind'].iloc[0]}):")
                for _, row in table.head(args.top).iterrows():
                    print(f"  {row['feature']:34s} {row['importance']:.5f}")
            except TypeError as exc:
                print(f"\nNo importance available: {exc}")
            print(CAUSAL_DISCLAIMER)

        if args.shap:
            from .train import assemble  # noqa: PLC0415

            data = assemble()
            values, _ = shap_analysis(pipeline, data.X[feature_names])
            mean_abs = np.abs(values.values).mean(axis=0)
            order = np.argsort(mean_abs)[::-1][:args.top]
            print(f"\nTop {args.top} features by mean |SHAP|:")
            for i in order:
                print(f"  {feature_names[i]:34s} {mean_abs[i]:.5f}")
            print(CAUSAL_DISCLAIMER)

    if not args.model and not args.predictions:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
