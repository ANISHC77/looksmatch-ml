"""Phase 8 -- human ratings as a statistical consensus target.

What the target actually is
---------------------------
The model's target is THE MEAN RATING A PANEL OF HUMAN RATERS GAVE. That is a
measurement of a population's aggregated response, with a standard error and a
confidence interval, not a property of the person being rated. Nothing in this
pipeline establishes that a face "is" a 7.2; it establishes that a particular
group of raters, drawn from a particular population, averaged 7.2, plus or minus
an interval, and that the figure would move with a different panel.

Every aggregate produced here therefore travels with ``n_ratings``, ``std`` and
a confidence interval. Downstream code should weight or filter by these -- a
mean of 3 ratings is not the same evidence as a mean of 30.

Reliability sets the ceiling
----------------------------
If raters disagree with each other, the consensus itself is noisy, and NO model
can predict it beyond that noise. :func:`intraclass_correlation` reports
ICC(2,k), the reliability of the mean rating, which is an approximate upper
bound on the R^2 any model can achieve against this target. Check it BEFORE
concluding that a model underperformed.

Input format (data/labels/ratings.csv)
--------------------------------------
    participant_id,rater_id,rating
    P0001,R001,7
    P0001,R002,8
    P0002,R001,5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .config import LABELS_DIR, RATINGS_CSV, TARGETS_CSV, ensure_dirs

REQUIRED_COLUMNS = ("participant_id", "rater_id", "rating")


@dataclass
class RatingScale:
    """The scale raters used. Stored so aggregates are interpretable later."""

    minimum: float = 1.0
    maximum: float = 10.0

    def out_of_range(self, values: pd.Series) -> pd.Series:
        return (values < self.minimum) | (values > self.maximum)


# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------

def load_ratings(path: Path | str = RATINGS_CSV) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Collect ratings from independent human raters and save them as a CSV with "
            "columns: participant_id,rater_id,rating\n"
            "See data/labels/ratings.csv.template."
        )
    table = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required column(s): {missing}")
    table["rating"] = pd.to_numeric(table["rating"], errors="coerce")
    return table


def validate_ratings(
    ratings: pd.DataFrame,
    scale: RatingScale | None = None,
    known_participants: Sequence[str] | None = None,
) -> list[str]:
    """Return a list of problems. Empty means the ratings are structurally sound."""
    scale = scale or RatingScale()
    problems: list[str] = []

    n_null = int(ratings["rating"].isna().sum())
    if n_null:
        problems.append(f"{n_null} rating(s) are missing or non-numeric")

    valid = ratings.dropna(subset=["rating"])
    bad = int(scale.out_of_range(valid["rating"]).sum())
    if bad:
        problems.append(
            f"{bad} rating(s) fall outside the declared scale "
            f"[{scale.minimum}, {scale.maximum}]"
        )

    duplicates = ratings.duplicated(subset=["participant_id", "rater_id"], keep=False)
    if duplicates.any():
        pairs = ratings.loc[duplicates, ["participant_id", "rater_id"]].drop_duplicates()
        problems.append(
            f"{len(pairs)} participant/rater pair(s) appear more than once; "
            "one rater rating the same person twice double-weights that opinion"
        )

    if known_participants is not None:
        known = set(map(str, known_participants))
        unknown = sorted(set(ratings["participant_id"].astype(str)) - known)
        if unknown:
            shown = ", ".join(unknown[:5]) + (" ..." if len(unknown) > 5 else "")
            problems.append(f"{len(unknown)} rated participant(s) have no capture: {shown}")

    counts = ratings.groupby("participant_id")["rating"].count()
    thin = counts[counts < 3]
    if len(thin):
        problems.append(
            f"{len(thin)} participant(s) have fewer than 3 ratings; their means are "
            "very noisy and their confidence intervals will be wide"
        )

    per_rater = ratings.groupby("rater_id")["rating"].agg(["count", "std"])
    flat = per_rater[(per_rater["count"] >= 5) & (per_rater["std"].fillna(0) == 0)]
    if len(flat):
        problems.append(
            f"{len(flat)} rater(s) gave an identical score to everyone they rated; "
            "they contribute no ranking information"
        )
    return problems


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def aggregate_ratings(
    ratings: pd.DataFrame,
    confidence: float = 0.95,
    bootstrap: bool = False,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-participant consensus statistics.

    The confidence interval is the t-interval for the mean, which needs at least
    two ratings; participants with one rating get a mean but NaN bounds, because
    a single observation carries no information about its own spread. With
    ``bootstrap=True`` a percentile bootstrap interval is used instead, which
    makes no normality assumption -- worth it for small n or skewed scales.
    """
    valid = ratings.dropna(subset=["rating"])
    rng = np.random.default_rng(seed)
    rows = []

    for participant, group in valid.groupby("participant_id"):
        values = group["rating"].to_numpy(dtype=float)
        n = len(values)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if n > 1 else float("nan")
        sem = float(std / np.sqrt(n)) if n > 1 else float("nan")

        if n < 2:
            low = high = float("nan")
        elif bootstrap:
            draws = rng.choice(values, size=(n_boot, n), replace=True).mean(axis=1)
            alpha = (1 - confidence) / 2
            low, high = (float(v) for v in np.quantile(draws, [alpha, 1 - alpha]))
        else:
            margin = stats.t.ppf(0.5 + confidence / 2, df=n - 1) * sem
            low, high = mean - margin, mean + margin

        rows.append({
            "participant_id": participant,
            "mean_rating": mean,
            "median_rating": float(np.median(values)),
            "std_rating": std,
            "sem_rating": sem,
            "n_ratings": n,
            "ci_low": low,
            "ci_high": high,
            "ci_width": float(high - low) if np.isfinite(low) else float("nan"),
            "min_rating": float(values.min()),
            "max_rating": float(values.max()),
        })

    table = pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)
    table.attrs["confidence"] = confidence
    table.attrs["interval_method"] = "bootstrap" if bootstrap else "t"
    return table


# --------------------------------------------------------------------------
# Inter-rater reliability
# --------------------------------------------------------------------------

@dataclass
class ReliabilityResult:
    icc_single: float      # ICC(2,1): reliability of ONE rater
    icc_average: float     # ICC(2,k): reliability of the MEAN of k raters
    n_participants: int
    n_raters: int
    complete_block: bool
    note: str = ""

    def summary(self) -> str:
        lines = [
            f"ICC(2,1) single rater : {self.icc_single:.3f}",
            f"ICC(2,k) mean of {self.n_raters:<3d}  : {self.icc_average:.3f}",
            f"computed on           : {self.n_participants} participants x {self.n_raters} raters"
            + ("" if self.complete_block else "  (largest complete block)"),
            "",
            f"INTERPRETATION: ICC(2,k) = {self.icc_average:.3f} is the reliability of the",
            "mean rating, and therefore an approximate CEILING on the R^2 any model can",
            "reach against this target. A model scoring below it may be underfitting; a",
            "model scoring at or above it is fitting rater noise, not facial geometry.",
        ]
        if self.note:
            lines += ["", f"NOTE: {self.note}"]
        return "\n".join(lines)


def intraclass_correlation(ratings: pd.DataFrame) -> ReliabilityResult | None:
    """ICC(2,1) and ICC(2,k): two-way random effects, absolute agreement.

    Needs a complete participant x rater block. Rating designs are usually
    incomplete (each rater sees a subset), so we greedily extract the largest
    complete block and say what fraction of the data that used.
    """
    matrix = ratings.pivot_table(index="participant_id", columns="rater_id",
                                 values="rating", aggfunc="mean")
    complete = matrix.dropna()
    note = ""

    if complete.shape[0] < 2 or complete.shape[1] < 2:
        # Greedily drop the raters with the most gaps until a block survives.
        working = matrix.copy()
        while working.shape[1] >= 2 and working.dropna().shape[0] < 2:
            working = working.drop(columns=[working.isna().sum().idxmax()])
        complete = working.dropna()
        if complete.shape[0] < 2 or complete.shape[1] < 2:
            return None
        note = (
            f"ratings are incomplete; used the largest complete block "
            f"({complete.shape[0]} participants x {complete.shape[1]} raters) out of "
            f"{matrix.shape[0]} x {matrix.shape[1]}"
        )

    values = complete.to_numpy(dtype=float)
    n, k = values.shape

    grand = values.mean()
    ms_rows = k * ((values.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((values.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    residual = values - values.mean(axis=1, keepdims=True) - values.mean(axis=0, keepdims=True) + grand
    ms_err = (residual ** 2).sum() / ((n - 1) * (k - 1))

    denominator_single = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    denominator_avg = ms_rows + (ms_cols - ms_err) / n
    icc_single = (ms_rows - ms_err) / denominator_single if denominator_single != 0 else float("nan")
    icc_avg = (ms_rows - ms_err) / denominator_avg if denominator_avg != 0 else float("nan")

    return ReliabilityResult(
        icc_single=float(icc_single),
        icc_average=float(icc_avg),
        n_participants=n,
        n_raters=k,
        complete_block=not note,
        note=note,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and aggregate human ratings.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_agg = sub.add_parser("aggregate", help="write targets.csv from ratings.csv")
    p_agg.add_argument("--ratings", default=str(RATINGS_CSV))
    p_agg.add_argument("--out", default=str(TARGETS_CSV))
    p_agg.add_argument("--scale-min", type=float, default=1.0)
    p_agg.add_argument("--scale-max", type=float, default=10.0)
    p_agg.add_argument("--confidence", type=float, default=0.95)
    p_agg.add_argument("--bootstrap", action="store_true")
    p_agg.add_argument("--min-ratings", type=int, default=0,
                       help="drop participants with fewer than this many ratings")

    p_template = sub.add_parser("template", help="write the ratings.csv header template")

    args = parser.parse_args(argv)
    ensure_dirs()

    if args.command == "template":
        path = LABELS_DIR / "ratings.csv.template"
        path.write_text(
            "participant_id,rater_id,rating\n"
            "# One row per (participant, rater) pair. Delete these comment lines.\n"
            "# rating: numeric, on a consistent scale you declare via --scale-min/--scale-max.\n"
            "# Do NOT put rater demographics here; keep them in a separate raters.csv\n"
            "# joined on rater_id, so this file stays a minimal rating record.\n",
            encoding="utf-8",
        )
        print(f"Template written to {path}")
        return 0

    try:
        ratings = load_ratings(args.ratings)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        print("\nRun `python -m src.ratings template` to write a header template.")
        return 1
    scale = RatingScale(args.scale_min, args.scale_max)

    known = None
    features_path = Path("data/processed/features.csv")
    if features_path.exists():
        known = pd.read_csv(features_path, usecols=["participant_id"])["participant_id"].unique()

    problems = validate_ratings(ratings, scale, known)
    print(f"Ratings: {len(ratings)} rows, {ratings['participant_id'].nunique()} participants, "
          f"{ratings['rater_id'].nunique()} raters")
    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  * {problem}")
    else:
        print("No structural problems found.")

    reliability = intraclass_correlation(ratings)
    print("\n--- INTER-RATER RELIABILITY " + "-" * 34)
    if reliability is None:
        print("  Not computable: needs at least 2 participants rated by the same 2+ raters.")
        print("  Design fix: give every rater a shared 'anchor' subset of participants.")
    else:
        print(reliability.summary())

    targets = aggregate_ratings(ratings, args.confidence, args.bootstrap)
    if args.min_ratings > 0:
        before = len(targets)
        targets = targets[targets["n_ratings"] >= args.min_ratings].reset_index(drop=True)
        print(f"\nDropped {before - len(targets)} participant(s) with < {args.min_ratings} ratings.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(out, index=False)

    print("\n--- CONSENSUS TARGETS " + "-" * 40)
    print(f"  participants     : {len(targets)}")
    print(f"  ratings each     : median {targets['n_ratings'].median():.0f}, "
          f"min {targets['n_ratings'].min():.0f}, max {targets['n_ratings'].max():.0f}")
    print(f"  mean_rating      : {targets['mean_rating'].mean():.2f} "
          f"(sd across participants {targets['mean_rating'].std():.2f})")
    print(f"  within-participant sd: median {targets['std_rating'].median():.2f}  <- rater disagreement")
    print(f"  {args.confidence:.0%} CI width  : median {targets['ci_width'].median():.2f}")
    print(f"\nWritten to {out}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
