"""Participant-level data splitting.

THE RULE
--------
Every capture from one participant must land in the SAME side of any split.

Why this matters more here than in most ML problems: two captures of the same
face are nearly identical geometry. A random capture-level split puts one in
train and one in test, so the model can score well by recognising the individual
rather than learning anything about facial geometry in general. Test error then
reports memorisation, and the model collapses on genuinely new people. With a
handful of captures per participant the inflation is large, and it is invisible
unless you look for it.

Every function here groups by ``participant_id``, and
:func:`assert_no_leakage` RAISES rather than warns. Call it after any split you
did not get from this module.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .config import FEATURES_CSV


class LeakageError(AssertionError):
    """Raised when a participant appears on both sides of a split."""


def assert_no_leakage(train: pd.DataFrame, test: pd.DataFrame, column: str = "participant_id") -> None:
    """Raise if any participant appears in both frames."""
    overlap = set(train[column].astype(str)) & set(test[column].astype(str))
    if overlap:
        shown = ", ".join(sorted(overlap)[:10])
        raise LeakageError(
            f"{len(overlap)} participant(s) appear in BOTH train and test: {shown}. "
            "This inflates test scores. Split with src.splits, never with a random "
            "row-level split."
        )


@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    seed: int

    def summary(self) -> str:
        return "\n".join([
            f"TRAIN: {len(self.train):5d} captures  {self.train['participant_id'].nunique():4d} participants",
            f"TEST : {len(self.test):5d} captures  {self.test['participant_id'].nunique():4d} participants",
            f"seed : {self.seed}",
        ])


def participant_split(
    table: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    column: str = "participant_id",
) -> Split:
    """Hold out a fraction of PARTICIPANTS (not captures) for testing."""
    if column not in table.columns:
        raise ValueError(f"table has no '{column}' column")
    groups = table[column].astype(str)
    n_participants = groups.nunique()
    if n_participants < 2:
        raise ValueError(f"need at least 2 participants to split, found {n_participants}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(table, groups=groups))
    split = Split(
        train=table.iloc[train_idx].reset_index(drop=True),
        test=table.iloc[test_idx].reset_index(drop=True),
        seed=seed,
    )
    assert_no_leakage(split.train, split.test, column)
    return split


def participant_folds(
    table: pd.DataFrame,
    n_splits: int = 5,
    column: str = "participant_id",
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` for grouped k-fold cross-validation."""
    groups = table[column].astype(str)
    n_participants = groups.nunique()
    if n_participants < n_splits:
        raise ValueError(
            f"{n_splits}-fold CV needs at least {n_splits} participants, found {n_participants}"
        )
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(table, groups=groups):
        train_ids = set(groups.iloc[train_idx])
        test_ids = set(groups.iloc[test_idx])
        if train_ids & test_ids:  # GroupKFold guarantees this, but verify anyway
            raise LeakageError("GroupKFold produced overlapping participants")
        yield train_idx, test_idx


def split_report(table: pd.DataFrame, column: str = "participant_id") -> str:
    counts = table.groupby(column).size()
    return "\n".join([
        f"captures      : {len(table)}",
        f"participants  : {counts.size}",
        f"captures each : min {counts.min()}, median {counts.median():.0f}, max {counts.max()}",
        (f"NOTE: {int((counts > 1).sum())} participant(s) have multiple captures, so a "
         "row-level random split WOULD leak." if (counts > 1).any()
         else "Every participant has exactly one capture."),
    ])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a participant-level split.")
    parser.add_argument("--features", default=str(FEATURES_CSV))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=0, help="also preview this many CV folds")
    args = parser.parse_args(argv)

    path = Path(args.features)
    if not path.exists():
        print(f"{path} not found. Build it with `python -m src.dataset build`.")
        return 1

    table = pd.read_csv(path)
    print(split_report(table))
    print()
    split = participant_split(table, args.test_size, args.seed)
    print(split.summary())
    print("\nNo participant appears on both sides (verified).")

    if args.folds:
        print(f"\n{args.folds}-fold grouped CV:")
        for i, (train_idx, test_idx) in enumerate(participant_folds(table, args.folds), 1):
            n_train = table.iloc[train_idx]["participant_id"].nunique()
            n_test = table.iloc[test_idx]["participant_id"].nunique()
            print(f"  fold {i}: train {n_train:4d} participants / test {n_test:4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
