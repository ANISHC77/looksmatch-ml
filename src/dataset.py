"""Phase 7 -- turn a directory of captures into one feature table.

Privacy
-------
``features.csv`` carries ONLY the pseudonymous ``participant_id`` /
``capture_id`` plus numeric geometry. It deliberately excludes anything that
could re-identify a person or a device: no file paths, no timestamps, no device
model, no camera serial, no blend-shape traces, and no vertex coordinates. The
raw JSON in ``data/raw/`` is the identifiable artefact and never leaves the
machine; the CSV is the shareable derivative.

Re-identification caveat: a 70-dimensional set of facial measurements is not
anonymous in the strict sense -- it is a biometric template, and a sufficiently
determined party with reference geometry could match rows back to people. Treat
``features.csv`` as pseudonymous, not anonymous, and keep the ID mapping
separate from it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import FEATURES_CSV, PROCESSED_DIR, RAW_DIR, ensure_dirs
from .data_loader import iter_captures
from .features import FeatureSet, extract_features
from .landmarks import LandmarkSet
from .normalization import (
    AxisConvention,
    NormalizationConfig,
    default_axis_convention,
    generalized_procrustes,
    normalize_capture,
)

# Columns that identify a row but are not model inputs.
ID_COLUMNS = ["participant_id", "capture_id"]


def build_feature_table(
    root: Path | str = RAW_DIR,
    landmarks: LandmarkSet | None = None,
    convention: AxisConvention | None = None,
    config: NormalizationConfig | None = None,
    symmetry_method: str = "point_to_surface",
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Extract features for every capture under ``root``.

    Returns ``(table, problems)``; ``problems`` lists files that were skipped,
    so a silent shrinking of the cohort is impossible.
    """
    landmarks = landmarks if landmarks is not None else LandmarkSet.load()
    config = config or NormalizationConfig()
    convention = convention or default_axis_convention()
    if convention is None:
        raise ValueError(
            "No axis convention found. Run `python -m src.normalization detect --save` "
            "on your real captures first."
        )

    faces, problems = [], []
    for path, capture, error in iter_captures(root):
        if capture is None:
            problems.append(f"{path.name}: {error}")
            continue
        try:
            faces.append(normalize_capture(capture, convention, config))
        except ValueError as exc:
            problems.append(f"{path.name}: {exc}")

    if config.procrustes and faces:
        try:
            faces, _ = generalized_procrustes(faces)
        except ValueError as exc:
            problems.append(f"Procrustes skipped: {exc}")

    rows: list[dict] = []
    sources: dict[str, str] = {}
    for i, face in enumerate(faces, 1):
        if verbose:
            print(f"  [{i}/{len(faces)}] {face.key}", end="\r")
        features: FeatureSet = extract_features(face, landmarks, symmetry_method=symmetry_method)
        rows.append(features.to_row())
        sources.update(features.sources)
    if verbose and faces:
        print(" " * 70, end="\r")

    table = pd.DataFrame(rows)
    if not table.empty:
        ordered = ID_COLUMNS + [c for c in table.columns if c not in ID_COLUMNS]
        table = table[ordered]
    table.attrs["feature_sources"] = sources
    return table, problems


def feature_columns(table: pd.DataFrame) -> list[str]:
    """Numeric model-input columns (everything but the IDs)."""
    return [c for c in table.columns if c not in ID_COLUMNS]


def describe_table(table: pd.DataFrame) -> str:
    """Coverage report: which features are usable, which are entirely missing."""
    columns = feature_columns(table)
    if table.empty:
        return "Table is empty."
    coverage = table[columns].notna().mean().sort_values()
    dead = coverage[coverage == 0.0]
    partial = coverage[(coverage > 0.0) & (coverage < 1.0)]
    constant = [c for c in columns if table[c].notna().any() and table[c].nunique(dropna=True) <= 1]

    lines = [
        f"Rows (captures):      {len(table)}",
        f"Participants:         {table['participant_id'].nunique()}",
        f"Feature columns:      {len(columns)}",
        f"Fully populated:      {int((coverage == 1.0).sum())}",
        f"Partially populated:  {len(partial)}",
        f"Entirely missing:     {len(dead)}",
    ]
    if len(dead):
        lines.append("\nEntirely missing (dropped before modelling):")
        for name in dead.index:
            lines.append(f"  {name}")
    if len(partial):
        lines.append("\nPartially populated:")
        for name, value in partial.items():
            lines.append(f"  {name:34s} {value:.0%}")
    if constant:
        lines.append("\nZero-variance columns (carry no information for a model):")
        for name in constant:
            lines.append(f"  {name}")
    captures_per = table.groupby("participant_id").size()
    if (captures_per > 1).any():
        lines.append(
            f"\n{int((captures_per > 1).sum())} participant(s) have multiple captures "
            f"(max {int(captures_per.max())}). Splits MUST group by participant."
        )
    return "\n".join(lines)


def save_feature_table(table: pd.DataFrame, path: Path | str = FEATURES_CSV) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def load_feature_table(path: Path | str = FEATURES_CSV) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with `python -m src.dataset build`."
        )
    return pd.read_csv(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the feature table from captures.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="extract features and write features.csv")
    p_build.add_argument("root", nargs="?", default=str(RAW_DIR))
    p_build.add_argument("--out", default=str(FEATURES_CSV))
    p_build.add_argument("--origin", default="centroid")
    p_build.add_argument("--scale", default="centroid_size")
    p_build.add_argument("--procrustes", action="store_true",
                         help="apply generalized Procrustes refinement across the cohort")
    p_build.add_argument("--symmetry-method", default="point_to_surface",
                         choices=("nearest_vertex", "point_to_surface"))

    p_describe = sub.add_parser("describe", help="report coverage of an existing features.csv")
    p_describe.add_argument("--path", default=str(FEATURES_CSV))

    args = parser.parse_args(argv)
    ensure_dirs()

    if args.command == "describe":
        table = load_feature_table(args.path)
        print(describe_table(table))
        return 0

    config = NormalizationConfig(origin=args.origin, scale=args.scale, procrustes=args.procrustes)
    print(f"Extracting features from {args.root} ...")
    try:
        table, problems = build_feature_table(
            Path(args.root), config=config, symmetry_method=args.symmetry_method
        )
    except ValueError as exc:
        print(f"\n{exc}")
        return 1

    if problems:
        print(f"\n{len(problems)} capture(s) skipped:")
        for problem in problems:
            print(f"  {problem}")

    if table.empty:
        print("\nNo features extracted. Check `python -m src.validation` first.")
        return 1

    path = save_feature_table(table, args.out)
    print(f"\n{describe_table(table)}")
    print(f"\nWritten to {path}")
    print("\nNEXT: collect human ratings into data/labels/ratings.csv, then run")
    print("      `python -m src.ratings aggregate`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
