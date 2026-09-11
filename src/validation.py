"""Phase 3 -- validate captures and produce a human-readable report.

Two layers of checking:

  PER-CAPTURE  structural and numeric integrity of one file.
  COHORT       cross-file consistency: duplicate geometry, inconsistent mesh
               topology, participants with no usable captures.

On expected counts
------------------
Nothing here hard-codes a vertex or triangle count. The cohort's expected
topology is DERIVED as the modal (most common) counts across the captures that
actually load; captures that deviate are flagged relative to that. The known
ARKit reference topology (1220 / 2304) is used only to annotate the report with
a note, never to pass or fail a file.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .config import (
    ARKIT_REFERENCE_TRIANGLE_COUNT,
    ARKIT_REFERENCE_VERTEX_COUNT,
    OUTPUTS_DIR,
    PLAUSIBLE_FACE_EXTENT_M,
    RAW_DIR,
    VALIDATION_REPORT,
    SchemaMap,
    ensure_dirs,
)
from .data_loader import FaceCapture, iter_captures

ERROR = "ERROR"
WARNING = "WARNING"
INFO = "INFO"


@dataclass
class Issue:
    severity: str
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.message}"


@dataclass
class CaptureReport:
    """Validation outcome for a single file."""

    path: Path
    participant_id: str | None = None
    capture_id: str | None = None
    n_vertices: int | None = None
    n_triangles: int | None = None
    n_nan: int = 0
    n_inf: int = 0
    n_missing_fields: int = 0
    issues: list[Issue] = field(default_factory=list)
    geometry_hash: str | None = None
    loaded: bool = False

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == WARNING]

    @property
    def status(self) -> str:
        if self.errors:
            return "FAIL"
        if self.warnings:
            return "WARN"
        return "PASS"

    def add(self, severity: str, code: str, message: str) -> None:
        self.issues.append(Issue(severity, code, message))

    def render(self, relative_to: Path | None = None) -> str:
        label = self.path
        if relative_to is not None:
            try:
                label = self.path.relative_to(relative_to)
            except ValueError:
                pass
        lines = [str(label).replace("\\", "/"), self.status, ""]
        if self.loaded:
            lines += [
                f"Vertices: {self.n_vertices}",
                f"Triangles: {self.n_triangles}",
                f"NaN values: {self.n_nan}",
                f"Infinite values: {self.n_inf}",
                f"Missing values: {self.n_missing_fields}",
            ]
        for issue in self.issues:
            lines.append(f"  {issue}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Per-capture checks
# --------------------------------------------------------------------------

def validate_capture(capture: FaceCapture, path: Path | None = None) -> CaptureReport:
    """Structural and numeric checks on one loaded capture."""
    report = CaptureReport(
        path=path or capture.source_path or Path(capture.capture_id),
        participant_id=capture.participant_id,
        capture_id=capture.capture_id,
        n_vertices=capture.n_vertices,
        n_triangles=capture.n_triangles,
        geometry_hash=capture.geometry_hash(),
        loaded=True,
        n_missing_fields=len(capture.missing_fields),
    )

    verts = capture.vertices
    tris = capture.triangles

    # -- numeric integrity -------------------------------------------------
    nan_mask = np.isnan(verts)
    inf_mask = np.isinf(verts)
    report.n_nan = int(nan_mask.sum())
    report.n_inf = int(inf_mask.sum())
    if report.n_nan:
        rows = int(np.any(nan_mask, axis=1).sum())
        report.add(ERROR, "nan_vertices", f"{report.n_nan} NaN coordinate(s) across {rows} vertex/vertices")
    if report.n_inf:
        rows = int(np.any(inf_mask, axis=1).sum())
        report.add(ERROR, "inf_vertices", f"{report.n_inf} infinite coordinate(s) across {rows} vertex/vertices")

    # -- shape -------------------------------------------------------------
    if verts.ndim != 2 or verts.shape[1] != 3:
        report.add(ERROR, "vertex_shape", f"vertices have shape {verts.shape}, expected (N, 3)")
    if verts.shape[0] == 0:
        report.add(ERROR, "no_vertices", "capture contains zero vertices")
    if tris.ndim != 2 or tris.shape[1] != 3:
        report.add(ERROR, "triangle_shape", f"triangles have shape {tris.shape}, expected (M, 3)")
    if tris.shape[0] == 0:
        report.add(ERROR, "no_topology", "capture contains no triangle topology")

    # -- identity ----------------------------------------------------------
    if not capture.participant_id or str(capture.participant_id).strip() in {"", "None", "null"}:
        report.add(ERROR, "missing_participant_id", "participant_id is empty")
    if not capture.capture_id or str(capture.capture_id).strip() in {"", "None", "null"}:
        report.add(WARNING, "missing_capture_id", "capture_id is empty")

    # -- declared vs actual ------------------------------------------------
    if capture.declared_vertex_count is not None and capture.declared_vertex_count != capture.n_vertices:
        report.add(
            ERROR, "vertex_count_mismatch",
            f"file declares vertexCount={capture.declared_vertex_count} but {capture.n_vertices} vertices were parsed",
        )
    if capture.declared_triangle_count is not None and capture.declared_triangle_count != capture.n_triangles:
        report.add(
            ERROR, "triangle_count_mismatch",
            f"file declares triangleCount={capture.declared_triangle_count} but {capture.n_triangles} triangles were parsed",
        )

    # -- topology sanity ---------------------------------------------------
    if tris.size:
        degenerate = int(
            np.sum((tris[:, 0] == tris[:, 1]) | (tris[:, 1] == tris[:, 2]) | (tris[:, 0] == tris[:, 2]))
        )
        if degenerate:
            report.add(WARNING, "degenerate_triangles", f"{degenerate} triangle(s) repeat a vertex index")

        referenced = np.unique(tris)
        orphans = capture.n_vertices - referenced.size
        if orphans > 0:
            report.add(
                WARNING, "unreferenced_vertices",
                f"{orphans} vertex/vertices are not used by any triangle",
            )

    # -- geometric plausibility (catches unit errors and dead meshes) ------
    if verts.size and report.n_nan == 0 and report.n_inf == 0:
        diagonal = capture.bbox_diagonal()
        lo, hi = PLAUSIBLE_FACE_EXTENT_M
        if diagonal == 0:
            report.add(ERROR, "collapsed_mesh", "all vertices are identical (zero extent)")
        elif not (lo <= diagonal <= hi):
            report.add(
                WARNING, "implausible_scale",
                f"bounding-box diagonal is {diagonal:.4f} m, outside the plausible "
                f"range {lo}-{hi} m for a face in metres -- check the `units` field of the export",
            )

        duplicate_vertices = capture.n_vertices - np.unique(verts, axis=0).shape[0]
        if duplicate_vertices > 0:
            report.add(
                INFO, "duplicate_vertex_positions",
                f"{duplicate_vertices} vertex position(s) are exact duplicates of another",
            )

    # -- optional-but-valuable fields --------------------------------------
    if not capture.has_eye_transforms():
        report.add(
            WARNING, "no_eye_transforms",
            "leftEyeTransform/rightEyeTransform absent -- normalisation will fall back to "
            "a landmark-free reference (see docs/NORMALIZATION.md)",
        )
    if not capture.blend_shapes:
        report.add(INFO, "no_blend_shapes", "no blend-shape coefficients in this export")
    if capture.transform is None:
        report.add(INFO, "no_transform", "no anchor transform in this export")

    # -- assumptions surfaced by the loader --------------------------------
    for assumption in capture.assumptions:
        report.add(INFO, "assumption", assumption)

    return report


# --------------------------------------------------------------------------
# Cohort checks
# --------------------------------------------------------------------------

@dataclass
class CohortReport:
    reports: list[CaptureReport] = field(default_factory=list)
    expected_vertices: int | None = None
    expected_triangles: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def n_pass(self) -> int:
        return sum(1 for r in self.reports if r.status == "PASS")

    @property
    def n_warn(self) -> int:
        return sum(1 for r in self.reports if r.status == "WARN")

    @property
    def n_fail(self) -> int:
        return sum(1 for r in self.reports if r.status == "FAIL")

    @property
    def participants(self) -> set[str]:
        return {r.participant_id for r in self.reports if r.participant_id}

    def usable(self) -> list[CaptureReport]:
        return [r for r in self.reports if r.status != "FAIL"]

    def render(self, relative_to: Path | None = None) -> str:
        blocks = ["=" * 62, "LOOKSMATCH VALIDATION REPORT", "=" * 62, ""]
        if self.expected_vertices is not None:
            blocks.append(
                f"Cohort topology (derived from the modal counts of loaded files): "
                f"{self.expected_vertices} vertices / {self.expected_triangles} triangles"
            )
            if (self.expected_vertices == ARKIT_REFERENCE_VERTEX_COUNT
                    and self.expected_triangles == ARKIT_REFERENCE_TRIANGLE_COUNT):
                blocks.append(
                    "  (matches the commonly reported ARKit ARFaceGeometry topology -- informational only)"
                )
            blocks.append("")
        for report in self.reports:
            blocks.append(report.render(relative_to))
            blocks.append("")
        blocks += [
            "-" * 62,
            "SUMMARY",
            "-" * 62,
            f"Files checked:  {len(self.reports)}",
            f"PASS:           {self.n_pass}",
            f"WARN:           {self.n_warn}",
            f"FAIL:           {self.n_fail}",
            f"Participants:   {len(self.participants)}",
        ]
        if self.notes:
            blocks.append("")
            blocks.append("COHORT NOTES")
            for note in self.notes:
                blocks.append(f"  * {note}")
        return "\n".join(blocks)


def validate_cohort(
    root: Path | str = RAW_DIR, schema: SchemaMap | None = None
) -> CohortReport:
    """Validate every capture under ``root`` plus cross-file consistency."""
    schema = schema or SchemaMap.load()
    cohort = CohortReport()

    loaded: list[tuple[FaceCapture, CaptureReport]] = []
    for path, capture, error in iter_captures(root, schema):
        if capture is None:
            report = CaptureReport(path=path, loaded=False)
            report.add(ERROR, "load_failed", error or "unknown error")
            cohort.reports.append(report)
            continue
        report = validate_capture(capture, path)
        cohort.reports.append(report)
        loaded.append((capture, report))

    if not loaded:
        cohort.notes.append("No capture loaded successfully; cohort checks skipped.")
        return cohort

    # -- derive the expected topology from the data itself -----------------
    vertex_counts = Counter(c.n_vertices for c, _ in loaded)
    triangle_counts = Counter(c.n_triangles for c, _ in loaded)
    cohort.expected_vertices = vertex_counts.most_common(1)[0][0]
    cohort.expected_triangles = triangle_counts.most_common(1)[0][0]

    if len(vertex_counts) > 1:
        cohort.notes.append(
            f"Inconsistent vertex counts across the cohort: {dict(vertex_counts)}. "
            "Feature vectors are only comparable within a fixed topology."
        )
    if len(triangle_counts) > 1:
        cohort.notes.append(
            f"Inconsistent triangle counts across the cohort: {dict(triangle_counts)}."
        )

    for capture, report in loaded:
        if capture.n_vertices != cohort.expected_vertices:
            report.add(
                ERROR, "topology_mismatch",
                f"{capture.n_vertices} vertices differs from the cohort's modal "
                f"{cohort.expected_vertices}; this capture is not comparable to the rest",
            )
        if capture.n_triangles != cohort.expected_triangles:
            report.add(
                WARNING, "topology_mismatch_triangles",
                f"{capture.n_triangles} triangles differs from the cohort's modal {cohort.expected_triangles}",
            )

    # -- duplicate geometry ------------------------------------------------
    by_hash: dict[str, list[CaptureReport]] = defaultdict(list)
    for _, report in loaded:
        if report.geometry_hash:
            by_hash[report.geometry_hash].append(report)
    for digest, group in by_hash.items():
        if len(group) > 1:
            names = ", ".join(str(r.path.name) for r in group)
            for report in group:
                report.add(
                    ERROR, "duplicate_geometry",
                    f"byte-identical vertex data shared with: {names} (hash {digest})",
                )
            cohort.notes.append(f"Duplicate geometry group ({len(group)} files): {names}")

    # -- participants ------------------------------------------------------
    per_participant: dict[str, list[CaptureReport]] = defaultdict(list)
    for _, report in loaded:
        if report.participant_id:
            per_participant[report.participant_id].append(report)
    dead = [pid for pid, rs in per_participant.items() if all(r.status == "FAIL" for r in rs)]
    if dead:
        cohort.notes.append(f"Participants with no usable capture: {', '.join(sorted(dead))}")

    multi = {pid: len(rs) for pid, rs in per_participant.items() if len(rs) > 1}
    if multi:
        cohort.notes.append(
            f"{len(multi)} participant(s) have multiple captures. Every capture from a "
            "participant must stay in the same train/test split (src.splits enforces this)."
        )

    return cohort


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate exported ARKit captures.")
    parser.add_argument("root", nargs="?", default=str(RAW_DIR), help="directory to scan (default: data/raw)")
    parser.add_argument("--out", default=str(VALIDATION_REPORT), help="where to write the report")
    parser.add_argument("--quiet", action="store_true", help="write the report but print only the summary")
    args = parser.parse_args(argv)

    ensure_dirs()
    root = Path(args.root)
    cohort = validate_cohort(root)

    if not cohort.reports:
        print(f"No .json captures found under {root}.")
        print("Export captures from the iOS app into data/raw/<participant_id>/ and rerun.")
        return 1

    text = cohort.render(relative_to=root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    if args.quiet:
        print(f"Files: {len(cohort.reports)}  PASS: {cohort.n_pass}  WARN: {cohort.n_warn}  FAIL: {cohort.n_fail}")
    else:
        print(text)
    print(f"\nReport written to {out_path}")
    return 1 if cohort.n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
