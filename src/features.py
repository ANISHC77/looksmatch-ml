"""Phase 5 -- geometric feature extraction.

THIS MODULE ONLY MEASURES GEOMETRY.

Nothing here encodes a belief about what is or is not attractive. There are no
weights, no "ideal" ratios, no beauty constants. Every function answers a
question of the form "how long is this?" or "how far apart are these?". Whether
any of it relates to human ratings is an empirical question for the model, and
the model needs human labels to answer it.

Three tiers of feature, distinguished by ``FeatureSet.sources``:

  landmark_free   Computed from the mesh and the normalised frame alone.
                  Always available.
  derived_midline Computed from EXTREMA OF THE MIDLINE CURVE (see
                  :func:`derive_midline_landmarks`). Geometrically defined and
                  reproducible; approximates, but is not identical to, the
                  clinical landmark of the same name.
  eye_transform   Needs ARFaceAnchor eye transforms in the export.
  landmark        Needs indices in config/landmarks.json. NaN until you set them.

A feature that cannot be computed is emitted as NaN with a reason recorded in
``FeatureSet.missing_reasons``. It is never silently replaced with a guess.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import RAW_DIR
from .data_loader import load_capture, load_all
from .landmarks import LandmarkSet
from .normalization import NormalizedFace, normalize_all, normalize_capture
from .symmetry import SymmetryResult, measure_symmetry

N_PROFILE_BANDS = 9  # width/depth sampled at deciles of face height


# ==========================================================================
# Mesh quantities
# ==========================================================================

def surface_area(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """Total area of all triangles."""
    a, b, c = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def convex_hull_measures(vertices: np.ndarray) -> tuple[float, float]:
    """``(hull volume, hull area)``. Well-defined even for an open mesh, unlike
    the mesh's own volume -- ARFaceGeometry is a sheet, not a closed solid, so
    its enclosed volume is meaningless and we never compute it."""
    from scipy.spatial import ConvexHull  # noqa: PLC0415

    try:
        hull = ConvexHull(vertices)
        return float(hull.volume), float(hull.area)
    except Exception:
        return float("nan"), float("nan")


def shape_pca(vertices: np.ndarray) -> np.ndarray:
    """Eigenvalues of the vertex covariance, descending. Shape spread per axis."""
    centred = vertices - vertices.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(np.cov(centred.T))
    return np.sort(eigenvalues)[::-1]


def width_profile(vertices: np.ndarray, n_bands: int = N_PROFILE_BANDS) -> np.ndarray:
    """Left-right extent of the mesh inside each of ``n_bands`` horizontal slabs.

    Band 0 is the most INFERIOR (chin end), band n-1 the most SUPERIOR (forehead
    end), given the canonical frame's +Y = superior. This is a landmark-free
    stand-in for "jaw width", "cheekbone width", "forehead width": it measures
    the same thing -- how wide the face is at a given height -- but at a height
    defined by a proportion of the face rather than by an anatomical point.
    """
    return _profile(vertices, n_bands, axis=0, mode="extent")


def depth_profile(vertices: np.ndarray, n_bands: int = N_PROFILE_BANDS) -> np.ndarray:
    """Maximum anterior projection inside each horizontal slab (nose, chin, brow)."""
    return _profile(vertices, n_bands, axis=2, mode="max")


def _profile(vertices: np.ndarray, n_bands: int, axis: int, mode: str) -> np.ndarray:
    y = vertices[:, 1]
    edges = np.linspace(y.min(), y.max(), n_bands + 1)
    out = np.full(n_bands, np.nan)
    for i in range(n_bands):
        lo, hi = edges[i], edges[i + 1]
        mask = (y >= lo) & (y <= hi if i == n_bands - 1 else y < hi)
        if not mask.any():
            continue
        values = vertices[mask, axis]
        out[i] = float(values.max() - values.min()) if mode == "extent" else float(values.max())
    return out


# ==========================================================================
# Midline curve and the landmarks derivable from it
# ==========================================================================

@dataclass
class MidlineLandmarks:
    """Points found as EXTREMA of the midsagittal profile curve.

    Derivation, in the canonical frame (+Y superior, +Z anterior). We take the
    vertices lying within a narrow band of the midplane, reduce them to a curve
    z(y) by taking the most anterior point in each height bin, smooth it, then:

      nose_tip    global maximum of z              (most anterior midline point)
      nasion      local minimum of z ABOVE nose_tip (the dip at the nose root)
      subnasale   local minimum of z BELOW nose_tip (where nose meets lip)
      pogonion    local maximum of z BELOW subnasale (chin prominence)
      menton      lowest y on the curve            (bottom of the chin)

    These are reproducible geometric definitions, not picked indices. They
    APPROXIMATE the clinical landmarks of the same name; soft-tissue clinical
    definitions differ slightly. Each carries a ``found`` flag -- a curve
    without a clear local extremum yields None rather than a fallback guess.
    """

    y: np.ndarray
    z: np.ndarray
    nose_tip: np.ndarray | None = None
    nasion: np.ndarray | None = None
    subnasale: np.ndarray | None = None
    pogonion: np.ndarray | None = None
    menton: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    def found(self, name: str) -> bool:
        return getattr(self, name, None) is not None


def midline_curve(vertices: np.ndarray, band_fraction: float = 0.04,
                  n_bins: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Reduce the midsagittal strip to a curve ``z(y)``.

    ``band_fraction`` is the half-width of the strip as a fraction of total face
    width. 4% is narrow enough to stay on the midline and wide enough to contain
    vertices in every height bin.
    """
    width = float(vertices[:, 0].max() - vertices[:, 0].min())
    strip = vertices[np.abs(vertices[:, 0]) <= band_fraction * width]
    if len(strip) < n_bins // 2:  # widen rather than fail
        strip = vertices[np.abs(vertices[:, 0]) <= 3 * band_fraction * width]
    if len(strip) == 0:
        return np.array([]), np.array([])

    edges = np.linspace(strip[:, 1].min(), strip[:, 1].max(), n_bins + 1)
    ys, zs = [], []
    for i in range(n_bins):
        mask = (strip[:, 1] >= edges[i]) & (strip[:, 1] < edges[i + 1])
        if mask.any():
            ys.append(0.5 * (edges[i] + edges[i + 1]))
            zs.append(float(strip[mask, 2].max()))
    return np.asarray(ys), np.asarray(zs)


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving average with EDGE padding.

    ``np.convolve(..., mode="same")`` zero-pads, which drags the first and last
    few samples towards zero and fabricates local extrema at both ends of the
    curve -- exactly where the extremum search walks. Edge padding keeps a
    monotone run monotone.
    """
    if len(values) < window or window < 2:
        return values
    pad = window // 2
    kernel = np.ones(window) / window
    return np.convolve(np.pad(values, pad, mode="edge"), kernel, mode="valid")


def derive_midline_landmarks(vertices: np.ndarray) -> MidlineLandmarks:
    """Locate midline points as extrema of the midsagittal curve."""
    y, z = midline_curve(vertices)
    result = MidlineLandmarks(y=y, z=z)
    if len(y) < 10:
        result.notes.append("midline curve too sparse to locate landmarks")
        return result

    zs = _smooth(z)
    order = np.argsort(y)
    y, zs = y[order], zs[order]

    def point(i: int) -> np.ndarray:
        return np.array([0.0, y[i], zs[i]])

    tip = int(np.argmax(zs))
    result.nose_tip = point(tip)

    # Walk OUTWARD from the nose tip and take the FIRST interior local extremum.
    # Nasion is the dip immediately above the nose, not the deepest dip anywhere
    # above it; taking a global extremum over the region can latch onto a point
    # near the edge of the mesh. `margin` excludes the curve ends, where a
    # monotone run produces a spurious "extremum" at the boundary.
    def first_extremum(start: int, direction: int, kind: str, margin: int = 2) -> int | None:
        i = start + direction
        while margin <= i < len(zs) - margin:
            if kind == "min" and zs[i] < zs[i - 1] and zs[i] < zs[i + 1]:
                return i
            if kind == "max" and zs[i] > zs[i - 1] and zs[i] > zs[i + 1]:
                return i
            i += direction
        return None

    idx = first_extremum(tip, +1, "min")
    if idx is not None:
        result.nasion = point(idx)
    else:
        result.notes.append("no interior local minimum above the nose tip: nasion not found")

    sub = first_extremum(tip, -1, "min")
    if sub is not None:
        result.subnasale = point(sub)
    else:
        result.notes.append("no interior local minimum below the nose tip: subnasale not found")

    if sub is not None:
        pog = first_extremum(sub, -1, "max")
        if pog is not None:
            result.pogonion = point(pog)
        else:
            result.notes.append("no interior local maximum below subnasale: pogonion not found")

    result.menton = point(0)
    return result


# ==========================================================================
# Feature extraction
# ==========================================================================

@dataclass
class FeatureSet:
    participant_id: str
    capture_id: str
    values: dict[str, float] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    missing_reasons: dict[str, str] = field(default_factory=dict)

    def add(self, name: str, value: float, source: str, reason: str | None = None) -> None:
        self.values[name] = float(value)
        self.sources[name] = source
        if reason and not np.isfinite(value):
            self.missing_reasons[name] = reason

    @property
    def n_missing(self) -> int:
        return sum(1 for v in self.values.values() if not np.isfinite(v))

    def to_row(self) -> dict[str, float | str]:
        row: dict[str, float | str] = {
            "participant_id": self.participant_id,
            "capture_id": self.capture_id,
        }
        row.update(self.values)
        return row


def _ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def extract_features(
    face: NormalizedFace,
    landmarks: LandmarkSet | None = None,
    symmetry_result: SymmetryResult | None = None,
    symmetry_method: str = "point_to_surface",
) -> FeatureSet:
    """All geometric measurements for one normalised capture.

    Lengths are in NORMALISED units (divided by the scale reference), so they are
    size-invariant ratios. The metric scale itself is kept as
    ``scale_reference_m`` so absolute size is never lost -- ARKit is metrically
    calibrated, which makes real millimetres genuinely available, unlike a
    photo-based pipeline.
    """
    verts = face.vertices
    features = FeatureSet(face.participant_id, face.capture_id)

    # -- absolute scale (kept, not discarded) ------------------------------
    features.add("scale_reference_m", face.scale_reference_m, "landmark_free")

    # -- gross extents ------------------------------------------------------
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    extent = hi - lo
    features.add("face_width", float(extent[0]), "landmark_free")
    features.add("face_height", float(extent[1]), "landmark_free")
    features.add("face_depth", float(extent[2]), "landmark_free")
    features.add("width_height_ratio", _ratio(extent[0], extent[1]), "landmark_free")
    features.add("depth_height_ratio", _ratio(extent[2], extent[1]), "landmark_free")
    features.add("depth_width_ratio", _ratio(extent[2], extent[0]), "landmark_free")

    # -- surface and bulk ---------------------------------------------------
    area = surface_area(verts, face.triangles)
    hull_volume, hull_area = convex_hull_measures(verts)
    features.add("surface_area", area, "landmark_free")
    features.add("convex_hull_volume", hull_volume, "landmark_free")
    features.add("convexity", _ratio(hull_area, area), "landmark_free")

    eigenvalues = shape_pca(verts)
    features.add("pca_elongation", _ratio(eigenvalues[1], eigenvalues[0]), "landmark_free")
    features.add("pca_flatness", _ratio(eigenvalues[2], eigenvalues[0]), "landmark_free")

    # -- width and depth profiles ------------------------------------------
    widths = width_profile(verts)
    depths = depth_profile(verts)
    for i, value in enumerate(widths):
        features.add(f"width_band_{i}", value, "landmark_free")
    for i, value in enumerate(depths):
        features.add(f"depth_band_{i}", value, "landmark_free")

    # Band ratios. Band 0 is inferior (chin/jaw), band 8 superior (forehead).
    # Named by the region each band sits in, WITHOUT claiming a band edge
    # coincides with gonion or zygion -- those need real landmarks.
    features.add("width_lower_upper_ratio", _ratio(widths[1], widths[6]), "landmark_free")
    features.add("width_mid_upper_ratio", _ratio(widths[4], widths[6]), "landmark_free")
    features.add("width_profile_max", float(np.nanmax(widths)), "landmark_free")
    features.add("width_profile_argmax_frac", float(np.nanargmax(widths)) / (len(widths) - 1), "landmark_free")
    features.add("depth_profile_max", float(np.nanmax(depths)), "landmark_free")
    features.add("depth_profile_argmax_frac", float(np.nanargmax(depths)) / (len(depths) - 1), "landmark_free")

    # -- symmetry -----------------------------------------------------------
    if symmetry_result is None:
        symmetry_result = measure_symmetry(verts, face.triangles, method=symmetry_method)
    features.add("symmetry_rms", symmetry_result.symmetry_rms, "landmark_free")
    features.add("symmetry_p95", symmetry_result.symmetry_p95, "landmark_free")
    features.add("symmetry_max", symmetry_result.symmetry_max, "landmark_free")
    features.add("symmetry_midplane_tilt_deg", symmetry_result.plane_tilt_deg, "landmark_free")
    features.add("symmetry_discretization_floor", symmetry_result.effective_floor, "landmark_free")

    # -- midline-derived measurements --------------------------------------
    midline = derive_midline_landmarks(verts)
    reason = "midline extremum not present in the curve"

    def midline_gap(a: str, b: str, name: str) -> None:
        pa, pb = getattr(midline, a), getattr(midline, b)
        if pa is None or pb is None:
            features.add(name, float("nan"), "derived_midline", reason)
        else:
            features.add(name, float(abs(pa[1] - pb[1])), "derived_midline")

    midline_gap("nasion", "subnasale", "middle_third_height")
    midline_gap("subnasale", "menton", "lower_third_height")
    midline_gap("nasion", "menton", "nasion_menton_height")

    upper = float("nan")
    if midline.nasion is not None:
        upper = float(hi[1] - midline.nasion[1])
    features.add("upper_region_height", upper, "derived_midline",
                 "nasion not found (note: mesh may not reach the hairline, so this is "
                 "nasion-to-mesh-top, not the clinical upper third)")

    features.add("third_ratio_middle_lower",
                 _ratio(features.values["middle_third_height"], features.values["lower_third_height"]),
                 "derived_midline", reason)

    if midline.nose_tip is not None and midline.subnasale is not None:
        features.add("nose_projection", float(midline.nose_tip[2] - midline.subnasale[2]), "derived_midline")
        features.add("nose_height", float(abs(midline.nose_tip[1] - midline.subnasale[1])), "derived_midline")
    else:
        features.add("nose_projection", float("nan"), "derived_midline", reason)
        features.add("nose_height", float("nan"), "derived_midline", reason)

    if midline.pogonion is not None and midline.subnasale is not None:
        features.add("chin_projection", float(midline.pogonion[2] - midline.subnasale[2]), "derived_midline")
        features.add("chin_height", float(abs(midline.pogonion[1] - midline.menton[1])), "derived_midline")
    else:
        features.add("chin_projection", float("nan"), "derived_midline", reason)
        features.add("chin_height", float("nan"), "derived_midline", reason)

    if midline.nasion is not None and midline.nose_tip is not None:
        features.add("nasion_depth", float(midline.nose_tip[2] - midline.nasion[2]), "derived_midline")
    else:
        features.add("nasion_depth", float("nan"), "derived_midline", reason)

    # -- eye-transform features --------------------------------------------
    eye_reason = "export has no ARFaceAnchor eye transforms"
    if face.eye_positions is not None:
        left_eye, right_eye = face.eye_positions
        interocular = float(np.linalg.norm(left_eye - right_eye))
        features.add("interocular_distance", interocular, "eye_transform")
        features.add("interocular_face_width_ratio", _ratio(interocular, extent[0]), "eye_transform")
        features.add("eye_height_frac", _ratio(float((left_eye[1] + right_eye[1]) / 2 - lo[1]), extent[1]), "eye_transform")
        features.add("eye_vertical_disparity", float(abs(left_eye[1] - right_eye[1])), "eye_transform")
    else:
        for name in ("interocular_distance", "interocular_face_width_ratio",
                     "eye_height_frac", "eye_vertical_disparity"):
            features.add(name, float("nan"), "eye_transform", eye_reason)

    # -- landmark features (NaN until config/landmarks.json is populated) ---
    landmarks = landmarks or LandmarkSet()
    usable = bool(landmarks) and landmarks.applies_to(len(verts))
    lm_reason = (
        "config/landmarks.json has no indices (see docs/LANDMARKS.md)" if not landmarks
        else "landmark indices were picked on a different mesh topology" if not usable
        else "this landmark is not defined"
    )

    def lm_distance(name: str, a: str, b: str) -> None:
        value = landmarks.distance(verts, a, b) if usable else float("nan")
        features.add(name, value, f"landmark:{a}+{b}", lm_reason)

    lm_distance("lm_eye_spacing_inner", "left_eye_inner", "right_eye_inner")
    lm_distance("lm_eye_spacing_outer", "left_eye_outer", "right_eye_outer")
    lm_distance("lm_left_eye_width", "left_eye_inner", "left_eye_outer")
    lm_distance("lm_right_eye_width", "right_eye_inner", "right_eye_outer")
    lm_distance("lm_nose_width", "nose_left_ala", "nose_right_ala")
    lm_distance("lm_nose_height", "nasion", "subnasale")
    lm_distance("lm_mouth_width", "mouth_left_corner", "mouth_right_corner")
    lm_distance("lm_jaw_width", "left_gonion", "right_gonion")
    lm_distance("lm_cheekbone_width", "left_zygion", "right_zygion")
    lm_distance("lm_chin_height", "subnasale", "menton")
    lm_distance("lm_upper_third", "trichion", "glabella")
    lm_distance("lm_middle_third", "glabella", "subnasale")
    lm_distance("lm_lower_third", "subnasale", "menton")

    features.add("lm_jaw_cheekbone_ratio",
                 _ratio(features.values["lm_jaw_width"], features.values["lm_cheekbone_width"]),
                 "landmark:gonion+zygion", lm_reason)
    features.add("lm_mouth_nose_ratio",
                 _ratio(features.values["lm_mouth_width"], features.values["lm_nose_width"]),
                 "landmark:cheilion+alare", lm_reason)

    return features


def extract_all(
    root: Path | str = RAW_DIR,
    landmarks: LandmarkSet | None = None,
    symmetry_method: str = "point_to_surface",
) -> list[FeatureSet]:
    """Features for every loadable capture under ``root``."""
    landmarks = landmarks if landmarks is not None else LandmarkSet.load()
    faces = normalize_all(load_all(root))
    return [extract_features(f, landmarks, symmetry_method=symmetry_method) for f in faces]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract geometric features from one capture.")
    parser.add_argument("path")
    parser.add_argument("--symmetry-method", default="point_to_surface",
                        choices=("nearest_vertex", "point_to_surface"))
    args = parser.parse_args(argv)

    capture = load_capture(Path(args.path))
    face = normalize_capture(capture)
    features = extract_features(face, LandmarkSet.load(), symmetry_method=args.symmetry_method)

    print(f"{features.participant_id}/{features.capture_id}")
    print(f"{len(features.values)} features, {features.n_missing} unavailable\n")
    by_source: dict[str, list[str]] = {}
    for name, source in features.sources.items():
        by_source.setdefault(source.split(":")[0], []).append(name)
    for source in sorted(by_source):
        print(f"-- {source} " + "-" * (56 - len(source)))
        for name in by_source[source]:
            value = features.values[name]
            print(f"  {name:34s} {value:12.6f}" if np.isfinite(value) else f"  {name:34s} {'NaN':>12s}")
    if features.missing_reasons:
        print("\nWhy features are unavailable:")
        for reason in sorted(set(features.missing_reasons.values())):
            names = [n for n, r in features.missing_reasons.items() if r == reason]
            print(f"  * {reason}\n      ({len(names)} feature(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
