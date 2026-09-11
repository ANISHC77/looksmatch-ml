"""Phase 6 -- bilateral facial symmetry.

Exact definition of the metric
------------------------------
1. MIDLINE. The midsagittal plane is FITTED, not assumed. It is parameterised by
   a unit normal ``n`` and an offset ``d`` (the plane is ``{p : n.p = d}``), and
   both are optimised to minimise the reflection error below. We do not simply
   take ``x = 0``: a real head sits slightly off-axis in the anchor frame, and
   an assumed midline inflates the asymmetry of a perfectly symmetric face.

2. REFLECTION. Every vertex is mirrored through that plane:
       R(p) = p - 2 (n.p - d) n

3. CORRESPONDENCE. Each mirrored vertex is matched to the original mesh, either
   to the nearest point on the nearest TRIANGLE (``point_to_surface``, THE
   DEFAULT) or to the nearest VERTEX (``nearest_vertex``, marginally faster).

   ``nearest_vertex`` carries a DISCRETIZATION FLOOR that biases the metric
   HIGH: a mirrored point generally lands *between* vertices, so the matched
   distance stays positive even on a perfectly symmetric surface. Measured on a
   symmetric test surface sampled irregularly, this artefact alone produced
   rms = 0.062 -- larger than plenty of real asymmetries.

   The floor vanishes if the mesh topology is mirror-paired (each vertex has a
   counterpart at its mirrored position), which we expect of ARFaceGeometry but
   do not assume. ``mirror_pairing_fraction`` reports how many vertices pair up
   -- but note it is CONFOUNDED: pairing also falls on a genuinely asymmetric
   face, so a low value does not distinguish "coarse mesh" from "asymmetric
   person". That ambiguity is exactly why ``point_to_surface`` is the default:
   on an ARKit-sized mesh (~1220 vertices) it costs about 20% more time and has
   no floor at all, so the question never arises. Use ``nearest_vertex`` only
   for bulk exploratory passes, and read its output against
   ``discretization_floor``.

4. ERROR. Per-vertex asymmetry is the matched distance, divided by centroid size
   so the metric is DIMENSIONLESS and independent of head size:
       centroid_size = sqrt(mean_i ||v_i - centroid||^2)

5. SCALAR. The reported metric is the root-mean-square of those per-vertex
   normalised errors.

       symmetry_rms = sqrt( mean_i ( ||R(v_i) - match(R(v_i))|| / centroid_size )^2 )

   LOWER IS MORE SYMMETRIC. 0 means perfect bilateral symmetry. The value is a
   ratio of lengths, so it is comparable across people and capture distances.

We deliberately do NOT convert this into a 0-100 "symmetry score": any such
rescaling implies a calibration we have not established.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .data_loader import load_capture
from .normalization import NormalizedFace, normalize_capture

METHODS = ("nearest_vertex", "point_to_surface")


@dataclass
class SymmetryResult:
    """Outcome of one symmetry measurement."""

    symmetry_rms: float                 # the headline scalar; lower = more symmetric
    symmetry_mean: float
    symmetry_max: float
    symmetry_p95: float
    per_vertex: np.ndarray              # (N,) normalised asymmetry per vertex
    plane_normal: np.ndarray            # (3,) unit normal of the fitted midplane
    plane_offset: float                 # d, in the same units as the input
    centroid_size: float
    method: str
    plane_tilt_deg: float               # angle between fitted normal and the frame's left axis
    converged: bool
    discretization_floor: float = 0.0   # artefact scale: median vertex spacing / centroid size
    mirror_pairing_fraction: float = 0.0  # fraction of vertices with a true mirror counterpart
    notes: list[str] = field(default_factory=list)

    @property
    def effective_floor(self) -> float:
        """Upper bound on the discretization artefact in this measurement.

        Zero for ``point_to_surface``, which matches to the surface itself. For
        ``nearest_vertex`` we scale the raw floor by the fraction of vertices
        that did NOT find a mirror counterpart -- a CONSERVATIVE bound, because
        that fraction also falls when the face is genuinely asymmetric, so the
        bound is loosest exactly where asymmetry is real.
        """
        if self.method == "point_to_surface":
            return 0.0
        return self.discretization_floor * (1.0 - self.mirror_pairing_fraction)

    @property
    def exceeds_floor(self) -> bool:
        """Is the value clearly above the artefact bound?

        False does NOT mean "symmetric" -- for ``nearest_vertex`` it means the
        measurement is ambiguous and should be redone with ``point_to_surface``.
        """
        return self.symmetry_rms > 2.0 * self.effective_floor

    def verdict(self) -> str:
        if self.method == "point_to_surface":
            return "unbiased (point-to-surface matching has no discretization floor)"
        if self.effective_floor < 1e-9:
            return "unbiased (mesh topology is mirror-paired, so the floor cancels)"
        if self.exceeds_floor:
            return f"above the artefact bound ({self.symmetry_rms / self.effective_floor:.1f}x)"
        return ("AMBIGUOUS -- within the nearest_vertex artefact bound. This does not "
                "mean the face is symmetric; rerun with method='point_to_surface' to resolve")

    def summary(self) -> str:
        verdict = self.verdict()
        return "\n".join([
            f"symmetry_rms   : {self.symmetry_rms:.5f}   (dimensionless; lower = more symmetric)",
            f"symmetry_mean  : {self.symmetry_mean:.5f}",
            f"symmetry_p95   : {self.symmetry_p95:.5f}",
            f"symmetry_max   : {self.symmetry_max:.5f}",
            f"discret. floor : {self.discretization_floor:.5f} raw / {self.effective_floor:.5f} effective",
            f"  verdict      : {verdict}",
            f"mirror pairing : {self.mirror_pairing_fraction:.1%} of vertices have a mirror counterpart",
            f"midplane normal: {np.round(self.plane_normal, 4).tolist()}",
            f"midplane offset: {self.plane_offset:+.5f}",
            f"midplane tilt  : {self.plane_tilt_deg:.2f} deg from the frame's left-right axis",
            f"method         : {self.method}",
            f"converged      : {self.converged}",
        ])


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def reflect(points: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    """Mirror ``points`` through the plane ``{p : normal.p = offset}``."""
    unit = normal / np.linalg.norm(normal)
    signed = points @ unit - offset
    return points - 2.0 * signed[:, None] * unit[None, :]


def _point_to_surface_distance(queries: np.ndarray, vertices: np.ndarray, triangles: np.ndarray,
                               candidates: int = 12) -> np.ndarray:
    """Distance from each query point to the nearest point on the mesh surface.

    For tractability we only test the triangles incident on the ``candidates``
    nearest vertices, which is exact in practice for a mesh this dense.
    """
    tree = cKDTree(vertices)
    k = min(candidates, len(vertices))
    _, near_vertices = tree.query(queries, k=k)
    near_vertices = np.atleast_2d(near_vertices)

    # vertex -> incident triangles
    incident: list[list[int]] = [[] for _ in range(len(vertices))]
    for t_idx, tri in enumerate(triangles):
        for v_idx in tri:
            incident[int(v_idx)].append(t_idx)

    out = np.empty(len(queries), dtype=np.float64)
    for i, query in enumerate(queries):
        tri_ids = {t for v in near_vertices[i] for t in incident[int(v)]}
        if not tri_ids:
            out[i] = float(np.linalg.norm(vertices[near_vertices[i][0]] - query))
            continue
        tris = triangles[sorted(tri_ids)]
        out[i] = _closest_on_triangles(query, vertices[tris[:, 0]], vertices[tris[:, 1]], vertices[tris[:, 2]])
    return out


def _closest_on_triangles(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Minimum distance from ``point`` to a batch of triangles (Ericson's method)."""
    ab, ac, ap = b - a, c - a, point - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)

    bp = point - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)

    cp = point - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc

    closest = np.empty_like(a)

    # Vertex regions
    m_a = (d1 <= 0) & (d2 <= 0)
    m_b = (d3 >= 0) & (d4 <= d3)
    m_c = (d6 >= 0) & (d5 <= d6)
    closest[m_a] = a[m_a]
    closest[m_b] = b[m_b]
    closest[m_c] = c[m_c]

    # Edge regions
    m_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~m_a & ~m_b
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ab = np.where((d1 - d3) != 0, d1 / (d1 - d3), 0.0)
    closest[m_ab] = a[m_ab] + t_ab[m_ab, None] * ab[m_ab]

    m_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~m_a & ~m_c
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ac = np.where((d2 - d6) != 0, d2 / (d2 - d6), 0.0)
    closest[m_ac] = a[m_ac] + t_ac[m_ac, None] * ac[m_ac]

    m_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0) & ~m_b & ~m_c
    with np.errstate(divide="ignore", invalid="ignore"):
        t_bc = np.where(((d4 - d3) + (d5 - d6)) != 0, (d4 - d3) / ((d4 - d3) + (d5 - d6)), 0.0)
    closest[m_bc] = b[m_bc] + t_bc[m_bc, None] * (c[m_bc] - b[m_bc])

    # Interior
    done = m_a | m_b | m_c | m_ab | m_ac | m_bc
    interior = ~done
    if np.any(interior):
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(denom != 0, 1.0 / denom, 0.0)
        v = vb * inv
        w = vc * inv
        closest[interior] = a[interior] + v[interior, None] * ab[interior] + w[interior, None] * ac[interior]

    return float(np.min(np.linalg.norm(closest - point, axis=1)))


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------

def _reflection_error(params: np.ndarray, points: np.ndarray, tree: cKDTree) -> float:
    normal = params[:3]
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return 1e9
    mirrored = reflect(points, normal / norm, float(params[3]))
    distances, _ = tree.query(mirrored, k=1)
    return float(np.mean(distances ** 2))


def measure_symmetry(
    vertices: np.ndarray,
    triangles: np.ndarray | None = None,
    initial_normal: Sequence[float] = (1.0, 0.0, 0.0),
    method: str = "point_to_surface",
    fit_plane: bool = True,
) -> SymmetryResult:
    """Measure bilateral symmetry. See the module docstring for the definition."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"vertices must be (N, 3), got {verts.shape}")

    centroid = verts.mean(axis=0)
    centroid_size = float(np.sqrt(((verts - centroid) ** 2).sum(axis=1).mean()))
    if centroid_size == 0:
        raise ValueError("degenerate mesh: centroid size is zero")

    tree = cKDTree(verts)
    notes: list[str] = []

    normal = np.asarray(initial_normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    offset = float(normal @ centroid)
    converged = True

    if fit_plane:
        result = minimize(
            _reflection_error,
            x0=np.concatenate([normal, [offset]]),
            args=(verts, tree),
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-7, "fatol": 1e-12},
        )
        converged = bool(result.success)
        if not converged:
            notes.append("midplane optimisation did not converge; using the best point found")
        normal = result.x[:3] / np.linalg.norm(result.x[:3])
        offset = float(result.x[3])
    else:
        notes.append("midplane fixed at the initial guess (fit_plane=False)")

    mirrored = reflect(verts, normal, offset)

    # Diagnostics: how big is the discretization artefact, and does the mesh
    # topology cancel it by being mirror-paired?
    spacing, _ = tree.query(verts, k=2)          # k=2: self, then true nearest
    median_spacing = float(np.median(spacing[:, 1]))
    discretization_floor = median_spacing / centroid_size
    mirror_distances, _ = tree.query(mirrored, k=1)
    pairing = float(np.mean(mirror_distances < 0.1 * median_spacing)) if median_spacing > 0 else 0.0

    if method == "point_to_surface" and triangles is None:
        # No topology to project onto; fall back rather than fail, and say so,
        # because the fallback has a discretization floor the caller must read.
        method = "nearest_vertex"
        notes.append(
            "point_to_surface requested but no triangles were supplied; fell back to "
            "nearest_vertex, which carries a discretization floor"
        )

    if method == "point_to_surface":
        distances = _point_to_surface_distance(mirrored, verts, np.asarray(triangles))
    else:
        distances = mirror_distances
        if pairing < 0.5:
            notes.append(
                f"only {pairing:.0%} of vertices have a mirror counterpart, so nearest-vertex "
                f"matching carries a discretization floor of ~{discretization_floor:.4f}; "
                "prefer method='point_to_surface'"
            )

    per_vertex = np.asarray(distances, dtype=np.float64) / centroid_size
    tilt = float(np.degrees(np.arccos(np.clip(abs(normal[0]), 0.0, 1.0))))

    return SymmetryResult(
        discretization_floor=discretization_floor,
        mirror_pairing_fraction=pairing,
        symmetry_rms=float(np.sqrt((per_vertex ** 2).mean())),
        symmetry_mean=float(per_vertex.mean()),
        symmetry_max=float(per_vertex.max()),
        symmetry_p95=float(np.quantile(per_vertex, 0.95)),
        per_vertex=per_vertex,
        plane_normal=normal,
        plane_offset=offset,
        centroid_size=centroid_size,
        method=method,
        plane_tilt_deg=tilt,
        converged=converged,
        notes=notes,
    )


def measure_face_symmetry(face: NormalizedFace, method: str = "point_to_surface") -> SymmetryResult:
    """Symmetry of a normalised face, seeded with the canonical left-right axis."""
    return measure_symmetry(
        face.vertices,
        face.triangles,
        initial_normal=(1.0, 0.0, 0.0),  # canonical +X is subject's left
        method=method,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure bilateral facial symmetry.")
    parser.add_argument("path", help="a capture .json")
    parser.add_argument("--method", default="point_to_surface", choices=METHODS)
    parser.add_argument("--no-fit", action="store_true", help="do not fit the midplane; assume x=0")
    args = parser.parse_args(argv)

    capture = load_capture(Path(args.path))
    face = normalize_capture(capture)
    result = measure_symmetry(
        face.vertices, face.triangles, method=args.method, fit_plane=not args.no_fit
    )
    print(face.key)
    print(result.summary())
    for note in result.notes:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
