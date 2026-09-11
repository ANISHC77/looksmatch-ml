"""Phase 2 -- look at the geometry.

The point of this module is VERIFICATION: before trusting a single measurement,
you should see the exported iPhone mesh rendered from its own triangle topology
and confirm it looks like the face you captured.

Backends
--------
Open3D is used when it is importable (interactive, fast, proper lighting).
Open3D currently publishes no wheel for Python 3.13, so a matplotlib backend is
always available as a fallback and is selected automatically. Everything here
works either way; only the interaction quality differs. Force one with
``backend="matplotlib"`` or ``backend="open3d"``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .data_loader import load_capture
from .landmarks import LandmarkSet
from .normalization import normalize_capture
from .symmetry import measure_symmetry, reflect

BACKENDS = ("auto", "open3d", "matplotlib")


def open3d_available() -> bool:
    try:
        import open3d  # noqa: F401, PLC0415
        return True
    except ImportError:
        return False


def _resolve_backend(backend: str) -> str:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "auto":
        return "open3d" if open3d_available() else "matplotlib"
    if backend == "open3d" and not open3d_available():
        raise ImportError(
            "Open3D is not installed. It has no wheel for Python 3.13 -- either use "
            "backend='matplotlib', or create the venv with Python 3.12 (see README)."
        )
    return backend


# ==========================================================================
# Open3D backend
# ==========================================================================

def _o3d_mesh(vertices: np.ndarray, triangles: np.ndarray, colors: np.ndarray | None):
    import open3d as o3d  # noqa: PLC0415

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)),
    )
    mesh.compute_vertex_normals()
    if colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    else:
        mesh.paint_uniform_color([0.78, 0.72, 0.68])
    return mesh


def _show_open3d(vertices, triangles, title, colors, highlight, wireframe, show_axes, axis_size):
    import open3d as o3d  # noqa: PLC0415

    geometries = []
    if triangles is not None and len(triangles):
        mesh = _o3d_mesh(vertices, triangles, colors)
        geometries.append(mesh)
        if wireframe:
            geometries.append(o3d.geometry.LineSet.create_from_triangle_mesh(mesh))
    else:
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))
        if colors is not None:
            cloud.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(cloud)

    if show_axes:
        geometries.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
        )
    for name, point in (highlight or {}).items():
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=axis_size * 0.06)
        sphere.translate(np.asarray(point, dtype=np.float64))
        sphere.paint_uniform_color([0.9, 0.1, 0.1])
        sphere.compute_vertex_normals()
        geometries.append(sphere)

    o3d.visualization.draw_geometries(geometries, window_name=title, width=1000, height=1100)


# ==========================================================================
# matplotlib backend
# ==========================================================================

def _show_matplotlib(vertices, triangles, title, colors, highlight, wireframe, show_axes, axis_size):
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: PLC0415

    fig = plt.figure(figsize=(9, 10))
    ax = fig.add_subplot(111, projection="3d")

    if triangles is not None and len(triangles):
        polys = vertices[np.asarray(triangles)]
        if colors is not None:
            face_colors = colors[np.asarray(triangles)].mean(axis=1)
        else:
            # Shade by depth so the surface reads as 3D without a light model.
            depth = polys[:, :, 2].mean(axis=1)
            spread = np.ptp(depth)
            normalised = (depth - depth.min()) / spread if spread > 0 else np.zeros_like(depth)
            face_colors = plt.get_cmap("copper")(normalised)[:, :3]
        collection = Poly3DCollection(
            polys, facecolors=face_colors,
            edgecolors=("black" if wireframe else "none"),
            linewidths=(0.15 if wireframe else 0.0),
        )
        ax.add_collection3d(collection)
    else:
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=2,
                   c=(colors if colors is not None else vertices[:, 2]),
                   cmap=(None if colors is not None else "viridis"))

    for name, point in (highlight or {}).items():
        point = np.asarray(point, dtype=np.float64)
        ax.scatter(*point, s=60, c="red", depthshade=False)
        ax.text(point[0], point[1], point[2], f" {name}", fontsize=7, color="darkred")

    if show_axes:
        origin = np.zeros(3)
        for direction, colour, label in (
            ((1, 0, 0), "tab:red", "+X left"),
            ((0, 1, 0), "tab:green", "+Y superior"),
            ((0, 0, 1), "tab:blue", "+Z anterior"),
        ):
            vector = np.asarray(direction) * axis_size
            ax.quiver(*origin, *vector, color=colour, linewidth=2)
            ax.text(*(vector * 1.12), label, color=colour, fontsize=8)

    _set_equal_aspect(ax, vertices)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=8, azim=-88)  # near-frontal
    fig.tight_layout()
    plt.show()


def _set_equal_aspect(ax, points: np.ndarray) -> None:
    """Equal aspect on all three axes -- without this a face looks stretched."""
    centre = (points.max(axis=0) + points.min(axis=0)) / 2
    radius = float((points.max(axis=0) - points.min(axis=0)).max()) / 2 * 1.05
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))


# ==========================================================================
# Public API
# ==========================================================================

def show_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray | None = None,
    title: str = "Looksmatch mesh",
    colors: np.ndarray | None = None,
    highlight: Mapping[str, np.ndarray] | None = None,
    wireframe: bool = False,
    show_axes: bool = False,
    backend: str = "auto",
) -> None:
    """Render a mesh (or a point cloud when ``triangles`` is None)."""
    vertices = np.asarray(vertices, dtype=np.float64)
    axis_size = float((vertices.max(axis=0) - vertices.min(axis=0)).max()) * 0.35
    chosen = _resolve_backend(backend)
    renderer = _show_open3d if chosen == "open3d" else _show_matplotlib
    renderer(vertices, triangles, title, colors, highlight, wireframe, show_axes, axis_size)


def show_point_cloud(vertices: np.ndarray, title: str = "Looksmatch point cloud",
                     backend: str = "auto") -> None:
    """Render only the vertices, ignoring topology."""
    show_mesh(vertices, None, title, backend=backend)


def symmetry_colors(per_vertex: np.ndarray) -> np.ndarray:
    """Blue (symmetric) -> red (asymmetric), scaled to the 95th percentile."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    ceiling = float(np.quantile(per_vertex, 0.95)) or 1.0
    return plt.get_cmap("coolwarm")(np.clip(per_vertex / ceiling, 0, 1))[:, :3]


def show_symmetry(capture_path: Path | str, method: str = "point_to_surface",
                  backend: str = "auto") -> None:
    """Render the per-vertex asymmetry heat map and its mirrored overlay."""
    capture = load_capture(Path(capture_path))
    face = normalize_capture(capture)
    result = measure_symmetry(face.vertices, face.triangles, method=method)
    print(result.summary())
    for note in result.notes:
        print(f"  note: {note}")
    show_mesh(
        face.vertices, face.triangles,
        title=(f"{face.key} -- asymmetry (rms={result.symmetry_rms:.4f}, "
               f"floor={result.effective_floor:.4f}); red = least symmetric"),
        colors=symmetry_colors(result.per_vertex),
        backend=backend,
    )


def compare_original_and_mirror(capture_path: Path | str, backend: str = "auto") -> None:
    """Overlay the face and its own reflection through the fitted midplane."""
    capture = load_capture(Path(capture_path))
    face = normalize_capture(capture)
    result = measure_symmetry(face.vertices, face.triangles)
    mirrored = reflect(face.vertices, result.plane_normal, result.plane_offset)

    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig = plt.figure(figsize=(9, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*face.vertices.T, s=2, c="tab:blue", label="original", alpha=0.6)
    ax.scatter(*mirrored.T, s=2, c="tab:orange", label="mirrored", alpha=0.6)
    _set_equal_aspect(ax, face.vertices)
    ax.legend(loc="upper right")
    ax.set_title(f"{face.key} -- original vs reflection through the fitted midplane", fontsize=10)
    ax.view_init(elev=8, azim=-88)
    plt.show()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualise an exported ARKit capture.")
    parser.add_argument("path", help="a capture .json")
    parser.add_argument("--mode", default="mesh",
                        choices=("mesh", "cloud", "wireframe", "symmetry", "mirror"))
    parser.add_argument("--backend", default="auto", choices=BACKENDS)
    parser.add_argument("--raw", action="store_true",
                        help="show the raw exported vertices instead of the normalised frame")
    parser.add_argument("--frame", action="store_true",
                        help="draw the canonical axes, to confirm the detected convention")
    parser.add_argument("--landmarks", action="store_true", help="highlight configured landmarks")
    parser.add_argument("--symmetry-method", default="point_to_surface",
                        choices=("nearest_vertex", "point_to_surface"))
    args = parser.parse_args(argv)

    path = Path(args.path)
    if args.mode == "symmetry":
        show_symmetry(path, args.symmetry_method, args.backend)
        return 0
    if args.mode == "mirror":
        compare_original_and_mirror(path, args.backend)
        return 0

    capture = load_capture(path)
    print(capture.summary())

    if args.raw:
        vertices, label = capture.vertices, "raw exported vertices"
    else:
        face = normalize_capture(capture)
        vertices, label = face.vertices, f"normalised ({face.origin_mode}/{face.scale_mode})"

    highlight = None
    if args.landmarks:
        landmarks = LandmarkSet.load()
        if landmarks and landmarks.applies_to(capture.n_vertices):
            highlight = {n: vertices[i] for n, i in landmarks.indices.items()}
        else:
            print("(no applicable landmarks configured -- nothing to highlight)")

    backend = _resolve_backend(args.backend)
    print(f"\nRendering with the {backend} backend: {label}")
    if backend == "matplotlib":
        print("(Open3D unavailable; see README for the Python 3.12 route to the interactive viewer)")

    show_mesh(
        vertices,
        None if args.mode == "cloud" else capture.triangles,
        title=f"{capture.key} -- {label}",
        highlight=highlight,
        wireframe=(args.mode == "wireframe"),
        show_axes=args.frame,
        backend=backend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
