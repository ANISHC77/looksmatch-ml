"""Anatomical landmark indices -- loading, picking, and verification.

Apple publishes no landmark index for ARFaceGeometry, so this module never
guesses one. ``config/landmarks.json`` ships empty and every landmark-based
feature is emitted as NaN until YOU identify the indices on YOUR mesh:

    python -m src.landmarks pick   data/raw/P0001/frontal_001.json
    python -m src.landmarks verify data/raw/P0001/frontal_001.json
    python -m src.landmarks check  data/raw

The landmark-free half of the pipeline needs none of this.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import LANDMARKS_PATH
from .data_loader import load_capture, load_all

# Landmarks that come in subject-left / subject-right pairs. These are only
# usable once the left/right sign is resolved (i.e. the export carries eye
# transforms) -- otherwise "left" and "right" could be swapped.
LATERAL_PAIRS = [
    ("left_eye_inner", "right_eye_inner"),
    ("left_eye_outer", "right_eye_outer"),
    ("nose_left_ala", "nose_right_ala"),
    ("mouth_left_corner", "mouth_right_corner"),
    ("left_gonion", "right_gonion"),
    ("left_zygion", "right_zygion"),
]


@dataclass
class LandmarkSet:
    """Named vertex indices, valid for exactly one mesh topology."""

    indices: dict[str, int] = field(default_factory=dict)
    definitions: dict[str, str] = field(default_factory=dict)
    vertex_count: int | None = None
    source: Path | None = None

    @classmethod
    def load(cls, path: Path | None = None) -> "LandmarkSet":
        path = path or LANDMARKS_PATH
        if not path.exists():
            return cls()
        blob = json.loads(path.read_text(encoding="utf-8"))
        raw = blob.get("landmarks", {})
        return cls(
            indices={k: int(v) for k, v in raw.items() if v is not None},
            definitions=blob.get("definitions", {}),
            vertex_count=(blob.get("topology") or {}).get("vertex_count"),
            source=path,
        )

    def save(self, path: Path | None = None) -> Path:
        path = path or LANDMARKS_PATH
        blob = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"landmarks": {}}
        landmarks = blob.setdefault("landmarks", {})
        for name in set(landmarks) | set(self.indices):
            landmarks[name] = self.indices.get(name)
        blob.setdefault("topology", {})["vertex_count"] = self.vertex_count
        path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        return path

    # -- access ------------------------------------------------------------

    def __bool__(self) -> bool:
        return bool(self.indices)

    def has(self, *names: str) -> bool:
        return all(n in self.indices for n in names)

    def applies_to(self, n_vertices: int) -> bool:
        """Indices are only meaningful for the topology they were picked on."""
        return self.vertex_count is None or self.vertex_count == n_vertices

    def point(self, vertices: np.ndarray, name: str) -> np.ndarray | None:
        idx = self.indices.get(name)
        if idx is None or idx >= len(vertices):
            return None
        return vertices[idx]

    def distance(self, vertices: np.ndarray, a: str, b: str) -> float:
        """Euclidean distance between two landmarks, NaN if either is undefined."""
        pa, pb = self.point(vertices, a), self.point(vertices, b)
        if pa is None or pb is None:
            return float("nan")
        return float(np.linalg.norm(pa - pb))

    @property
    def defined(self) -> list[str]:
        return sorted(self.indices)

    def report(self, all_names: Sequence[str] | None = None) -> str:
        names = list(all_names or self.definitions.keys() or self.indices.keys())
        lines = [f"{len(self.indices)}/{len(names)} landmarks defined"]
        if self.vertex_count:
            lines.append(f"picked on a mesh of {self.vertex_count} vertices")
        for name in sorted(names):
            idx = self.indices.get(name)
            lines.append(f"  {name:22s} {'index ' + str(idx) if idx is not None else '-- undefined'}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Interactive picking
# --------------------------------------------------------------------------

def pick_indices(vertices: np.ndarray, triangles: np.ndarray) -> list[int]:
    """Open a viewer and return the vertex indices the user clicked.

    Uses Open3D's editing visualiser when available (shift+click to pick);
    otherwise falls back to a matplotlib scatter where clicking picks the
    nearest projected vertex.
    """
    try:
        import open3d as o3d  # noqa: PLC0415
    except ImportError:
        return _pick_matplotlib(vertices, triangles)

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles),
    )
    mesh.compute_vertex_normals()
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))
    print("Open3D picker: SHIFT+LEFT-CLICK a vertex to pick it, press Q when done.")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Pick landmarks (shift+click)")
    vis.add_geometry(cloud)
    vis.run()
    vis.destroy_window()
    return list(vis.get_picked_points())


def _pick_matplotlib(vertices: np.ndarray, triangles: np.ndarray) -> list[int]:
    """Fallback picker: click on a 3D scatter, nearest projected vertex wins."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    picked: list[int] = []
    fig = plt.figure(figsize=(8, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=2, c=vertices[:, 2], cmap="viridis")
    ax.set_title("Click a vertex to pick it (rotate first, then click). Close the window when done.")
    _equal_aspect(ax, vertices)

    def on_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        # Project every vertex with the current view matrix and take the nearest.
        proj = ax.get_proj()
        from matplotlib.transforms import Bbox  # noqa: PLC0415
        import mpl_toolkits.mplot3d.proj3d as proj3d  # noqa: PLC0415

        xs, ys, _ = proj3d.proj_transform(vertices[:, 0], vertices[:, 1], vertices[:, 2], proj)
        screen = ax.transData.transform(np.column_stack([xs, ys]))
        distances = np.hypot(screen[:, 0] - event.x, screen[:, 1] - event.y)
        idx = int(np.argmin(distances))
        picked.append(idx)
        print(f"  picked vertex {idx} at {np.round(vertices[idx], 5).tolist()}")

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    return picked


def _equal_aspect(ax, points: np.ndarray) -> None:
    """Equal aspect ratio -- essential, or the face looks distorted."""
    centre = (points.max(axis=0) + points.min(axis=0)) / 2
    radius = float((points.max(axis=0) - points.min(axis=0)).max()) / 2
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage ARFaceGeometry landmark indices.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pick = sub.add_parser("pick", help="click vertices on a real mesh to find their indices")
    p_pick.add_argument("path")
    p_pick.add_argument("--assign", nargs="*", default=None,
                        help="landmark names to assign to the picks, in click order")

    p_verify = sub.add_parser("verify", help="render the mesh with configured landmarks highlighted")
    p_verify.add_argument("path")

    p_check = sub.add_parser("check", help="check the landmark set against a cohort")
    p_check.add_argument("root", nargs="?", default="data/raw")

    p_show = sub.add_parser("show", help="print the current landmark configuration")

    args = parser.parse_args(argv)
    landmarks = LandmarkSet.load()

    if args.command == "show":
        print(landmarks.report())
        if not landmarks:
            print("\nNo landmarks defined. The landmark-free pipeline still runs;")
            print("landmark-based features will be NaN. Use `pick` to define them.")
        return 0

    if args.command == "pick":
        capture = load_capture(Path(args.path))
        print(f"{capture.key}: {capture.n_vertices} vertices")
        picks = pick_indices(capture.vertices, capture.triangles)
        if not picks:
            print("No vertices picked.")
            return 1
        print(f"\nPicked indices: {picks}")
        if args.assign:
            if len(args.assign) != len(picks):
                print(f"ERROR: {len(args.assign)} names given but {len(picks)} vertices picked.")
                return 1
            for name, idx in zip(args.assign, picks):
                landmarks.indices[name] = int(idx)
                print(f"  {name} = {idx}")
            landmarks.vertex_count = capture.n_vertices
            print(f"\nSaved to {landmarks.save()}")
            print("Now run `verify` to confirm each landmark sits where you expect.")
        else:
            print("\nRerun with --assign <name> [<name> ...] to store these, e.g.:")
            print(f"  python -m src.landmarks pick {args.path} --assign nose_tip chin_tip")
        return 0

    if args.command == "verify":
        from .visualization import show_mesh  # noqa: PLC0415

        capture = load_capture(Path(args.path))
        if not landmarks:
            print("No landmarks defined yet; nothing to verify.")
            return 1
        if not landmarks.applies_to(capture.n_vertices):
            print(f"ERROR: landmarks were picked on a {landmarks.vertex_count}-vertex mesh, "
                  f"but this capture has {capture.n_vertices}. They do not transfer.")
            return 1
        print(landmarks.report())
        show_mesh(
            capture.vertices, capture.triangles,
            title=f"{capture.key} -- landmark verification",
            highlight={n: capture.vertices[i] for n, i in landmarks.indices.items()},
        )
        return 0

    # check
    captures = load_all(Path(args.root))
    if not captures:
        print(f"No loadable captures under {args.root}.")
        return 1
    if not landmarks:
        print("No landmarks defined; nothing to check.")
        return 0
    counts = {c.n_vertices for c in captures}
    print(f"Cohort vertex counts: {sorted(counts)}")
    bad = [c.key for c in captures if not landmarks.applies_to(c.n_vertices)]
    if bad:
        print(f"\n{len(bad)} capture(s) have a topology these indices do NOT apply to:")
        for key in bad[:10]:
            print(f"  {key}")
        return 1
    print(f"\nLandmark set applies to all {len(captures)} capture(s).")
    unresolved = [a for a, b in LATERAL_PAIRS if landmarks.has(a) != landmarks.has(b)]
    if unresolved:
        print(f"WARNING: these lateral landmarks are defined on only one side: {unresolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
