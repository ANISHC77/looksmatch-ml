"""Phase 1 -- load ARFaceGeometry JSON exported by the iOS app into NumPy.

DESIGN NOTE / READ THIS FIRST
-----------------------------
At the time of writing, the exact key names used by the iOS exporter were not
available to inspect. Rather than guess once and hard-code, this loader:

  * resolves every field through ``config/schema_map.json`` (edit that file, not
    this one, when a key name differs);
  * accepts the several plausible serialisations of each array shape;
  * records EVERY inference it makes in ``FaceCapture.assumptions`` so nothing
    is silently guessed.

Run ``python -m src.inspect_schema <file.json>`` on a real capture to see the
actual key tree and which canonical fields resolved. See docs/ASSUMPTIONS.md.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .config import (
    FACE_LOCAL_SPACES,
    RAW_DIR,
    UNIT_TO_METRES,
    WORLD_SPACES,
    SchemaMap,
    load_json,
)


class CaptureLoadError(Exception):
    """Raised when a capture cannot be turned into a usable mesh at all."""


# --------------------------------------------------------------------------
# Key resolution
# --------------------------------------------------------------------------

def _norm_key(key: str) -> str:
    """Canonical form of a key for matching: lowercase, no '_', ' ' or '-'."""
    return key.lower().replace("_", "").replace(" ", "").replace("-", "")


def _get_child(node: Any, segment: str) -> tuple[bool, Any]:
    """Fetch ``segment`` from a dict, matching case/underscore-insensitively."""
    if not isinstance(node, dict):
        return False, None
    if segment in node:  # exact hit first: cheapest and unambiguous
        return True, node[segment]
    target = _norm_key(segment)
    for key, value in node.items():
        if isinstance(key, str) and _norm_key(key) == target:
            return True, value
    return False, None


def _resolve_path(blob: Any, dotted: str) -> tuple[bool, Any, str]:
    """Resolve a dot-path like 'geometry.vertices'. Returns (found, value, path)."""
    node = blob
    seen: list[str] = []
    for segment in dotted.split("."):
        found, node = _get_child(node, segment)
        if not found:
            return False, None, ""
        seen.append(segment)
    return True, node, ".".join(seen)


def _recursive_find(blob: Any, leaf: str, max_depth: int = 4) -> tuple[bool, Any, str]:
    """Breadth-first search for any key matching ``leaf`` anywhere in the tree.

    Last-resort fallback for exporters that nest fields somewhere unexpected.
    Depth-bounded so a pathological file cannot make this expensive.
    """
    target = _norm_key(leaf)
    queue: list[tuple[Any, str, int]] = [(blob, "", 0)]
    while queue:
        node, prefix, depth = queue.pop(0)
        if depth > max_depth or not isinstance(node, dict):
            continue
        for key, value in node.items():
            if not isinstance(key, str):
                continue
            path = prefix + "." + key if prefix else key
            if _norm_key(key) == target:
                return True, value, path
            if isinstance(value, dict):
                queue.append((value, path, depth + 1))
    return False, None, ""


def resolve_field(blob: dict[str, Any], candidates: Sequence[str]) -> tuple[Any, str | None]:
    """Return ``(value, matched_path)`` for the first candidate that resolves."""
    for candidate in candidates:
        found, value, path = _resolve_path(blob, candidate)
        if found and value is not None:
            return value, path
    for candidate in candidates:  # fallback: search the tree for the leaf name
        leaf = candidate.split(".")[-1]
        found, value, path = _recursive_find(blob, leaf)
        if found and value is not None:
            return value, path
    return None, None


# --------------------------------------------------------------------------
# Array coercion
# --------------------------------------------------------------------------

def as_vertex_array(raw: Any, assumptions: list[str]) -> np.ndarray:
    """Coerce any plausible vertex serialisation into float64 ``(N, 3)``.

    Accepts ``[[x,y,z], ...]``, flat ``[x,y,z,x,y,z, ...]``,
    ``[{"x":..,"y":..,"z":..}, ...]``, ``{"x":[..],"y":[..],"z":[..]}``, and
    ``(N, 4)`` (SIMD float3 padded to 16 bytes -- 4th component dropped).
    """
    if raw is None:
        raise CaptureLoadError("no vertex data found")

    if isinstance(raw, dict):  # {"x": [...], "y": [...], "z": [...]}
        cols = []
        for axis in ("x", "y", "z"):
            found, value = _get_child(raw, axis)
            if not found:
                raise CaptureLoadError("vertex dict is missing component " + axis)
            cols.append(np.asarray(value, dtype=np.float64).ravel())
        if len({c.size for c in cols}) != 1:
            raise CaptureLoadError("vertex components x/y/z have different lengths")
        assumptions.append("vertices_from_component_arrays")
        return np.column_stack(cols)

    if not isinstance(raw, (list, tuple)):
        raise CaptureLoadError("vertices have unsupported type " + type(raw).__name__)
    if len(raw) == 0:
        raise CaptureLoadError("vertex array is empty")

    if isinstance(raw[0], dict):  # [{"x":..,"y":..,"z":..}, ...]
        out = np.empty((len(raw), 3), dtype=np.float64)
        for i, item in enumerate(raw):
            for j, axis in enumerate(("x", "y", "z")):
                found, value = _get_child(item, axis)
                if not found:
                    raise CaptureLoadError(f"vertex {i} is missing component {axis}")
                out[i, j] = float(value)
        assumptions.append("vertices_from_dict_records")
        return out

    arr = np.asarray(raw, dtype=np.float64)

    if arr.ndim == 1:
        if arr.size % 3 != 0:
            raise CaptureLoadError(
                f"flat vertex array length {arr.size} is not divisible by 3"
            )
        assumptions.append("vertices_reshaped_from_flat_xyz")
        return arr.reshape(-1, 3)

    if arr.ndim == 2:
        if arr.shape[1] == 3:
            return arr
        if arr.shape[1] == 4:
            # simd_float3 is 16-byte aligned, so a raw memory dump yields a
            # padding 4th component. Drop it, but say so.
            assumptions.append("vertices_had_4_components_dropped_4th_simd_padding")
            return arr[:, :3]
        raise CaptureLoadError(f"vertices have shape {arr.shape}; expected (N, 3)")

    raise CaptureLoadError(f"vertices have {arr.ndim} dimensions; expected 1 or 2")


def as_index_array(raw: Any, n_vertices: int, assumptions: list[str]) -> np.ndarray:
    """Coerce triangle indices into int32 ``(M, 3)``.

    ARKit's ``ARFaceGeometry.triangleIndices`` is a FLAT Int16 array, so the
    flat case is the expected one.
    """
    if raw is None:
        raise CaptureLoadError("no triangle index data found")

    if isinstance(raw, (list, tuple)) and len(raw) and isinstance(raw[0], dict):
        out = np.empty((len(raw), 3), dtype=np.int64)
        for i, item in enumerate(raw):
            vals = [v for _, v in sorted(item.items())]
            if len(vals) != 3:
                raise CaptureLoadError(f"triangle {i} has {len(vals)} indices, expected 3")
            out[i] = vals
        assumptions.append("triangles_from_dict_records_sorted_by_key")
        arr: np.ndarray = out
    else:
        arr = np.asarray(raw)
        if not np.issubdtype(arr.dtype, np.number):
            raise CaptureLoadError("triangle indices are not numeric")
        if arr.ndim == 1:
            if arr.size % 3 != 0:
                raise CaptureLoadError(
                    f"flat index array length {arr.size} is not divisible by 3"
                )
            arr = arr.reshape(-1, 3)
            assumptions.append("triangles_reshaped_from_flat_ijk")
        elif arr.ndim != 2 or arr.shape[1] != 3:
            raise CaptureLoadError(
                f"triangle indices have shape {arr.shape}; expected (M, 3)"
            )

    if not np.all(np.equal(np.mod(arr, 1), 0)):
        raise CaptureLoadError("triangle indices contain non-integer values")
    arr = arr.astype(np.int32)

    if arr.min() < 0:
        raise CaptureLoadError(f"triangle indices contain negative value {arr.min()}")
    if arr.max() >= n_vertices:
        raise CaptureLoadError(
            f"triangle index {arr.max()} is out of range for {n_vertices} vertices"
        )
    return arr


def as_matrix4(raw: Any, assumptions: list[str], label: str) -> np.ndarray | None:
    """Coerce a 4x4 transform to row-major ``(4, 4)`` with translation in col 3.

    Swift's ``simd_float4x4`` is COLUMN-major (``columns.0 ... columns.3``). An
    exporter that flattens the columns in order produces, after a naive
    row-major reshape, the TRANSPOSE of the conventional matrix. We detect the
    convention by checking where the ``[0,0,0,1]`` homogeneous row/column landed
    and transpose if needed. Ambiguous matrices are left as-is and flagged.
    """
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size != 16:
            assumptions.append(f"{label}_ignored_unexpected_length_{arr.size}")
            return None
        arr = arr.reshape(4, 4)
    if arr.shape != (4, 4):
        assumptions.append(f"{label}_ignored_unexpected_shape")
        return None

    bottom_row_is_affine = bool(np.allclose(arr[3, :], [0, 0, 0, 1], atol=1e-5))
    right_col_is_affine = bool(np.allclose(arr[:, 3], [0, 0, 0, 1], atol=1e-5))

    if right_col_is_affine and not bottom_row_is_affine:
        # Column-major flat dump reshaped row-wise: transpose back.
        assumptions.append(f"{label}_transposed_from_column_major_simd_layout")
        return arr.T
    if not bottom_row_is_affine and not right_col_is_affine:
        assumptions.append(f"{label}_not_affine_layout_left_as_given")
    return arr


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def as_blend_shapes(raw: Any, assumptions: list[str]) -> dict[str, float]:
    """Coerce blend-shape coefficients into ``{location_name: coefficient}``."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items() if _is_finite_number(v)}
    if isinstance(raw, (list, tuple)):
        out: dict[str, float] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            name: str | None = None
            value: Any = None
            for key, val in item.items():
                token = _norm_key(str(key))
                if token in {"name", "location", "blendshape", "key", "id"}:
                    name = str(val)
                elif token in {"value", "coefficient", "weight", "amount"}:
                    value = val
            if name is not None and _is_finite_number(value):
                out[name] = float(value)
        if out:
            assumptions.append("blend_shapes_from_name_value_records")
        return out
    return {}


# --------------------------------------------------------------------------
# The capture record
# --------------------------------------------------------------------------

@dataclass
class FaceCapture:
    """One ARFaceGeometry capture, in NumPy form."""

    participant_id: str
    capture_id: str
    vertices: np.ndarray                      # (N, 3) float64, metres
    triangles: np.ndarray                     # (M, 3) int32
    timestamp: str | None = None
    timestamp_epoch: float | None = None
    blend_shapes: dict[str, float] = field(default_factory=dict)
    transform: np.ndarray | None = None       # (4, 4) anchor pose
    left_eye_transform: np.ndarray | None = None
    right_eye_transform: np.ndarray | None = None
    look_at_point: np.ndarray | None = None
    texture_coordinates: np.ndarray | None = None
    camera: dict[str, Any] = field(default_factory=dict)
    coordinate_space: str = "face_local"
    units: str = "m"
    source_path: Path | None = None
    declared_vertex_count: int | None = None
    declared_triangle_count: int | None = None
    resolved_paths: dict[str, str] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    # -- derived -----------------------------------------------------------

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_triangles(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def key(self) -> str:
        return f"{self.participant_id}/{self.capture_id}"

    def geometry_hash(self) -> str:
        """Stable hash of the vertex data, for exact-duplicate detection."""
        return hashlib.sha256(
            np.ascontiguousarray(self.vertices, dtype=np.float64).tobytes()
        ).hexdigest()[:16]

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def bbox_diagonal(self) -> float:
        lo, hi = self.bbox()
        return float(np.linalg.norm(hi - lo))

    def centroid(self) -> np.ndarray:
        return self.vertices.mean(axis=0)

    def has_eye_transforms(self) -> bool:
        return self.left_eye_transform is not None and self.right_eye_transform is not None

    def eye_positions(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Left/right eye origins, in the same frame as ``vertices``.

        ARKit's ``ARFaceAnchor.leftEyeTransform`` / ``rightEyeTransform`` are
        expressed relative to the face anchor -- the same frame as
        ``ARFaceGeometry.vertices``. These are the only anatomically-named
        points ARKit gives us WITHOUT guessing vertex indices, so they are the
        preferred basis for scale and rotation normalisation.
        """
        if self.left_eye_transform is None or self.right_eye_transform is None:
            return None
        return (
            self.left_eye_transform[:3, 3].copy(),
            self.right_eye_transform[:3, 3].copy(),
        )

    def summary(self) -> str:
        lines = [
            f"Participant: {self.participant_id}",
            f"Capture:     {self.capture_id}",
            f"Vertices:    {self.n_vertices}",
            f"Triangles:   {self.n_triangles}",
        ]
        if self.timestamp:
            lines.append(f"Timestamp:   {self.timestamp}")
        lines.append(f"Space/units: {self.coordinate_space} / {self.units}")
        lo, hi = self.bbox()
        lines.append(
            "Bounding box (m): "
            f"x[{lo[0]:+.4f},{hi[0]:+.4f}] "
            f"y[{lo[1]:+.4f},{hi[1]:+.4f}] "
            f"z[{lo[2]:+.4f},{hi[2]:+.4f}]"
        )
        lines.append(f"Bbox diagonal:   {self.bbox_diagonal():.4f} m")
        lines.append(f"Blend shapes:    {len(self.blend_shapes)}")
        eyes = "yes" if self.has_eye_transforms() else "NO (see docs/ASSUMPTIONS.md)"
        lines.append(f"Eye transforms:  {eyes}")
        if self.missing_fields:
            lines.append("Unresolved:      " + ", ".join(self.missing_fields))
        if self.assumptions:
            lines.append("Assumptions:     " + ", ".join(self.assumptions))
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _parse_timestamp(raw: Any) -> tuple[str | None, float | None]:
    """Return ``(display string, epoch seconds)`` from a numeric or ISO value."""
    if raw is None:
        return None, None
    if _is_finite_number(raw):
        value = float(raw)
        # ARKit frame timestamps are a monotonic clock since boot, NOT Unix
        # time. Only read as an epoch if it lands in a sane calendar range.
        if 1e9 < value < 4e9:
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat(), value
        return str(value), value
    text = str(raw)
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            dt = datetime.strptime(text.replace("Z", "+0000"), fmt)
            return text, dt.timestamp()
        except ValueError:
            continue
    return text, None


def load_capture(
    path: Path | str,
    schema: SchemaMap | None = None,
    participant_id: str | None = None,
) -> FaceCapture:
    """Load one JSON capture into a :class:`FaceCapture`.

    ``participant_id`` overrides whatever is in the file; when the file carries
    no participant field, the parent directory name is used (the
    ``data/raw/P0001/frontal_001.json`` convention).
    """
    path = Path(path)
    schema = schema or SchemaMap.load()
    blob = load_json(path)
    if not isinstance(blob, dict):
        raise CaptureLoadError(
            f"{path.name}: top level of JSON is {type(blob).__name__}, expected an object"
        )

    assumptions: list[str] = []
    resolved: dict[str, str] = {}
    missing: list[str] = []

    def grab(name: str) -> Any:
        value, matched = resolve_field(blob, schema.candidates(name))
        if matched is not None:
            resolved[name] = matched
        else:
            missing.append(name)
        return value

    # -- geometry (mandatory) ---------------------------------------------
    vertices = as_vertex_array(grab("vertices"), assumptions)
    triangles = as_index_array(grab("triangle_indices"), vertices.shape[0], assumptions)

    # -- units -------------------------------------------------------------
    units_raw = grab("units")
    units = str(units_raw).strip().lower() if units_raw else "m"
    if units not in UNIT_TO_METRES:
        if units_raw is not None:
            assumptions.append(f"unrecognised_units_{units}_treated_as_metres")
        units = "m"
    scale_to_m = UNIT_TO_METRES[units]
    if scale_to_m != 1.0:
        vertices = vertices * scale_to_m
        assumptions.append(f"vertices_converted_{units}_to_metres")

    # -- identity ----------------------------------------------------------
    pid_raw = grab("participant_id")
    if participant_id is not None:
        pid = str(participant_id)
    elif pid_raw is not None:
        pid = str(pid_raw)
    else:
        pid = path.parent.name
        assumptions.append("participant_id_inferred_from_parent_directory")

    cid_raw = grab("capture_id")
    capture_id = str(cid_raw) if cid_raw is not None else path.stem
    if cid_raw is None:
        assumptions.append("capture_id_inferred_from_filename")

    timestamp, epoch = _parse_timestamp(grab("timestamp"))

    # -- pose --------------------------------------------------------------
    transform = as_matrix4(grab("transform"), assumptions, "transform")
    left_eye = as_matrix4(grab("left_eye_transform"), assumptions, "left_eye_transform")
    right_eye = as_matrix4(grab("right_eye_transform"), assumptions, "right_eye_transform")

    look_at_raw = grab("look_at_point")
    look_at = None
    if look_at_raw is not None:
        vec = np.asarray(look_at_raw, dtype=np.float64).ravel()
        look_at = vec[:3] * scale_to_m if vec.size >= 3 else None

    # -- vertex space ------------------------------------------------------
    space_raw = grab("coordinate_space")
    if space_raw is None:
        coordinate_space = "face_local"
        assumptions.append("coordinate_space_defaulted_to_face_local")
    else:
        token = _norm_key(str(space_raw))
        if token in FACE_LOCAL_SPACES:
            coordinate_space = "face_local"
        elif token in WORLD_SPACES:
            coordinate_space = "world"
        else:
            coordinate_space = "face_local"
            assumptions.append("unrecognised_coordinate_space_treated_as_face_local")

    # -- optional payloads -------------------------------------------------
    blend_shapes = as_blend_shapes(grab("blend_shapes"), assumptions)

    tex_raw = grab("texture_coordinates")
    texture_coordinates = None
    if tex_raw is not None:
        tex = np.asarray(tex_raw, dtype=np.float64)
        if tex.ndim == 1 and tex.size % 2 == 0:
            tex = tex.reshape(-1, 2)
        texture_coordinates = tex if tex.ndim == 2 and tex.shape[1] == 2 else None

    camera: dict[str, Any] = {}
    for name in ("camera", "camera_transform", "camera_intrinsics", "device_model", "tracking_state"):
        value, matched = resolve_field(blob, schema.candidates(name))
        if value is not None:
            camera[name] = value
            resolved[name] = matched or name

    declared_v = grab("vertex_count")
    declared_t = grab("triangle_count")

    return FaceCapture(
        participant_id=pid,
        capture_id=capture_id,
        vertices=vertices,
        triangles=triangles,
        timestamp=timestamp,
        timestamp_epoch=epoch,
        blend_shapes=blend_shapes,
        transform=transform,
        left_eye_transform=left_eye,
        right_eye_transform=right_eye,
        look_at_point=look_at,
        texture_coordinates=texture_coordinates,
        camera=camera,
        coordinate_space=coordinate_space,
        units=units,
        source_path=path,
        declared_vertex_count=int(declared_v) if _is_finite_number(declared_v) else None,
        declared_triangle_count=int(declared_t) if _is_finite_number(declared_t) else None,
        resolved_paths=resolved,
        missing_fields=[m for m in missing if m not in resolved],
        assumptions=assumptions,
    )


def find_capture_files(root: Path | str = RAW_DIR) -> list[Path]:
    """All ``*.json`` under ``root``, sorted, excluding dotfiles."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.json") if not p.name.startswith("."))


def iter_captures(
    root: Path | str = RAW_DIR, schema: SchemaMap | None = None
) -> Iterator[tuple[Path, FaceCapture | None, str | None]]:
    """Yield ``(path, capture, error)`` for every JSON under ``root``.

    Never raises for a bad file -- the error text is yielded instead, so one
    corrupt capture cannot abort a cohort-wide run.
    """
    schema = schema or SchemaMap.load()
    for path in find_capture_files(root):
        try:
            yield path, load_capture(path, schema=schema), None
        except (CaptureLoadError, ValueError) as exc:
            yield path, None, str(exc)


def load_all(root: Path | str = RAW_DIR, schema: SchemaMap | None = None) -> list[FaceCapture]:
    """Every capture that loads. Failures are skipped silently -- use
    :mod:`src.validation` when you need to know about them."""
    return [cap for _, cap, err in iter_captures(root, schema) if cap is not None and err is None]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Load ARFaceGeometry JSON and print a summary."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(RAW_DIR),
        help="a capture .json, or a directory to scan (default: data/raw)",
    )
    args = parser.parse_args(argv)

    target = Path(args.path)
    targets: list[tuple[Path, FaceCapture | None, str | None]]
    if target.is_file():
        try:
            targets = [(target, load_capture(target), None)]
        except (CaptureLoadError, ValueError) as exc:
            targets = [(target, None, str(exc))]
    else:
        targets = list(iter_captures(target))

    if not targets:
        print(f"No .json captures found under {target}.")
        print("Export captures from the iOS app into data/raw/<participant_id>/ and rerun.")
        return 1

    ok = 0
    for path, capture, error in targets:
        print("=" * 62)
        print(path)
        if error is not None or capture is None:
            print(f"FAILED TO LOAD: {error}")
            continue
        ok += 1
        print(capture.summary())
    print("=" * 62)
    print(f"{ok}/{len(targets)} capture(s) loaded.")
    return 0 if ok == len(targets) else 1


if __name__ == "__main__":
    raise SystemExit(main())
