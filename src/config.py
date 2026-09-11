"""Project paths, tunable constants, and on-disk configuration loading.

Nothing in here knows about faces. It exists so that no other module has to
hard-code a filesystem path or a magic number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_DIR = DATA_DIR / "labels"
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SCHEMA_MAP_PATH = CONFIG_DIR / "schema_map.json"
LANDMARKS_PATH = CONFIG_DIR / "landmarks.json"

FEATURES_CSV = PROCESSED_DIR / "features.csv"
RATINGS_CSV = LABELS_DIR / "ratings.csv"
TARGETS_CSV = PROCESSED_DIR / "targets.csv"
VALIDATION_REPORT = OUTPUTS_DIR / "validation_report.txt"


def ensure_dirs() -> None:
    """Create the writable output directories if they do not exist."""
    for d in (RAW_DIR, PROCESSED_DIR, LABELS_DIR, MODELS_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Reference values -- INFORMATIONAL ONLY, never enforced
# --------------------------------------------------------------------------

# Apple's ARFaceGeometry has historically had this topology on TrueDepth
# devices. We record it ONLY to annotate reports with "matches/differs from the
# common ARKit topology". Validation never fails a capture for differing: the
# authoritative expectation is derived from your own cohort (see
# validation.CohortExpectation), which is learned from the real files.
ARKIT_REFERENCE_VERTEX_COUNT = 1220
ARKIT_REFERENCE_TRIANGLE_COUNT = 2304

# Plausibility bounds for a human face mesh expressed in metres, used only to
# catch unit errors (e.g. millimetres exported without a `units` field) and
# gross corruption. Deliberately wide.
PLAUSIBLE_FACE_EXTENT_M = (0.05, 0.40)  # (min, max) bounding-box diagonal

# Unit conversion factors -> metres.
UNIT_TO_METRES = {
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
    "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
}

# Vertex-space declarations the loader understands.
FACE_LOCAL_SPACES = {"face", "face_local", "facelocal", "anchor", "local", "model"}
WORLD_SPACES = {"world", "global", "scene"}


# --------------------------------------------------------------------------
# Config loading
# --------------------------------------------------------------------------

@dataclass
class SchemaMap:
    """Canonical-field -> candidate-key mapping, loaded from schema_map.json."""

    fields: dict[str, list[str]] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None = None) -> "SchemaMap":
        path = Path(path) if path is not None else SCHEMA_MAP_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Schema map not found at {path}. It ships with the repo; "
                f"restore it or pass an explicit path."
            )
        with path.open("r", encoding="utf-8") as fh:
            blob: dict[str, Any] = json.load(fh)
        return cls(
            fields=blob.get("fields", {}),
            required=blob.get("required", []),
        )

    def candidates(self, canonical: str) -> list[str]:
        return list(self.fields.get(canonical, [canonical]))

    @property
    def canonical_fields(self) -> list[str]:
        return list(self.fields.keys())


def load_json(path: Path | str) -> Any:
    """Read a JSON file, raising a message that names the file on failure."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name}: malformed JSON at line {exc.lineno} col {exc.colno}: {exc.msg}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name}: not valid UTF-8 text ({exc.reason})") from exc
