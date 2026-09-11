"""Print the real key structure of an exported capture.

THIS IS THE FIRST THING TO RUN on a JSON file from the iOS app. It makes no
assumptions at all -- it just shows what is actually in the file and reports
which canonical pipeline fields the current ``config/schema_map.json`` manages
to resolve. Use the output to fix the schema map (not the Python) when a key
name differs.

    python -m src.inspect_schema data/raw/P0001/frontal_001.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from .config import RAW_DIR, SchemaMap, load_json
from .data_loader import resolve_field

MAX_PREVIEW = 6


def _describe(value: Any) -> str:
    """One-line description of a JSON value: type, size, and a short preview."""
    if isinstance(value, dict):
        return f"object ({len(value)} keys)"
    if isinstance(value, list):
        if not value:
            return "array (empty)"
        head = value[0]
        kind = type(head).__name__
        if isinstance(head, list):
            inner = f"[{len(head)}]" if head else "[]"
            return f"array ({len(value)} x {kind}{inner})"
        if isinstance(head, dict):
            return f"array ({len(value)} x object, keys={sorted(head.keys())[:5]})"
        preview = ", ".join(repr(v) for v in value[:3])
        return f"array ({len(value)} x {kind}) [{preview}, ...]"
    if isinstance(value, str):
        text = value if len(value) <= 60 else value[:57] + "..."
        return f"string {text!r}"
    if value is None:
        return "null"
    return f"{type(value).__name__} {value!r}"


def _walk(node: Any, prefix: str = "", depth: int = 0, max_depth: int = 4) -> list[str]:
    """Render the key tree, descending into objects but not into bulk arrays."""
    lines: list[str] = []
    if not isinstance(node, dict) or depth > max_depth:
        return lines
    for key in sorted(node.keys(), key=str):
        value = node[key]
        path = f"{prefix}.{key}" if prefix else str(key)
        lines.append(f"{'  ' * depth}{key}: {_describe(value)}")
        if isinstance(value, dict):
            lines.extend(_walk(value, path, depth + 1, max_depth))
        elif isinstance(value, list) and value and isinstance(value[0], dict) and len(value) <= MAX_PREVIEW:
            lines.extend(_walk(value[0], path + "[0]", depth + 1, max_depth))
    return lines


def inspect(path: Path, schema: SchemaMap) -> int:
    print("=" * 70)
    print(f"FILE: {path}")
    print(f"SIZE: {path.stat().st_size / 1024:.1f} KiB")
    print("=" * 70)

    try:
        blob = load_json(path)
    except ValueError as exc:
        print(f"UNREADABLE: {exc}")
        return 1

    if not isinstance(blob, dict):
        print(f"Top level is {type(blob).__name__}, expected an object.")
        return 1

    print("\n--- ACTUAL KEY TREE " + "-" * 50)
    for line in _walk(blob):
        print("  " + line)

    print("\n--- CANONICAL FIELD RESOLUTION " + "-" * 39)
    print("  (how config/schema_map.json maps onto this file)\n")
    unresolved: list[str] = []
    for canonical in schema.canonical_fields:
        value, matched = resolve_field(blob, schema.candidates(canonical))
        if matched is None:
            unresolved.append(canonical)
            status = "-- not found"
            print(f"  {canonical:22s} {status}")
        else:
            print(f"  {canonical:22s} -> {matched:28s} {_describe(value)}")

    print("\n--- WHAT TO DO NEXT " + "-" * 50)
    required_missing = [f for f in schema.required if f in unresolved]
    if required_missing:
        print("  BLOCKING -- these are required and did not resolve:")
        for name in required_missing:
            print(f"    * {name}")
        print("  Add the real key name to config/schema_map.json -> fields ->")
        print("  the matching entry, at the FRONT of the list. No code change needed.")
    else:
        print("  All required fields resolved. `python -m src.data_loader <file>`")
        print("  should now work on this capture.")

    optional_missing = [f for f in unresolved if f not in schema.required]
    if optional_missing:
        print("\n  Optional fields not found (pipeline still runs without them):")
        for name in optional_missing:
            print(f"    * {name}")
        if "left_eye_transform" in optional_missing or "right_eye_transform" in optional_missing:
            print("\n  NOTE: leftEyeTransform / rightEyeTransform were not found.")
            print("  These are the only anatomically-named points ARKit provides")
            print("  without guessing vertex indices, and normalisation uses them")
            print("  for scale and roll when present. Exporting them from the iOS")
            print("  app (ARFaceAnchor.leftEyeTransform / .rightEyeTransform) is")
            print("  the single highest-value addition you can make to the export.")
    return 0 if not required_missing else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", default=str(RAW_DIR),
                        help="capture .json, or a directory (first file is inspected)")
    parser.add_argument("--all", action="store_true", help="inspect every file in the directory")
    args = parser.parse_args(argv)

    target = Path(args.path)
    schema = SchemaMap.load()

    if target.is_file():
        paths = [target]
    else:
        found = sorted(p for p in target.rglob("*.json") if not p.name.startswith("."))
        if not found:
            print(f"No .json captures found under {target}.")
            print("Export one capture from the iOS app into data/raw/ and rerun.")
            return 1
        paths = found if args.all else found[:1]
        if not args.all and len(found) > 1:
            print(f"[{len(found)} files found; inspecting the first. Use --all for every file.]\n")

    return max(inspect(p, schema) for p in paths)


if __name__ == "__main__":
    raise SystemExit(main())
