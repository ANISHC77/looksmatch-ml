# Normalisation: every transformation, documented

Goal: make two faces comparable regardless of distance from the camera,
translation, rotation, and overall size.

Implemented in `src/normalization.py`.

---

## The key structural fact

**ARKit has already solved head pose for us.**

`ARFaceGeometry.vertices` are expressed relative to the `ARFaceAnchor`, not to
the camera. The anchor *is* the head's own coordinate frame, tracked by ARKit
every frame. So a face-local capture is **already** free of camera distance,
translation and rotation, before this pipeline touches it.

This is not a small convenience. In a photo- or video-based pipeline, pose
normalisation is an estimation problem with real error. Here it is exact, and it
comes free from the sensor.

What remains is therefore *not* "estimate the head pose". It is:

1. know which ARKit axis is which (a fixed property of the format, **detected
   once**, then reused);
2. pick an origin;
3. pick a scale;
4. optionally clean up residual tracker jitter.

**Caveat:** all of the above holds only if the exporter wrote *face-local*
vertices. If it wrote world-space vertices, the anchor transform is inverted
first — see assumption A4 in `ASSUMPTIONS.md`, which you must confirm.

---

## The canonical Looksmatch frame

Right-handed:

| Axis | Direction |
|------|-----------|
| `+X` | the **subject's** left |
| `+Y` | superior (up) |
| `+Z` | anterior (out of the face) |

"Left" always means the subject's own left, never the viewer's.

---

## Step 1 — to face-local space

```
if coordinate_space == "world":   vertices := inverse(anchor_transform) · vertices
else:                             vertices unchanged
```

`to_face_local()`. Raises if world-space vertices arrive without a transform,
since head pose then cannot be undone.

## Step 2 — to the canonical frame

A **signed axis permutation** `R`, so `canonical = raw @ R.T`. `R` is built from
the detected `AxisConvention` and is always kept right-handed — a reflection
would swap the subject's left and right.

### How the convention is detected

Run once over your real captures (`python -m src.normalization detect`), then
cached in `config/axis_convention.json`. Each rule is a measurable property of
the geometry, and all evidence is written into the file.

**Left–right axis — by bilateral symmetry.** For each of the three axes, mirror
the mesh across the plane normal to it and measure the RMS nearest-neighbour
error, normalised by centroid size. A face is bilaterally symmetric about
exactly one of them, so that axis scores far lower. The evidence records the
per-axis errors and the margin between best and second-best; a margin below
`2×` triggers a warning, because the detection was not decisive.

**Left–right sign — from the eye transforms.** Apple names
`leftEyeTransform` and `rightEyeTransform`, so their relative position settles
which direction is the subject's left. **Without them the sign is UNRESOLVED**:
one is chosen, `side_resolved` is set `False`, and side-specific features are
suppressed. Geometry alone cannot answer this question.

**Anterior axis and sign — by skewness.** The nose is a protruding tail on an
otherwise compact distribution, so of the two remaining axes the anterior one
has the larger absolute skewness, and the tail points anteriorly.

**Superior axis and sign.** The remaining axis. Sign from the eye midpoint
sitting above the mesh centroid when eye transforms exist; otherwise from the
face being broader superiorly (forehead, cheekbones) than inferiorly (the chin
tapers).

**Handedness guard.** If the detected signs produce a left-handed frame, the
anterior sign — the least certain of the three — is flipped, and
`anterior_sign_flipped_for_handedness` is recorded.

### Confirm it visually

Detection is a set of heuristics over real data, so it is not taken on trust:

```bash
python -m src.visualization data/raw/P0001/capture.json --frame
```

Green `+Y` must point at the forehead, blue `+Z` out of the face. Then:

```bash
python -m src.normalization detect --save --confirm
```

Until confirmed, every normalised face carries a note saying so.

## Step 3 — translation

`vertices := vertices − origin`

| `origin` | Definition | Notes |
|---|---|---|
| `centroid` *(default)* | mean of all vertices | Landmark-free, robust; no single vertex moves it much. |
| `eye_midpoint` | midpoint of the two eye transforms | Anatomically interpretable. Requires eye transforms. |
| `nose_tip_extremum` | most anterior vertex | A geometric extremum, **not** a picked landmark. Sensitive to a single noisy vertex. |

## Step 4 — scale

`vertices := vertices / scale_reference`

| `scale` | Definition | Notes |
|---|---|---|
| `centroid_size` *(default)* | `sqrt(mean‖v − c‖²)` | The standard measure in geometric morphometrics. Uses every vertex; no landmarks. |
| `interocular` | `‖left_eye − right_eye‖` | Anatomically meaningful. Requires eye transforms. |
| `none` | `1.0` | Keeps true metres. |

The divisor is **always retained** as the feature `scale_reference_m`, so
absolute size is never lost. ARKit is metrically calibrated, so real
millimetres are genuinely available here — unlike a photo pipeline, where
absolute scale is unrecoverable. Whether size matters to human ratings is an
empirical question, and discarding it would prevent the model from answering it.

## Step 5 — optional Procrustes refinement

`--procrustes`. Generalized Procrustes analysis against the cohort mean shape,
**rotation only** (no scaling, no reflection).

Correspondence is by vertex **index**, which is exact because every capture in a
cohort shares one ARKit topology — validation enforces that. The Kabsch solution
carries the usual determinant correction forbidding reflection.

This removes residual orientation jitter from the ARKit tracker. It is a
refinement of an already-good alignment, not the primary pose normalisation.

---

## What is recorded with every normalised face

`NormalizedFace` carries the full provenance: `origin_mode`, `origin_offset_m`,
`scale_mode`, `scale_reference_m`, the `axis_convention` used, `side_resolved`,
and a `notes` list of every caveat that applied. No transformation is applied
without being recoverable from the object.

---

## Verifying normalisation actually worked

The real test needs **two captures of the same person at different distances and
head angles**. Capture that pair early — it is the cheapest possible check on
the whole front half of the pipeline.

```bash
python -m src.dataset build
python -c "
import pandas as pd
t = pd.read_csv('data/processed/features.csv')
same = t[t.participant_id == 'P0001']
print(same[['capture_id','face_width','face_height','width_height_ratio','symmetry_rms']])
"
```

Ratio features (`width_height_ratio`, `symmetry_rms`, the band ratios) should
agree closely across those captures. If they do not, normalisation is not doing
its job — the most likely cause is assumption **A4** (vertices exported in world
space without being declared as such).
