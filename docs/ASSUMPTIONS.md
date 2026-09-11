# Assumptions, and how to check them against real data

This pipeline was written before a real capture from the iOS app was available
to inspect. Everything it could not verify is listed here. Nothing in this file
is settled; each item says what to run and what to look for.

Work through **Tier 1** with your first real capture before trusting any number
the pipeline produces.

---

## Tier 1 — must be checked with the first real capture

### A1. JSON key names

**Assumed:** keys resembling `vertices`, `triangleIndices`, `participantId`,
`blendShapes`, `transform`. All candidates live in `config/schema_map.json`.

**Check:**
```bash
python -m src.inspect_schema data/raw/P0001/your_capture.json
```
This prints the file's real key tree and which canonical fields resolved.

**If a field did not resolve:** add the real key name to the front of the
matching list in `config/schema_map.json`. **No Python change is needed.**

---

### A2. Vertex array layout

**Assumed:** one of nested `[[x,y,z],…]`, flat `[x,y,z,x,y,z,…]`, dict records,
or component arrays. The loader detects which and records the choice in
`FaceCapture.assumptions`.

**Edge case handled:** a `(N,4)` array is read as SIMD `float3` padded to 16
bytes and the 4th component is dropped, recorded as
`vertices_had_4_components_dropped_4th_simd_padding`. **If your exporter
actually writes meaningful `w` values, this is wrong** — say so and the loader
will be changed.

**Check:** the `Assumptions:` line of `python -m src.data_loader <file>`.

---

### A3. Vertex and triangle counts

**Not assumed.** No count is hard-coded anywhere. ARKit's commonly reported
1220 / 2304 appears only in `src/config.py` as an *informational* reference that
annotates the validation report; it never passes or fails a capture.

The authoritative expectation is **derived from your own cohort**: the modal
counts across captures that load. A capture deviating from the cohort mode is
flagged, because a feature vector is only comparable within a fixed topology.

**Check:** the header line of `python -m src.validation`.

---

### A4. Coordinate space of the exported vertices

**Assumed when the export does not say:** vertices are in **face-anchor local
space** — ARKit's native `ARFaceGeometry.vertices` output. Recorded as
`coordinate_space_defaulted_to_face_local`.

**This matters a great deal.** If the app instead baked the anchor transform
into the vertices (world space), then every capture carries the head pose and
camera distance, and normalisation must invert the transform. The pipeline does
that correctly *if told* — but it cannot detect the difference reliably.

**Check:** in the iOS app, does the export write `faceGeometry.vertices`
directly, or `transform * vertex`? If the latter, add
`"coordinateSpace": "world"` to the export (and make sure the anchor `transform`
is exported too).

**Symptom if wrong:** captures of the same person at different distances or head
angles produce wildly different measurements; `python -m src.normalization
detect` reports a weak left-right margin (`< 2x`).

---

### A5. Units

**Assumed:** metres, which is what ARKit reports. Add a `units` field to the
export if you convert to mm or cm; the loader will rescale.

**Check:** `Bbox diagonal` in `python -m src.data_loader <file>` should be
roughly **0.15–0.30 m** for a face. Validation warns outside 0.05–0.40 m.
A diagonal near 200 means millimetres.

---

### A6. Axis convention

**Not assumed — detected**, from your real data, with evidence you can inspect.
See `docs/NORMALIZATION.md` for the detection rules.

```bash
python -m src.normalization detect --save
python -m src.visualization data/raw/P0001/your_capture.json --frame
```

The second command draws the canonical axes on the mesh. **Look at it.** The
green `+Y` arrow must point at the forehead and the blue `+Z` arrow out of the
face. When it does:

```bash
python -m src.normalization detect --save --confirm
```

Until you pass `--confirm`, every normalised face carries the note
*"axis convention not yet visually confirmed"*.

---

### A7. Subject left vs subject right

**Cannot be resolved from geometry alone.** A face mesh does not say which side
is the person's left.

It **is** resolved if the export contains `ARFaceAnchor.leftEyeTransform` and
`rightEyeTransform`, which Apple names explicitly.

**Without them:** `NormalizedFace.side_resolved` is `False`, and side-specific
features are suppressed — bilateral and midline features are unaffected, and
symmetry is unaffected (it is side-agnostic by construction).

**This is the single highest-value addition you can make to the iOS export**, so
it is worth saying plainly:

```swift
// ARFaceAnchor exposes these directly; both are relative to the face anchor,
// i.e. the same frame as ARFaceGeometry.vertices.
"leftEyeTransform":  flatten(faceAnchor.leftEyeTransform),
"rightEyeTransform": flatten(faceAnchor.rightEyeTransform),
"lookAtPoint":       [faceAnchor.lookAtPoint.x, .y, .z],
```

They also give a true interpupillary distance for scale normalisation, and they
are the only anatomically-named points ARKit provides without guessing vertex
indices.

---

## Tier 2 — checked automatically, but worth understanding

### A8. Matrix layout (column-major vs row-major)

Swift's `simd_float4x4` is **column-major** (`columns.0 … columns.3`). An
exporter that flattens columns in order produces, after a naive row-major
reshape, the **transpose** of the conventional matrix.

The loader detects this by checking where the `[0,0,0,1]` homogeneous row/column
landed, and transposes when needed, recording
`transform_transposed_from_column_major_simd_layout`. A matrix that is affine in
neither layout is left alone and flagged
(`..._not_affine_layout_left_as_given`) — investigate if you see that.

Covered by `tests/test_pipeline.py::TestMatrixCoercion`.

---

### A9. Timestamps

`ARFrame.timestamp` is a **monotonic clock since device boot**, not Unix time.
The loader only interprets a number as an epoch when it falls in a sane calendar
range (2001–2096); otherwise it is kept verbatim. ISO-8601 strings parse
normally. Timestamps are not used in any feature and never reach `features.csv`.

---

### A10. Landmark indices

**Not assumed — none are shipped.** Apple publishes no landmark index for
ARFaceGeometry, and the reverse-engineered lists circulating online are
unversioned and mutually inconsistent. `config/landmarks.json` ships **empty**
and every landmark-based feature is `NaN` until you populate it yourself.

See `docs/LANDMARKS.md`. Landmark-free and midline-derived features need none of
this and work on day one.

---

### A11. Midline-derived points

`nose_tip`, `nasion`, `subnasale`, `pogonion`, `menton` are found as **extrema
of the midsagittal profile curve**, by rules documented in
`src/features.py::MidlineLandmarks`. They are reproducible geometric
definitions, not picked indices, and they *approximate* the clinical landmarks
of the same name rather than matching them exactly.

When a curve has no clear extremum, the point is reported as **not found** and
dependent features are `NaN`. Nothing is back-filled with a guess.

**Check once you have a real face:** run
`python -m src.visualization <file> --mode mesh` and confirm the curve extrema
make sense; the profile itself is plotted by `notebooks/explore_features.ipynb`.

---

### A12. `upper_region_height` is not the clinical upper third

The clinical upper third runs trichion (hairline) → glabella. **ARFaceGeometry
does not reliably extend to the hairline.** The feature named
`upper_region_height` is therefore measured *nasion → top of the mesh*, which
depends on where ARKit's mesh happens to end. It is included because it is
reproducible, but it is **not** the anatomical upper third and should not be
reported as one. `middle_third_height` and `lower_third_height` are on firmer
ground.

---

## Tier 3 — properties of the method, not of your data

### A13. Symmetry metric

Definition, including the fitted midplane and the discretization floor, is in
the `src/symmetry.py` module docstring. Two points worth repeating:

* The midplane is **fitted**, not assumed to be `x = 0`. A head sitting slightly
  off-axis in the anchor frame would otherwise register as asymmetric.
* The default method is `point_to_surface`, which has **no discretization
  floor**. The faster `nearest_vertex` has one, and its `mirror_pairing_fraction`
  diagnostic is confounded — pairing falls both on a coarse mesh *and* on a
  genuinely asymmetric face. This was found during development and is pinned by
  a regression test.

### A14. Scale reference

Default is **centroid size** — `sqrt(mean ||v − centroid||²)` — the standard
scale measure in geometric morphometrics. It uses every vertex, so no single
noisy point moves it, and it needs no landmarks.

`interocular` is available when eye transforms are exported and is more
anatomically interpretable. `none` keeps true metres, which is meaningful here
because **ARKit is metrically calibrated** — a genuine advantage over
photo-based pipelines, and the reason `scale_reference_m` is retained as a
feature rather than discarded.

### A15. Re-identification

`features.csv` excludes paths, timestamps, device identifiers and raw geometry.
But a ~70-dimensional set of facial measurements is a **biometric template**, not
anonymous data. Treat the CSV as *pseudonymous*, keep the participant-ID mapping
somewhere else, and keep `data/raw/` off version control (it is gitignored).
