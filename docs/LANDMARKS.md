# Landmarks: how points on the face are identified

## Why `config/landmarks.json` ships empty

Apple does not publish a landmark index for `ARFaceGeometry`. There is no
documented, guaranteed-stable mapping from "vertex 1089" to "tip of the nose".

Index lists do circulate online. They are reverse-engineered, unversioned, and
**they disagree with each other**. Hard-coding one would mean every downstream
measurement is silently wrong in a way no test could catch — the numbers would
look perfectly reasonable and mean nothing.

So the pipeline ships with **no landmark indices at all**, and takes one of three
routes to any given measurement instead.

---

## The three routes, in order of preference

### 1. Landmark-free (always available)

Measurements that need no named point:

* gross extents, ratios, surface area, convex-hull volume, PCA shape descriptors
* **width and depth profiles** sampled at deciles of face height
* symmetry (the midplane is fitted; see `src/symmetry.py`)

The profile features deserve a note. `width_band_2` measures how wide the face is
at 25% of its height — which is roughly where the jaw is. It is *not* called
`jaw_width`, because the band edge is defined by a proportion of the face, not by
the gonion. It measures the same underlying thing in a reproducible way, without
claiming an anatomical correspondence it has not earned.

### 2. Derived from the midline curve (available, documented rules)

Points found as **extrema of the midsagittal profile** — see
`src/features.py::MidlineLandmarks` for the exact rules:

| Point | Rule |
|---|---|
| `nose_tip` | global maximum of `z` on the midline curve |
| `nasion` | first local minimum of `z` **above** the nose tip |
| `subnasale` | first local minimum of `z` **below** the nose tip |
| `pogonion` | first local maximum of `z` **below** subnasale |
| `menton` | lowest `y` on the curve |

"First" means *nearest to the nose tip*, walking outward — not the most extreme
point in the region. Nasion is the dip immediately above the nose; taking a
global extremum can latch onto something near the edge of the mesh instead.

These are reproducible geometric definitions. They **approximate** the clinical
landmarks of the same name; soft-tissue clinical definitions differ slightly.

When a curve has no clear extremum, the point is reported **not found** and every
dependent feature is `NaN`. Nothing is back-filled.

### 3. Hand-picked indices (optional, yours to define)

For measurements that genuinely need a named point — eye corners, mouth corners,
alae, gonion, zygion — you identify the index on **your own mesh**.

---

## Populating the landmark file

```bash
# 1. Click points on a real capture; prints the vertex index of each click.
python -m src.landmarks pick data/raw/P0001/frontal_001.json

# 2. Re-run assigning names, in click order.
python -m src.landmarks pick data/raw/P0001/frontal_001.json \
    --assign nose_tip chin_tip mouth_left_corner mouth_right_corner

# 3. Render with landmarks highlighted and CHECK EACH ONE sits where you think.
python -m src.landmarks verify data/raw/P0001/frontal_001.json

# 4. Confirm the set applies across the cohort.
python -m src.landmarks check data/raw

# 5. See current state at any time.
python -m src.landmarks show
```

With Open3D installed, picking is shift+left-click in its editing visualiser.
Without it, a matplotlib fallback picks the nearest projected vertex to a click —
rotate to a good viewpoint first, since accuracy depends on the projection.

`config/landmarks.json` carries a `definitions` block with the anatomical
definition of each name (nasion, exocanthion, gonion, zygion, …). Read it before
picking — the point is to pick the landmark the definition describes.

---

## Two constraints that will bite you

**Indices are topology-specific.** A landmark set records the `vertex_count` of
the mesh it was picked on, and the pipeline **refuses** to apply it to a mesh
with a different count. If `python -m src.validation` reports inconsistent vertex
counts, your indices do not transfer across the differing captures.

**Indices assume ARKit's vertex ordering is stable.** It has been in practice,
but Apple does not guarantee it across iOS versions or device generations. If
you capture across a major iOS update, re-run `python -m src.landmarks verify` on
a new capture before trusting the old indices.

---

## Left and right

`left_*` and `right_*` mean the **subject's** left and right, not the viewer's.

Which signed axis corresponds to the subject's left is resolved from the ARKit
eye transforms; without them it is **unresolved**, and side-specific features are
suppressed rather than guessed. See assumption A7 in `ASSUMPTIONS.md`.

This does not affect symmetry, which is side-agnostic by construction, nor any
bilateral distance such as `lm_jaw_width` — swapping the labels on a pair of
points does not change the distance between them.
