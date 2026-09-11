# Looksmatch — ML pipeline

Python data-processing and machine-learning pipeline for 3D facial geometry
exported from an iOS ARKit / TrueDepth application.

```
iPhone TrueDepth → ARFaceGeometry → JSON → [ this repo ]
    → validation → normalised 3D geometry → facial measurements
    → human ratings → ML model → prediction
```

This repo contains **only** the Python side. It does not build the iOS app, and
it never generates or simulates TrueDepth data — every code path expects real
exported captures.

---

## Status

| Phase | Module | State |
|---|---|---|
| 1. JSON loader | `src/data_loader.py` | Built. Schema-flexible; **needs a real capture to confirm key names.** |
| 2. Visualisation | `src/visualization.py` | Built. Open3D when available, matplotlib fallback. |
| 3. Validation | `src/validation.py` | Built. Per-capture + cohort checks. |
| 4. Normalisation | `src/normalization.py` | Built. Axis convention **detected from your data**, not assumed. |
| 5. Features | `src/features.py` | Built. ~70 measurements across 4 availability tiers. |
| 6. Symmetry | `src/symmetry.py` | Built. Fitted midplane, dimensionless metric. |
| 7. Dataset table | `src/dataset.py` | Built. Writes `features.csv`. |
| 8. Human ratings | `src/ratings.py` | Built. Aggregation + inter-rater reliability. **Awaiting real ratings.** |
| 9. Baseline models | `src/train.py` | Built, **not run.** Refuses to run without real labels. |
| 10. Evaluation | `src/evaluate.py` | Built. Metrics, fairness audit, SHAP. |
| Future. Deep learning | — | **Not started, by design.** See "Deferred" below. |

59 unit tests pass (`pytest tests/`). They exercise the maths and the leakage
guarantees on geometric primitives and synthetic tables — never on fabricated
face data.

---

## Install

Requires Python 3.11+. This project is set up with **Python 3.13.7** on Windows.

```bash
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux

.venv\Scripts\python -m pytest tests/ -q                  # verify: 59 passed
```

### A note on Open3D

`requirements.txt` deliberately does **not** install Open3D. **Open3D publishes
no wheel for Python 3.13**, so `pip install open3d` fails on this interpreter.

Everything works without it — `src/visualization.py` falls back to matplotlib
automatically and tells you which backend it used. The fallback is less fluid to
rotate, and landmark picking is click-nearest-projected-vertex instead of
shift+click.

To get the Open3D viewer, build the venv on Python 3.12 instead:

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt open3d
```

---

## Quick start, once you have a capture

Put exported JSON in `data/raw/<participant_id>/`, then:

```bash
# 0. What is actually in the file? Run this FIRST. Assumes nothing.
python -m src.inspect_schema data/raw/P0001/frontal_001.json

# 1. Load it and print a summary.
python -m src.data_loader data/raw/P0001/frontal_001.json

# 2. LOOK AT IT. Confirm the exported geometry is a face.
python -m src.visualization data/raw/P0001/frontal_001.json
python -m src.visualization data/raw/P0001/frontal_001.json --mode wireframe

# 3. Validate the whole cohort.
python -m src.validation

# 4. Detect the axis convention, then confirm it visually.
python -m src.normalization detect --save
python -m src.visualization data/raw/P0001/frontal_001.json --frame
python -m src.normalization detect --save --confirm

# 5. Symmetry.
python -m src.symmetry data/raw/P0001/frontal_001.json
python -m src.visualization data/raw/P0001/frontal_001.json --mode symmetry

# 6. Features for one capture, then the whole cohort.
python -m src.features data/raw/P0001/frontal_001.json
python -m src.dataset build
python -m src.dataset describe
```

**If step 0 reports an unresolved required field**, add the real key name to the
front of the matching list in `config/schema_map.json`. No Python change needed.

Everything from step 7 on needs human ratings and is described below.

---

## Design decisions worth knowing

**Nothing about attractiveness is hard-coded.** `src/features.py` measures
geometry and nothing else — no weights, no ideal ratios, no beauty constants.
Whether any measurement relates to human ratings is left entirely to the model,
which needs labels to answer it.

**The target is a human consensus, not a fact.** `src/ratings.py` produces a mean
rating with a standard deviation, a rating count and a confidence interval — a
measurement of how one panel responded, not a property of a person. It also
reports **ICC(2,k)**, the reliability of that mean, which is an approximate
ceiling on the R² any model can reach. Check it before concluding a model
underperformed.

**Splits are always at the participant level.** Two captures of one face are
nearly identical geometry; a random row-level split lets a model score well by
recognising the individual. `src/splits.py` groups by `participant_id` and
`assert_no_leakage()` **raises** rather than warns.

**Assumptions are recorded, not hidden.** Every inference the loader makes lands
in `FaceCapture.assumptions` and surfaces in the validation report. Unavailable
measurements are `NaN` with a stated reason — never a silent fallback value.

**Counts are derived, never hard-coded.** No vertex or triangle count is asserted
anywhere. The cohort's expected topology is the modal count across captures that
actually load.

Full detail: **`docs/ASSUMPTIONS.md`** (what to verify with your first real
capture), **`docs/NORMALIZATION.md`** (every transformation), **`docs/LANDMARKS.md`**
(how points are identified).

---

## The highest-value change to the iOS export

Export `ARFaceAnchor.leftEyeTransform` and `rightEyeTransform`.

They are the **only anatomically-named points ARKit provides** without guessing
vertex indices. They resolve subject-left vs subject-right — which geometry alone
cannot — and give a true interpupillary distance for scale normalisation. Without
them, side-specific features are suppressed rather than guessed.

```swift
"leftEyeTransform":  flatten(faceAnchor.leftEyeTransform),
"rightEyeTransform": flatten(faceAnchor.rightEyeTransform),
"lookAtPoint":       [faceAnchor.lookAtPoint.x, faceAnchor.lookAtPoint.y, faceAnchor.lookAtPoint.z],
```

Also worth adding: `"coordinateSpace": "face"` (or `"world"`), and `"units": "m"` —
both remove a real ambiguity (assumptions A4 and A5).

---

## Phases 7–10, once ratings exist

```bash
python -m src.ratings template            # writes the header template
# ... collect ratings into data/labels/ratings.csv ...

python -m src.ratings aggregate           # validates, reports ICC, writes targets.csv
python -m src.splits --folds 5            # inspect the participant-level split
python -m src.train --models ridge random_forest xgboost --save
python -m src.evaluate --model models/xgboost.pkl --importance --shap
```

`ratings.csv` is `participant_id,rater_id,rating` — one row per pair.

`src/train.py` **exits with instructions** rather than running if
`targets.csv` is absent. There is no demo mode and no synthetic target: a model
trained on invented ratings produces output indistinguishable from a real result.

### Fairness auditing

```bash
python -m src.evaluate --predictions outputs/predictions_xgboost.csv \
    --demographics data/labels/demographics.csv --group-column ancestry_group
```

Demographic attributes are **never model inputs** and never adjust a prediction.
They are used only to measure whether error differs across groups — an audit for
dataset and model bias. Per-group calibration would encode different standards
per group, which is the opposite of the goal. Groups below `min_group_size` are
reported but flagged as underpowered.

### Interpretability

`src/evaluate.py` provides model importances, permutation importance, and SHAP —
each printed with `CAUSAL_DISCLAIMER`, which spells out the difference between
*correlation this model learned* and *a causal claim about attractiveness*, and
the three specific reasons the ranking can mislead (collinearity, confounding,
target noise).

---

## Deferred: deep learning on raw geometry

PointNet / PointNet++ / GNN / mesh networks are **deliberately not implemented**.
They come after the classical baseline and the dataset pipeline work, and they
need substantially more labelled data than a first cohort will have. Starting
there would produce an unfalsifiable result on a pipeline whose front half has
not yet been checked against real captures.

---

## Layout

```
config/     schema_map.json       key aliases  -- EDIT THIS, not the loader
            landmarks.json        landmark indices -- ships empty, by design
            axis_convention.json  written by `normalization detect` (gitignored)
data/raw/         exported JSON, one dir per participant   (gitignored)
data/processed/   features.csv, targets.csv                (gitignored)
data/labels/      ratings.csv                              (gitignored)
src/        the pipeline, one module per phase
docs/       ASSUMPTIONS.md, NORMALIZATION.md, LANDMARKS.md
notebooks/  explore_mesh.ipynb, explore_features.ipynb
tests/      test_pipeline.py
```

`data/` and `models/` are gitignored: captured face geometry is **personal
biometric data**. `features.csv` excludes paths, timestamps and device
identifiers, but a set of facial measurements is a biometric template — treat it
as pseudonymous, not anonymous, and keep the ID mapping separate.
