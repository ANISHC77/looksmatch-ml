"""Unit tests for the Looksmatch pipeline.

SCOPE NOTE
----------
These tests use GEOMETRIC PRIMITIVES (grids, spheres, known transforms) and
SYNTHETIC TABLES (integer ratings, participant ids). They exist to prove the
maths and the leakage guarantees are correct.

They never simulate TrueDepth output and none of this data reaches data/raw/,
data/processed/ or any model. Correctness of the pipeline ON REAL CAPTURES can
only be established with real captures -- see docs/ASSUMPTIONS.md for what must
be checked once you export one.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import symmetry
from src.config import SchemaMap
from src.data_loader import (
    CaptureLoadError,
    FaceCapture,
    as_blend_shapes,
    as_index_array,
    as_matrix4,
    as_vertex_array,
    load_capture,
    resolve_field,
)
from src.evaluate import evaluate_predictions, group_metrics
from src.features import derive_midline_landmarks, surface_area, width_profile
from src.normalization import AxisConvention, NormalizedFace, procrustes_align
from src.ratings import aggregate_ratings, intraclass_correlation, validate_ratings
from src.splits import LeakageError, assert_no_leakage, participant_folds, participant_split
from src.validation import validate_capture


# ==========================================================================
# Fixtures: geometric primitives
# ==========================================================================

def symmetric_grid(n: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """A triangulated surface, mirror-symmetric about x=0 by construction."""
    xs = np.linspace(-1.0, 1.0, n)
    ys = np.linspace(-1.3, 1.3, n)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    gz = -(0.5 * gx ** 2 + 0.3 * gy ** 2)
    vertices = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    triangles = np.array(
        [t for i in range(n - 1) for j in range(n - 1)
         for t in ([i * n + j, i * n + j + 1, (i + 1) * n + j + 1],
                   [i * n + j, (i + 1) * n + j + 1, (i + 1) * n + j])],
        dtype=np.int32,
    )
    return vertices, triangles


@pytest.fixture
def grid():
    return symmetric_grid()


# ==========================================================================
# Loader: array coercion
# ==========================================================================

class TestVertexCoercion:
    def test_nested_lists(self):
        out = as_vertex_array([[1, 2, 3], [4, 5, 6]], [])
        assert out.shape == (2, 3)
        assert out.dtype == np.float64

    def test_flat_array_reshapes(self):
        notes: list[str] = []
        out = as_vertex_array([1, 2, 3, 4, 5, 6], notes)
        assert out.shape == (2, 3)
        assert "vertices_reshaped_from_flat_xyz" in notes

    def test_flat_array_not_divisible_by_three_raises(self):
        with pytest.raises(CaptureLoadError, match="divisible by 3"):
            as_vertex_array([1, 2, 3, 4], [])

    def test_dict_records(self):
        notes: list[str] = []
        out = as_vertex_array([{"x": 1, "y": 2, "z": 3}], notes)
        assert np.allclose(out, [[1, 2, 3]])
        assert "vertices_from_dict_records" in notes

    def test_component_arrays(self):
        notes: list[str] = []
        out = as_vertex_array({"x": [1, 4], "y": [2, 5], "z": [3, 6]}, notes)
        assert np.allclose(out, [[1, 2, 3], [4, 5, 6]])
        assert "vertices_from_component_arrays" in notes

    def test_simd_padding_dropped(self):
        """simd_float3 is 16-byte aligned; a raw dump yields a 4th component."""
        notes: list[str] = []
        out = as_vertex_array([[1, 2, 3, 0], [4, 5, 6, 0]], notes)
        assert out.shape == (2, 3)
        assert any("simd_padding" in n for n in notes)

    def test_empty_raises(self):
        with pytest.raises(CaptureLoadError, match="empty"):
            as_vertex_array([], [])


class TestIndexCoercion:
    def test_flat_indices_reshape(self):
        notes: list[str] = []
        out = as_index_array([0, 1, 2, 1, 2, 3], 4, notes)
        assert out.shape == (2, 3)
        assert out.dtype == np.int32
        assert "triangles_reshaped_from_flat_ijk" in notes

    def test_out_of_range_raises(self):
        with pytest.raises(CaptureLoadError, match="out of range"):
            as_index_array([0, 1, 99], 4, [])

    def test_negative_raises(self):
        with pytest.raises(CaptureLoadError, match="negative"):
            as_index_array([0, 1, -2], 4, [])

    def test_non_integer_raises(self):
        with pytest.raises(CaptureLoadError, match="non-integer"):
            as_index_array([0.0, 1.5, 2.0], 4, [])


class TestMatrixCoercion:
    """Swift's simd_float4x4 is column-major; a naive reshape transposes it."""

    def test_row_major_preserved(self):
        matrix = np.array([[1, 0, 0, 7], [0, 1, 0, 8], [0, 0, 1, 9], [0, 0, 0, 1]], float)
        out = as_matrix4(matrix.ravel().tolist(), [], "t")
        assert np.allclose(out[:3, 3], [7, 8, 9])

    def test_column_major_flat_is_transposed_back(self):
        matrix = np.array([[1, 0, 0, 7], [0, 1, 0, 8], [0, 0, 1, 9], [0, 0, 0, 1]], float)
        notes: list[str] = []
        out = as_matrix4(matrix.T.ravel().tolist(), notes, "t")  # column-major dump
        assert np.allclose(out[:3, 3], [7, 8, 9]), "translation must land in column 3"
        assert any("column_major" in n for n in notes)

    def test_wrong_length_ignored(self):
        notes: list[str] = []
        assert as_matrix4([1, 2, 3], notes, "t") is None
        assert any("unexpected_length" in n for n in notes)


def test_blend_shapes_from_records():
    notes: list[str] = []
    out = as_blend_shapes([{"name": "jawOpen", "value": 0.5}], notes)
    assert out == {"jawOpen": 0.5}


def test_resolve_field_is_case_and_underscore_insensitive():
    blob = {"faceGeometry": {"triangle_indices": [0, 1, 2]}}
    value, path = resolve_field(blob, ["faceGeometry.triangleIndices"])
    assert value == [0, 1, 2]
    assert path is not None


def test_resolve_field_falls_back_to_recursive_search():
    blob = {"payload": {"nested": {"blendShapes": {"jawOpen": 0.1}}}}
    value, path = resolve_field(blob, ["blendShapes"])
    assert value == {"jawOpen": 0.1}


# ==========================================================================
# Loader: whole-file round trip
# ==========================================================================

def test_load_capture_round_trip(tmp_path, grid):
    """A file in the documented schema loads with the expected shapes."""
    vertices, triangles = grid
    payload = {
        "participantId": "P0001",
        "captureId": "frontal_001",
        "timestamp": "2026-01-15T10:30:00",
        "vertices": vertices.ravel().tolist(),
        "triangleIndices": triangles.ravel().tolist(),
        "blendShapes": {"jawOpen": 0.02},
        "coordinateSpace": "face",
    }
    path = tmp_path / "frontal_001.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    capture = load_capture(path, schema=SchemaMap.load())
    assert capture.participant_id == "P0001"
    assert capture.capture_id == "frontal_001"
    assert capture.vertices.shape == vertices.shape
    assert capture.triangles.shape == triangles.shape
    assert capture.coordinate_space == "face_local"
    assert np.allclose(capture.vertices, vertices)


def test_malformed_json_names_the_file(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"vertices": [1, 2,', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        load_capture(path)


def test_participant_id_falls_back_to_directory(tmp_path, grid):
    vertices, triangles = grid
    folder = tmp_path / "P0042"
    folder.mkdir()
    path = folder / "c1.json"
    path.write_text(json.dumps({
        "vertices": vertices.ravel().tolist(),
        "triangleIndices": triangles.ravel().tolist(),
    }), encoding="utf-8")
    capture = load_capture(path)
    assert capture.participant_id == "P0042"
    assert "participant_id_inferred_from_parent_directory" in capture.assumptions


# ==========================================================================
# Validation
# ==========================================================================

def _capture(vertices, triangles, **kwargs) -> FaceCapture:
    return FaceCapture(
        participant_id=kwargs.pop("participant_id", "P0001"),
        capture_id="c1", vertices=vertices, triangles=triangles, **kwargs,
    )


class TestValidation:
    def test_clean_capture_has_no_errors(self, grid):
        report = validate_capture(_capture(*grid))
        assert not report.errors, [str(i) for i in report.errors]

    def test_nan_is_an_error(self, grid):
        vertices, triangles = grid
        vertices = vertices.copy()
        vertices[5, 1] = np.nan
        report = validate_capture(_capture(vertices, triangles))
        assert report.status == "FAIL"
        assert report.n_nan == 1
        assert any(i.code == "nan_vertices" for i in report.errors)

    def test_infinity_is_an_error(self, grid):
        vertices, triangles = grid
        vertices = vertices.copy()
        vertices[3, 0] = np.inf
        report = validate_capture(_capture(vertices, triangles))
        assert report.n_inf == 1
        assert report.status == "FAIL"

    def test_missing_participant_id_is_an_error(self, grid):
        report = validate_capture(_capture(*grid, participant_id=""))
        assert any(i.code == "missing_participant_id" for i in report.errors)

    def test_declared_count_mismatch_is_caught(self, grid):
        vertices, triangles = grid
        report = validate_capture(_capture(vertices, triangles, declared_vertex_count=9999))
        assert any(i.code == "vertex_count_mismatch" for i in report.errors)

    def test_collapsed_mesh_is_an_error(self, grid):
        _, triangles = grid
        report = validate_capture(_capture(np.zeros((441, 3)), triangles))
        assert any(i.code == "collapsed_mesh" for i in report.errors)

    def test_degenerate_triangles_warn(self, grid):
        vertices, triangles = grid
        triangles = triangles.copy()
        triangles[0] = [5, 5, 6]
        report = validate_capture(_capture(vertices, triangles))
        assert any(i.code == "degenerate_triangles" for i in report.warnings)

    def test_report_renders_the_requested_fields(self, grid):
        text = validate_capture(_capture(*grid)).render()
        for field in ("Vertices:", "Triangles:", "NaN values:", "Missing values:"):
            assert field in text


# ==========================================================================
# Symmetry
# ==========================================================================

class TestSymmetry:
    def test_symmetric_mesh_scores_zero(self, grid):
        result = symmetry.measure_symmetry(*grid)
        assert result.symmetry_rms == pytest.approx(0.0, abs=1e-9)
        assert result.mirror_pairing_fraction == pytest.approx(1.0)

    def test_asymmetry_is_detected(self, grid):
        vertices, triangles = grid
        dented = vertices.copy()
        dented[dented[:, 0] > 0, 2] -= 0.1
        result = symmetry.measure_symmetry(dented, triangles)
        assert result.symmetry_rms > 0.01
        assert result.method == "point_to_surface", "the default must be the floor-free method"
        assert result.exceeds_floor

    def test_nearest_vertex_floor_is_conservative_on_real_asymmetry(self, grid):
        """Regression test for a confound found during development.

        `mirror_pairing_fraction` falls both when the mesh is coarse AND when the
        face is genuinely asymmetric, so the nearest_vertex artefact bound can
        swallow a real 0.1-deep dent. point_to_surface must still see it, which
        is why it is the default.
        """
        vertices, triangles = grid
        dented = vertices.copy()
        dented[dented[:, 0] > 0, 2] -= 0.1
        coarse = symmetry.measure_symmetry(dented, triangles, method="nearest_vertex")
        exact = symmetry.measure_symmetry(dented, triangles, method="point_to_surface")
        assert not coarse.exceeds_floor, "the confound is real and must stay documented"
        assert exact.exceeds_floor, "the floor-free method must still detect the dent"
        assert exact.effective_floor == 0.0

    def test_metric_is_scale_invariant(self, grid):
        """Doubling the mesh must not change a dimensionless ratio."""
        vertices, triangles = grid
        dented = vertices.copy()
        dented[dented[:, 0] > 0, 2] -= 0.1
        small = symmetry.measure_symmetry(dented, triangles)
        large = symmetry.measure_symmetry(dented * 2.0, triangles)
        assert small.symmetry_rms == pytest.approx(large.symmetry_rms, rel=1e-6)

    def test_midplane_is_fitted_not_assumed(self, grid):
        """A mesh shifted off x=0 is still perfectly symmetric about its own midline."""
        vertices, triangles = grid
        shifted = vertices + np.array([0.37, 0.0, 0.0])
        result = symmetry.measure_symmetry(shifted, triangles)
        assert result.symmetry_rms == pytest.approx(0.0, abs=1e-6)
        assert result.plane_offset == pytest.approx(0.37, abs=1e-3)

    def test_discretization_floor_is_reported(self):
        """Irregular sampling of a SYMMETRIC surface must be flagged, not scored."""
        rng = np.random.default_rng(1)
        pts = rng.uniform(-1, 1, size=(600, 2)) * np.array([1.0, 1.3])
        vertices = np.column_stack([pts[:, 0], pts[:, 1], -(0.5 * pts[:, 0] ** 2 + 0.3 * pts[:, 1] ** 2)])
        result = symmetry.measure_symmetry(vertices)
        assert result.mirror_pairing_fraction < 0.1
        assert not result.exceeds_floor, "artefact must not read as real asymmetry"

    def test_reflection_is_an_involution(self, grid):
        vertices, _ = grid
        normal = np.array([0.6, 0.8, 0.0])
        once = symmetry.reflect(vertices, normal, 0.3)
        twice = symmetry.reflect(once, normal, 0.3)
        assert np.allclose(twice, vertices)


# ==========================================================================
# Normalisation
# ==========================================================================

class TestNormalization:
    def test_procrustes_undoes_a_known_rotation(self, grid):
        vertices, _ = grid
        angle = 0.4
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0],
                             [0, 0, 1]])
        recovered = procrustes_align(vertices @ rotation.T, vertices)
        assert np.allclose(recovered, vertices - vertices.mean(axis=0), atol=1e-9)

    def test_procrustes_never_reflects(self):
        """Reflecting a face would swap its left and right sides."""
        rng = np.random.default_rng(0)
        source = rng.normal(size=(60, 3))
        target = source.copy()
        target[:, 0] *= -1
        aligned = procrustes_align(source, target)
        assert not np.allclose(aligned, target - target.mean(axis=0), atol=1e-6)

    def test_axis_convention_matrix_is_a_signed_permutation(self):
        convention = AxisConvention(left=(0, 1), superior=(1, 1), anterior=(2, 1))
        assert np.allclose(convention.matrix(), np.eye(3))
        assert convention.is_right_handed()

    def test_axis_convention_round_trips_through_json(self, tmp_path):
        convention = AxisConvention(left=(2, -1), superior=(1, 1), anterior=(0, 1))
        path = tmp_path / "axes.json"
        convention.save(path)
        assert AxisConvention.load(path).left == (2, -1)


# ==========================================================================
# Features
# ==========================================================================

class TestFeatures:
    def test_surface_area_of_a_unit_square(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
        triangles = np.array([[0, 1, 2], [0, 2, 3]])
        assert surface_area(vertices, triangles) == pytest.approx(1.0)

    def test_width_profile_has_one_value_per_band(self, grid):
        profile = width_profile(grid[0], n_bands=9)
        assert profile.shape == (9,)
        assert np.all(np.isfinite(profile))

    def test_midline_reports_missing_points_rather_than_guessing(self):
        """A curve with no nose-root dip must yield nasion=None, not a fallback."""
        y = np.linspace(-1.0, 1.0, 120)
        z = 0.30 * np.exp(-((y - 0.05) / 0.10) ** 2) - 0.25 * y ** 2
        band = np.repeat(np.column_stack([np.zeros_like(y), y, z]), 3, axis=0)
        band[:, 0] += np.tile([-1e-4, 0.0, 1e-4], len(y))
        midline = derive_midline_landmarks(band)
        assert midline.nose_tip is not None
        assert midline.nose_tip[1] == pytest.approx(0.05, abs=0.05)
        assert midline.nasion is None
        assert any("nasion" in n for n in midline.notes)

    def test_midline_finds_the_nearest_extremum(self):
        """Nasion is the dip just above the nose, not the deepest dip anywhere."""
        y = np.linspace(-1.0, 1.0, 120)
        z = (0.30 * np.exp(-((y - 0.05) / 0.10) ** 2)
             + 0.10 * np.exp(-((y - 0.42) / 0.12) ** 2) - 0.25 * y ** 2)
        band = np.repeat(np.column_stack([np.zeros_like(y), y, z]), 3, axis=0)
        band[:, 0] += np.tile([-1e-4, 0.0, 1e-4], len(y))
        midline = derive_midline_landmarks(band)
        assert midline.nasion is not None
        assert 0.05 < midline.nasion[1] < 0.42


# ==========================================================================
# Splits -- the correctness property that matters most
# ==========================================================================

def _cohort(n_participants: int = 40, captures_each: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "participant_id": np.repeat([f"P{i:04d}" for i in range(n_participants)], captures_each),
        "capture_id": [f"c{j}" for _ in range(n_participants) for j in range(captures_each)],
        "feature_a": np.arange(n_participants * captures_each, dtype=float),
    })


class TestSplits:
    def test_no_participant_spans_the_split(self):
        split = participant_split(_cohort(), test_size=0.25, seed=0)
        assert not set(split.train["participant_id"]) & set(split.test["participant_id"])

    def test_every_capture_of_a_participant_stays_together(self):
        table = _cohort(captures_each=3)
        split = participant_split(table, test_size=0.25, seed=0)
        for frame in (split.train, split.test):
            for _, group in frame.groupby("participant_id"):
                assert len(group) == 3, "a participant's captures were split apart"

    def test_all_rows_are_accounted_for(self):
        table = _cohort()
        split = participant_split(table, test_size=0.25, seed=0)
        assert len(split.train) + len(split.test) == len(table)

    def test_assert_no_leakage_raises_on_overlap(self):
        table = _cohort(4, 2)
        with pytest.raises(LeakageError, match="BOTH train and test"):
            assert_no_leakage(table, table)

    def test_grouped_folds_never_leak(self):
        table = _cohort(30, 2)
        seen_test: set[str] = set()
        for train_idx, test_idx in participant_folds(table, n_splits=5):
            train_ids = set(table.iloc[train_idx]["participant_id"])
            test_ids = set(table.iloc[test_idx]["participant_id"])
            assert not train_ids & test_ids
            assert not seen_test & test_ids, "a participant appeared in two test folds"
            seen_test |= test_ids
        assert seen_test == set(table["participant_id"])

    def test_split_is_deterministic_for_a_seed(self):
        table = _cohort()
        a = participant_split(table, 0.25, seed=7)
        b = participant_split(table, 0.25, seed=7)
        assert list(a.test["participant_id"]) == list(b.test["participant_id"])


# ==========================================================================
# Ratings
# ==========================================================================

class TestRatings:
    def test_aggregate_computes_the_requested_statistics(self):
        ratings = pd.DataFrame({
            "participant_id": ["P1"] * 4 + ["P2"] * 4,
            "rater_id": ["R1", "R2", "R3", "R4"] * 2,
            "rating": [6, 7, 8, 7, 3, 4, 5, 4],
        })
        out = aggregate_ratings(ratings).set_index("participant_id")
        assert out.loc["P1", "mean_rating"] == pytest.approx(7.0)
        assert out.loc["P1", "median_rating"] == pytest.approx(7.0)
        assert out.loc["P1", "n_ratings"] == 4
        assert out.loc["P1", "ci_low"] < 7.0 < out.loc["P1", "ci_high"]

    def test_single_rating_has_no_confidence_interval(self):
        ratings = pd.DataFrame({"participant_id": ["P1"], "rater_id": ["R1"], "rating": [7]})
        row = aggregate_ratings(ratings).iloc[0]
        assert row["mean_rating"] == 7.0
        assert np.isnan(row["ci_low"]), "one observation cannot bound its own spread"

    def test_wider_disagreement_gives_a_wider_interval(self):
        tight = pd.DataFrame({"participant_id": ["P1"] * 5, "rater_id": list("ABCDE"),
                              "rating": [7, 7, 7, 8, 7]})
        loose = pd.DataFrame({"participant_id": ["P1"] * 5, "rater_id": list("ABCDE"),
                              "rating": [1, 10, 4, 9, 6]})
        assert (aggregate_ratings(tight).iloc[0]["ci_width"]
                < aggregate_ratings(loose).iloc[0]["ci_width"])

    def test_validation_flags_duplicates_and_out_of_range(self):
        ratings = pd.DataFrame({
            "participant_id": ["P1", "P1", "P2"],
            "rater_id": ["R1", "R1", "R2"],
            "rating": [7, 8, 99],
        })
        problems = validate_ratings(ratings)
        assert any("more than once" in p for p in problems)
        assert any("outside the declared scale" in p for p in problems)

    def test_icc_is_high_when_raters_agree(self):
        rng = np.random.default_rng(0)
        truth = rng.uniform(2, 9, size=30)
        rows = [{"participant_id": f"P{i}", "rater_id": f"R{r}",
                 "rating": truth[i] + rng.normal(0, 0.3)}
                for i in range(30) for r in range(5)]
        result = intraclass_correlation(pd.DataFrame(rows))
        assert result is not None
        assert result.icc_average > 0.9

    def test_icc_is_low_when_raters_disagree(self):
        rng = np.random.default_rng(0)
        rows = [{"participant_id": f"P{i}", "rater_id": f"R{r}",
                 "rating": rng.uniform(1, 10)}
                for i in range(30) for r in range(5)]
        result = intraclass_correlation(pd.DataFrame(rows))
        assert result is not None
        assert result.icc_average < 0.5


# ==========================================================================
# Metrics
# ==========================================================================

class TestMetrics:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        metrics = evaluate_predictions(y, y)
        assert metrics.mae == pytest.approx(0.0)
        assert metrics.r2 == pytest.approx(1.0)
        assert metrics.pearson_r == pytest.approx(1.0)

    def test_mean_only_prediction_scores_zero_r2(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        metrics = evaluate_predictions(y, np.full_like(y, y.mean()))
        assert metrics.r2 == pytest.approx(0.0)
        assert not metrics.beats_baseline

    def test_worse_than_the_mean_gives_negative_r2(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        metrics = evaluate_predictions(y, np.array([4.0, 3.0, 2.0, 1.0]))
        assert metrics.r2 < 0

    def test_group_metrics_flags_small_groups(self):
        table = pd.DataFrame({
            "mean_rating": [5.0] * 14,
            "predicted": [5.1] * 14,
            "group": ["A"] * 12 + ["B"] * 2,
        })
        out = group_metrics(table, "group", min_group_size=10).set_index("group")
        assert not out.loc["A", "underpowered"]
        assert out.loc["B", "underpowered"]
