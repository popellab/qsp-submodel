"""Tests for cascade cuts: BFS splitting, DAG building, posterior conversion."""

import numpy as np
import pytest

from qsp_inference.submodel.parameter_groups import (
    CascadeCut,
)


# =============================================================================
# BFS component splitting
# =============================================================================


def test_find_components_excludes_cascade_params():
    """Cascade cut param should split a mega-component into two."""
    from qsp_inference.submodel.comparison import (
        _find_components_lightweight,
    )

    # Three targets: A has {P, X}, B has {P, Y}, C has {X, Z}
    # Without cascade: A-B linked via P, A-C linked via X → one component
    # With cascade on P: A-C linked via X (one comp), B alone (another comp)
    # P appears in both components
    lightweight = [
        {"target_id": "A", "qsp_params": {"P", "X"}, "filename": "A.yaml"},
        {"target_id": "B", "qsp_params": {"P", "Y"}, "filename": "B.yaml"},
        {"target_id": "C", "qsp_params": {"X", "Z"}, "filename": "C.yaml"},
    ]

    # Without cascade: should be 1 component
    comps_no_cut = _find_components_lightweight(lightweight, None)
    assert len(comps_no_cut) == 1
    assert comps_no_cut[0]["params"] == {"P", "X", "Y", "Z"}

    # With cascade on P: should be 2 components
    comps_cut = _find_components_lightweight(lightweight, None, cascade_cut_params=frozenset({"P"}))
    assert len(comps_cut) == 2

    # Find which component has which targets
    comp_with_ac = next(c for c in comps_cut if "A.yaml" in c["target_filenames"])
    comp_with_b = next(c for c in comps_cut if "B.yaml" in c["target_filenames"])

    # A+C component has X, Z, and P (cascade param included but didn't merge)
    assert "P" in comp_with_ac["params"]
    assert "X" in comp_with_ac["params"]
    assert "Z" in comp_with_ac["params"]
    assert "C.yaml" in comp_with_ac["target_filenames"]

    # B component has Y and P
    assert "P" in comp_with_b["params"]
    assert "Y" in comp_with_b["params"]
    assert "B.yaml" in comp_with_b["target_filenames"]


def test_cascade_param_appears_in_multiple_components():
    """The cascade cut param should appear in both resulting components."""
    from qsp_inference.submodel.comparison import (
        _find_components_lightweight,
    )

    lightweight = [
        {"target_id": "A", "qsp_params": {"shared", "a_only"}, "filename": "A.yaml"},
        {"target_id": "B", "qsp_params": {"shared", "b_only"}, "filename": "B.yaml"},
    ]

    comps = _find_components_lightweight(
        lightweight, None, cascade_cut_params=frozenset({"shared"})
    )
    assert len(comps) == 2

    # Both components should contain "shared"
    for comp in comps:
        assert "shared" in comp["params"]


# =============================================================================
# DAG building
# =============================================================================


def test_build_stage_dag_simple():
    """Two components with one cascade cut → 2 stages."""
    from qsp_inference.submodel.comparison import _build_stage_dag

    components = [
        {"params": {"P", "X"}, "target_filenames": ["A.yaml"]},
        {"params": {"P", "Y"}, "target_filenames": ["B.yaml"]},
    ]
    cuts = [CascadeCut(parameter="P", upstream=["A"])]
    lightweight = [
        {"target_id": "A", "qsp_params": {"P", "X"}, "filename": "A.yaml"},
        {"target_id": "B", "qsp_params": {"P", "Y"}, "filename": "B.yaml"},
    ]

    stages, edges = _build_stage_dag(components, cuts, lightweight)

    assert len(stages) == 2
    assert 0 in stages[0]  # upstream (A) is stage 0
    assert 1 in stages[1]  # downstream (B) is stage 1
    assert edges["P"]["upstream_comp"] == 0
    assert edges["P"]["downstream_comps"] == [1]


def test_build_stage_dag_multihop():
    """A→B→C chain via two cascade cuts → 3 stages."""
    from qsp_inference.submodel.comparison import _build_stage_dag

    components = [
        {"params": {"P1", "X"}, "target_filenames": ["A.yaml"]},
        {"params": {"P1", "P2", "Y"}, "target_filenames": ["B.yaml"]},
        {"params": {"P2", "Z"}, "target_filenames": ["C.yaml"]},
    ]
    cuts = [
        CascadeCut(parameter="P1", upstream=["A"]),
        CascadeCut(parameter="P2", upstream=["B"]),
    ]
    lightweight = [
        {"target_id": "A", "qsp_params": {"P1", "X"}, "filename": "A.yaml"},
        {"target_id": "B", "qsp_params": {"P1", "P2", "Y"}, "filename": "B.yaml"},
        {"target_id": "C", "qsp_params": {"P2", "Z"}, "filename": "C.yaml"},
    ]

    stages, edges = _build_stage_dag(components, cuts, lightweight)

    assert len(stages) == 3
    assert 0 in stages[0]  # A
    assert 1 in stages[1]  # B
    assert 2 in stages[2]  # C


def test_build_stage_dag_no_cuts():
    """No cascade cuts → all components in one stage."""
    from qsp_inference.submodel.comparison import _build_stage_dag

    components = [
        {"params": {"X"}, "target_filenames": ["A.yaml"]},
        {"params": {"Y"}, "target_filenames": ["B.yaml"]},
    ]

    stages, edges = _build_stage_dag(components, [], [])

    assert len(stages) == 1
    assert set(stages[0]) == {0, 1}
    assert edges == {}


def test_build_stage_dag_cycle_detection():
    """Circular cascade cuts should raise ValueError."""
    from qsp_inference.submodel.comparison import _build_stage_dag

    # A depends on B via P1, B depends on A via P2 → cycle
    components = [
        {"params": {"P1", "P2", "X"}, "target_filenames": ["A.yaml"]},
        {"params": {"P1", "P2", "Y"}, "target_filenames": ["B.yaml"]},
    ]
    cuts = [
        CascadeCut(parameter="P1", upstream=["A"]),
        CascadeCut(parameter="P2", upstream=["B"]),
    ]
    lightweight = [
        {"target_id": "A", "qsp_params": {"P1", "P2", "X"}, "filename": "A.yaml"},
        {"target_id": "B", "qsp_params": {"P1", "P2", "Y"}, "filename": "B.yaml"},
    ]

    with pytest.raises(ValueError, match="[Cc]ycle"):
        _build_stage_dag(components, cuts, lightweight)


def test_build_stage_dag_missing_upstream_target():
    """Upstream target not found should raise ValueError."""
    from qsp_inference.submodel.comparison import _build_stage_dag

    components = [
        {"params": {"P", "X"}, "target_filenames": ["A.yaml"]},
    ]
    cuts = [CascadeCut(parameter="P", upstream=["nonexistent"])]
    lightweight = [
        {"target_id": "A", "qsp_params": {"P", "X"}, "filename": "A.yaml"},
    ]

    with pytest.raises(ValueError, match="not found"):
        _build_stage_dag(components, cuts, lightweight)


# =============================================================================
# Posterior → PriorSpec conversion
# =============================================================================


def test_posterior_to_prior_spec_lognormal():
    """Lognormal samples should convert to lognormal PriorSpec."""
    from qsp_inference.submodel.comparison import _posterior_to_prior_spec
    from qsp_inference.submodel.inference import PriorSpec

    rng = np.random.default_rng(42)
    true_mu, true_sigma = -3.0, 0.5
    samples = rng.lognormal(true_mu, true_sigma, size=5000)

    original = PriorSpec(
        name="test_param", distribution="lognormal", units="1/day", mu=-4.0, sigma=1.0
    )
    result = _posterior_to_prior_spec(samples, "test_param", original)

    assert result.distribution == "lognormal"
    assert result.name == "test_param"
    assert result.units == "1/day"
    assert abs(result.mu - true_mu) < 0.1
    assert abs(result.sigma - true_sigma) < 0.1


def test_posterior_to_prior_spec_floor():
    """Very tight samples should still have sigma >= 0.01."""
    from qsp_inference.submodel.comparison import _posterior_to_prior_spec
    from qsp_inference.submodel.inference import PriorSpec

    # Nearly constant samples
    samples = np.full(1000, 0.05) + np.random.default_rng(0).normal(0, 1e-10, 1000)
    samples = np.abs(samples)  # ensure positive

    original = PriorSpec(name="test", distribution="lognormal", units="nM", mu=0, sigma=1.0)
    result = _posterior_to_prior_spec(samples, "test", original)

    assert result.sigma >= 0.01


# =============================================================================
# Orphaned cache cleanup
# =============================================================================


def test_cleanup_orphaned_caches(tmp_path):
    """Orphaned cache files (from old component structure) should be deleted."""
    import json

    from qsp_inference.submodel.comparison import (
        _cleanup_orphaned_caches,
        _compute_hash,
    )

    cache_dir = tmp_path / ".compare_cache"
    cache_dir.mkdir()

    # Active components: {A, B} and {C}
    active = [
        {"params": {"A", "B"}, "target_filenames": ["t1.yaml"]},
        {"params": {"C"}, "target_filenames": ["t2.yaml"]},
    ]

    # Create cache files: one matching {A, B}, one matching {C},
    # and one orphan matching old mega-component {A, B, C}
    for params in [{"A", "B"}, {"C"}, {"A", "B", "C"}]:
        comp_hash = _compute_hash("\n".join(sorted(params)))
        path = cache_dir / f"comp_{comp_hash}.json"
        path.write_text(json.dumps({"fits": {}, "diag": {}, "samples": {}}))

    assert len(list(cache_dir.glob("comp_*.json"))) == 3

    deleted = _cleanup_orphaned_caches(cache_dir, active)

    assert deleted == 1
    remaining = list(cache_dir.glob("comp_*.json"))
    assert len(remaining) == 2

    # Verify the orphan ({A, B, C}) is gone
    orphan_hash = _compute_hash("\n".join(sorted({"A", "B", "C"})))
    assert not (cache_dir / f"comp_{orphan_hash}.json").exists()

    # Verify active ones survive
    for comp in active:
        h = _compute_hash("\n".join(sorted(comp["params"])))
        assert (cache_dir / f"comp_{h}.json").exists()
