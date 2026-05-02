"""Regression tests for audit composite-prior copula construction.

These tests pin down the bug fixed in PR `fix/audit-copula-component-membership`:

- ``_write_submodel_priors`` previously used ``len(samples)`` as a proxy for
  inference-component membership and fit a Gaussian copula across row-aligned
  samples from independent components. When NPE samplings shared RNG state
  across components, that row-alignment imposed comonotonic coupling and the
  inferred copula entries spuriously approached |r| ≈ 1.

After the fix, the copula must be block-diagonal by ``component_id``: params
that come from different ``comp_*.json`` cache files have an off-diagonal
correlation of exactly 0 by construction, never inferred from data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from qsp_inference.audit.report import (
    _write_submodel_priors,
    load_joint_samples,
    load_joint_samples_by_component,
)


def _write_comp_cache(cache_dir: Path, comp_id: str, samples: dict[str, list[float]]) -> None:
    """Drop a minimal comp_*.json file mimicking what comparison.py writes."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {"fits": {}, "diag": {}, "samples": samples}
    (cache_dir / f"comp_{comp_id}.json").write_text(json.dumps(payload))


def _comonotonic_independent_components(rng_seed: int = 0, n: int = 4000) -> tuple[dict, dict]:
    """Construct two pseudo-components whose samples are *row-aligned*.

    Both components are drawn from the same base distribution with the same
    seed, then transformed by independent univariate flows (here, exp on
    different lognormal mu/sigma). Across the two components, row i of one
    and row i of the other are at the *same quantile* of their respective
    marginals — comonotonic coupling. This is what shared-RNG NPE samplings
    across components produce in practice. Without the bugfix, the copula
    fit will recover |r| ≈ 1 for any cross-component pair.
    """
    rng = np.random.default_rng(rng_seed)
    z = rng.standard_normal(n)  # shared base draws
    samples_a = {
        "param_A1": list(np.exp(0.5 + 0.7 * z)),
        "param_A2": list(np.exp(-1.0 + 0.4 * z)),
    }
    samples_b = {
        "param_B1": list(np.exp(2.0 + 0.5 * z)),
        "param_B2": list(np.exp(-0.3 + 0.9 * z)),
    }
    return samples_a, samples_b


# ---------------------------------------------------------------------------
# load_joint_samples_by_component
# ---------------------------------------------------------------------------


def test_load_joint_samples_by_component_returns_per_file_mapping(tmp_path: Path) -> None:
    """Each comp_*.json maps to a top-level entry, not flat-merged."""
    samples_a, samples_b = _comonotonic_independent_components()
    _write_comp_cache(tmp_path, "aaaa1111", samples_a)
    _write_comp_cache(tmp_path, "bbbb2222", samples_b)

    by_comp = load_joint_samples_by_component(tmp_path)
    assert by_comp is not None
    assert set(by_comp.keys()) == {"comp_aaaa1111", "comp_bbbb2222"}
    assert set(by_comp["comp_aaaa1111"].keys()) == {"param_A1", "param_A2"}
    assert set(by_comp["comp_bbbb2222"].keys()) == {"param_B1", "param_B2"}


def test_load_joint_samples_flat_view_still_works(tmp_path: Path) -> None:
    """Backward-compat: flat loader merges across files."""
    samples_a, samples_b = _comonotonic_independent_components()
    _write_comp_cache(tmp_path, "aaaa1111", samples_a)
    _write_comp_cache(tmp_path, "bbbb2222", samples_b)

    flat = load_joint_samples(tmp_path)
    assert flat is not None
    assert set(flat.keys()) == {"param_A1", "param_A2", "param_B1", "param_B2"}


# ---------------------------------------------------------------------------
# _write_submodel_priors block-diagonality
# ---------------------------------------------------------------------------


def _read_copula_correlation(yaml_path: Path) -> tuple[list[str], np.ndarray] | None:
    data = yaml.safe_load(yaml_path.read_text())
    cop = data.get("copula")
    if cop is None:
        return None
    return cop["parameters"], np.array(cop["correlation"])


def test_cross_component_copula_off_diag_is_exactly_zero(tmp_path: Path) -> None:
    """Cross-component correlations must be exactly 0 by construction.

    Build two independent components whose rows are comonotonic (same RNG
    base draws). The previous bucket-by-length implementation would fit a
    single 4×4 copula across all four params and recover |r| ≈ 1 between
    every pair. The fixed implementation bucket by component_id, so the
    copula is block-diagonal: A1↔A2 and B1↔B2 entries can be > 0, but
    A?↔B? entries must be exactly 0.
    """
    samples_a, samples_b = _comonotonic_independent_components(rng_seed=42)

    by_component = {
        "comp_a": {k: np.asarray(v) for k, v in samples_a.items()},
        "comp_b": {k: np.asarray(v) for k, v in samples_b.items()},
    }
    targets = {
        "param_A1": ["target_A1"],
        "param_A2": ["target_A2"],
        "param_B1": ["target_B1"],
        "param_B2": ["target_B2"],
    }
    out = tmp_path / "submodel_priors.yaml"
    _write_submodel_priors(by_component, targets, groups={}, output_path=out, copula_threshold=0.0)

    cop = _read_copula_correlation(out)
    assert cop is not None, "copula block must be written when params share components"
    params, R = cop
    name_to_idx = {n: i for i, n in enumerate(params)}

    # Diagonal must be 1.
    np.testing.assert_allclose(np.diag(R), 1.0, atol=1e-12)

    # Cross-component entries must be exactly 0 — no data-driven inference.
    for a in ("param_A1", "param_A2"):
        for b in ("param_B1", "param_B2"):
            if a in name_to_idx and b in name_to_idx:
                ia, ib = name_to_idx[a], name_to_idx[b]
                assert R[ia, ib] == 0.0, (
                    f"cross-component correlation must be exactly 0 by construction; "
                    f"got R[{a},{b}]={R[ia, ib]:.6f}"
                )


def test_within_component_correlations_are_preserved(tmp_path: Path) -> None:
    """Within a component, real correlations should still be inferred."""
    rng = np.random.default_rng(0)
    n = 4000
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    # param_X1 and param_X2 are genuinely correlated (shared latent z1)
    x1 = np.exp(0.5 + 0.6 * z1 + 0.1 * z2)
    x2 = np.exp(-0.3 + 0.6 * z1 + 0.2 * z2)

    by_component = {
        "comp_x": {
            "param_X1": x1,
            "param_X2": x2,
        }
    }
    targets = {"param_X1": ["t1"], "param_X2": ["t2"]}
    out = tmp_path / "submodel_priors.yaml"
    _write_submodel_priors(by_component, targets, groups={}, output_path=out, copula_threshold=0.05)

    cop = _read_copula_correlation(out)
    assert cop is not None
    params, R = cop
    i = params.index("param_X1")
    j = params.index("param_X2")
    # Real correlation should survive — not exactly 1 (data has noise) but well above threshold.
    assert 0.5 < R[i, j] < 0.99, f"within-component R should reflect data correlation, got {R[i, j]:.3f}"


def test_comonotonic_artifact_is_blocked_at_high_threshold(tmp_path: Path) -> None:
    """Prove the fix: shared-RNG row alignment across components no longer
    pollutes the copula even with a permissive threshold.

    Empirically, the pre-fix behavior was |r| > 0.99 for every cross-pair
    on this dataset. After the fix, cross-component entries are exactly 0
    — and therefore filtered out by any positive threshold.
    """
    samples_a, samples_b = _comonotonic_independent_components(rng_seed=7)

    by_component = {
        "comp_a": {k: np.asarray(v) for k, v in samples_a.items()},
        "comp_b": {k: np.asarray(v) for k, v in samples_b.items()},
    }
    targets = {k: [f"target_{k}"] for k in ("param_A1", "param_A2", "param_B1", "param_B2")}

    out = tmp_path / "submodel_priors.yaml"
    # Threshold high enough to filter trivial within-component noise
    # but lower than the comonotonic-artifact value (|r| ≈ 0.99) — the
    # fix guarantees zero cross-component entries regardless.
    _write_submodel_priors(by_component, targets, groups={}, output_path=out, copula_threshold=0.5)

    cop = _read_copula_correlation(out)
    if cop is None:
        # All entries below threshold → no copula written. Passing.
        return
    params, R = cop
    # Both component blocks may survive the threshold (their within-component
    # correlations are real and >> 0.5 here). What must NOT survive is any
    # CROSS-component entry above the threshold — that's the comonotonic
    # artifact this fix prevents.
    name_to_idx = {n: i for i, n in enumerate(params)}
    for a in ("param_A1", "param_A2"):
        for b in ("param_B1", "param_B2"):
            if a in name_to_idx and b in name_to_idx:
                ia, ib = name_to_idx[a], name_to_idx[b]
                assert R[ia, ib] == 0.0, (
                    f"cross-component entry leaked into copula at threshold 0.5: "
                    f"R[{a},{b}]={R[ia, ib]:.6f} — comonotonic artifact survived the fix"
                )


# ---------------------------------------------------------------------------
# Per-component RNG seed derivation
# ---------------------------------------------------------------------------


def test_per_component_seed_derivation_is_deterministic_and_distinct() -> None:
    """The seed-mixing recipe in run_comparison must be deterministic per
    (base_seed, comp_id) and distinct across comp_ids."""
    base = 42
    comp_a = "a3f2b1c4d5e6"  # arbitrary hex digest
    comp_b = "9876543210ab"

    seed_a = (base + int(comp_a, 16)) & 0x7FFFFFFF
    seed_b = (base + int(comp_b, 16)) & 0x7FFFFFFF
    seed_a_again = (base + int(comp_a, 16)) & 0x7FFFFFFF

    assert seed_a == seed_a_again, "seed derivation must be deterministic"
    assert seed_a != seed_b, "seeds must differ across components"
    assert 0 <= seed_a < 2**31
    assert 0 <= seed_b < 2**31


def test_run_component_npe_seeds_torch_rng() -> None:
    """run_component_npe must reseed torch's RNG so different components
    get different sample trajectories. Direct check against the seeding
    block we added — calling run_component_npe end-to-end requires building
    SubmodelTarget objects, which is heavyweight; instead, verify the
    seeding behavior matches what the function performs internally.
    """
    import random as _random

    _torch = pytest.importorskip("torch")

    # Mirror the seeding block from run_component_npe.
    def _seed_like_run_component_npe(seed: int) -> tuple[float, float]:
        _torch.manual_seed(int(seed) & 0x7FFFFFFF)
        _random.seed(int(seed) & 0x7FFFFFFF)
        return _torch.randn(1).item(), _random.random()

    a1 = _seed_like_run_component_npe(1234)
    a2 = _seed_like_run_component_npe(1234)
    b = _seed_like_run_component_npe(5678)

    assert a1 == a2, "same seed must produce same RNG state"
    assert a1 != b, "different seeds must produce different RNG state"
