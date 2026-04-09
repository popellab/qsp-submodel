"""Tests for qsp_inference.submodel.parameter_groups."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from qsp_inference.submodel.parameter_groups import (
    CascadeCut,
    DeltaPrior,
    GroupMember,
    GroupPrior,
    ParameterGroup,
    ParameterGroupsConfig,
    load_parameter_groups,
)

try:
    import jax  # noqa: F401
    import numpyro  # noqa: F401

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


# =============================================================================
# Schema validation tests
# =============================================================================


def test_valid_group():
    """Minimal valid parameter group."""
    group = ParameterGroup(
        group_id="test_death_rates",
        description="Test group",
        base_prior=GroupPrior(distribution="lognormal", mu=-4.1, sigma=0.5),
        between_member_sd=GroupPrior(distribution="half_normal", sigma=0.4),
        members=[
            GroupMember(name="k_a_death", units="1/day"),
            GroupMember(name="k_b_death", units="1/day"),
        ],
    )
    assert group.member_names == {"k_a_death", "k_b_death"}


def test_group_with_delta_priors():
    """Group with informative delta priors on members."""
    group = ParameterGroup(
        group_id="test",
        base_prior=GroupPrior(distribution="lognormal", mu=-4.1, sigma=0.5),
        between_member_sd=GroupPrior(distribution="half_normal", sigma=0.4),
        members=[
            GroupMember(
                name="k_a",
                units="1/day",
                delta_prior=DeltaPrior(mu=-0.5, sigma=0.3),
            ),
            GroupMember(name="k_b", units="1/day"),
        ],
    )
    assert group.members[0].delta_prior is not None
    assert group.members[0].delta_prior.mu == -0.5
    assert group.members[1].delta_prior is None


def test_rejects_single_member():
    """Groups must have at least 2 members."""
    with pytest.raises(Exception):
        ParameterGroup(
            group_id="test",
            base_prior=GroupPrior(distribution="lognormal", mu=0, sigma=1),
            between_member_sd=GroupPrior(distribution="half_normal", sigma=0.3),
            members=[GroupMember(name="k_a", units="1/day")],
        )


def test_rejects_duplicate_members():
    """No duplicate member names within a group."""
    with pytest.raises(Exception, match="Duplicate"):
        ParameterGroup(
            group_id="test",
            base_prior=GroupPrior(distribution="lognormal", mu=0, sigma=1),
            between_member_sd=GroupPrior(distribution="half_normal", sigma=0.3),
            members=[
                GroupMember(name="k_a", units="1/day"),
                GroupMember(name="k_a", units="1/day"),
            ],
        )


def test_rejects_invalid_base_prior_distribution():
    """base_prior must be lognormal or normal."""
    with pytest.raises(Exception, match="base_prior"):
        ParameterGroup(
            group_id="test",
            base_prior=GroupPrior(distribution="half_normal", mu=0, sigma=1),
            between_member_sd=GroupPrior(distribution="half_normal", sigma=0.3),
            members=[
                GroupMember(name="k_a", units="1/day"),
                GroupMember(name="k_b", units="1/day"),
            ],
        )


def test_rejects_invalid_tau_distribution():
    """between_member_sd must be half_normal."""
    with pytest.raises(Exception, match="between_member_sd"):
        ParameterGroup(
            group_id="test",
            base_prior=GroupPrior(distribution="lognormal", mu=0, sigma=1),
            between_member_sd=GroupPrior(distribution="lognormal", mu=0, sigma=0.3),
            members=[
                GroupMember(name="k_a", units="1/day"),
                GroupMember(name="k_b", units="1/day"),
            ],
        )


def test_config_rejects_overlapping_groups():
    """Same parameter cannot appear in multiple groups."""
    with pytest.raises(Exception, match="appears in both"):
        ParameterGroupsConfig(
            groups=[
                ParameterGroup(
                    group_id="group1",
                    base_prior=GroupPrior(distribution="lognormal", mu=0, sigma=1),
                    between_member_sd=GroupPrior(distribution="half_normal", sigma=0.3),
                    members=[
                        GroupMember(name="k_shared", units="1/day"),
                        GroupMember(name="k_a", units="1/day"),
                    ],
                ),
                ParameterGroup(
                    group_id="group2",
                    base_prior=GroupPrior(distribution="lognormal", mu=0, sigma=1),
                    between_member_sd=GroupPrior(distribution="half_normal", sigma=0.3),
                    members=[
                        GroupMember(name="k_shared", units="1/day"),
                        GroupMember(name="k_b", units="1/day"),
                    ],
                ),
            ]
        )


def test_config_all_grouped_params():
    """all_grouped_params aggregates across groups."""
    config = ParameterGroupsConfig(
        groups=[
            ParameterGroup(
                group_id="g1",
                base_prior=GroupPrior(distribution="lognormal", mu=0, sigma=1),
                between_member_sd=GroupPrior(distribution="half_normal", sigma=0.3),
                members=[
                    GroupMember(name="a", units="1/day"),
                    GroupMember(name="b", units="1/day"),
                ],
            ),
            ParameterGroup(
                group_id="g2",
                base_prior=GroupPrior(distribution="lognormal", mu=0, sigma=1),
                between_member_sd=GroupPrior(distribution="half_normal", sigma=0.3),
                members=[
                    GroupMember(name="c", units="1/day"),
                    GroupMember(name="d", units="1/day"),
                ],
            ),
        ]
    )
    assert config.all_grouped_params == {"a", "b", "c", "d"}
    assert config.get_group_for_param("a").group_id == "g1"
    assert config.get_group_for_param("c").group_id == "g2"
    assert config.get_group_for_param("z") is None


# =============================================================================
# Cascade cut schema tests
# =============================================================================


def test_cascade_cut_schema_valid():
    """Minimal valid cascade cut config."""
    config = ParameterGroupsConfig(
        cascade_cuts=[
            CascadeCut(
                parameter="IL1_50",
                upstream=["target_a", "target_b"],
                reason="Test reason",
            )
        ]
    )
    assert config.cascade_cut_params == {"IL1_50"}
    assert config.get_upstream_targets("IL1_50") == ["target_a", "target_b"]
    assert config.get_upstream_targets("nonexistent") == []


def test_cascade_cut_rejects_duplicate_param():
    """Same parameter in two cascade cuts should fail."""
    with pytest.raises(Exception, match="multiple cascade_cuts"):
        ParameterGroupsConfig(
            cascade_cuts=[
                CascadeCut(parameter="IL1_50", upstream=["t1"]),
                CascadeCut(parameter="IL1_50", upstream=["t2"]),
            ]
        )


def test_cascade_cut_rejects_group_overlap():
    """Cascade cut parameter that is also a group member should fail."""
    with pytest.raises(Exception, match="cascade_cut and a member"):
        ParameterGroupsConfig(
            groups=[
                ParameterGroup(
                    group_id="g1",
                    base_prior=GroupPrior(distribution="lognormal", mu=0, sigma=1),
                    between_member_sd=GroupPrior(distribution="half_normal", sigma=0.3),
                    members=[
                        GroupMember(name="IL1_50", units="nM"),
                        GroupMember(name="IL1_50_other", units="nM"),
                    ],
                )
            ],
            cascade_cuts=[
                CascadeCut(parameter="IL1_50", upstream=["target_a"]),
            ],
        )


def test_cascade_cut_empty_upstream_rejected():
    """Upstream must have at least 1 entry."""
    with pytest.raises(Exception):
        CascadeCut(parameter="IL1_50", upstream=[])


def test_cascade_cut_params_property():
    """cascade_cut_params returns the correct set."""
    config = ParameterGroupsConfig(
        cascade_cuts=[
            CascadeCut(parameter="a", upstream=["t1"]),
            CascadeCut(parameter="b", upstream=["t2"]),
        ]
    )
    assert config.cascade_cut_params == {"a", "b"}


def test_cascade_cut_yaml_roundtrip():
    """Round-trip: write YAML with cascade_cuts, load it, validate."""
    data = {
        "groups": [],
        "cascade_cuts": [
            {
                "parameter": "IL1_50",
                "upstream": ["target_a", "target_b"],
                "reason": "Test reason",
            },
            {
                "parameter": "k_iCAF_to_myCAF",
                "upstream": ["target_c"],
            },
        ],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        f.flush()
        config = load_parameter_groups(Path(f.name))

    assert len(config.cascade_cuts) == 2
    assert config.cascade_cut_params == {"IL1_50", "k_iCAF_to_myCAF"}
    assert config.cascade_cuts[0].upstream == ["target_a", "target_b"]


# =============================================================================
# YAML loading tests
# =============================================================================


def test_load_parameter_groups_from_yaml():
    """Round-trip: write YAML, load it, validate."""
    data = {
        "groups": [
            {
                "group_id": "CAF_death_rates",
                "description": "CAF subtype death rates",
                "between_member_sd": {"distribution": "half_normal", "sigma": 0.4},
                "members": [
                    {"name": "k_qpsc_death", "units": "1/day"},
                    {
                        "name": "k_iCAF_death",
                        "units": "1/day",
                        "delta_prior": {"mu": 0.0, "sigma": 0.3},
                    },
                    {"name": "k_myCAF_death", "units": "1/day"},
                    {"name": "k_apCAF_death", "units": "1/day"},
                ],
            }
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        f.flush()
        config = load_parameter_groups(Path(f.name))

    assert len(config.groups) == 1
    assert config.groups[0].group_id == "CAF_death_rates"
    assert config.groups[0].base_prior is None  # derived from CSV at runtime
    assert len(config.groups[0].members) == 4
    assert config.all_grouped_params == {
        "k_qpsc_death",
        "k_iCAF_death",
        "k_myCAF_death",
        "k_apCAF_death",
    }


def test_resolve_base_prior_from_csv():
    """base_prior=None derives from CSV priors of members."""
    from qsp_inference.submodel.inference import PriorSpec

    group = ParameterGroup(
        group_id="test",
        between_member_sd=GroupPrior(distribution="half_normal", sigma=0.3),
        members=[
            GroupMember(name="k_a", units="1/day"),
            GroupMember(name="k_b", units="1/day"),
        ],
    )
    assert group.base_prior is None

    prior_specs = {
        "k_a": PriorSpec(name="k_a", distribution="lognormal", units="1/day", mu=-4.0, sigma=0.5),
        "k_b": PriorSpec(name="k_b", distribution="lognormal", units="1/day", mu=-4.2, sigma=0.6),
    }
    bp = group.resolve_base_prior(prior_specs)
    assert bp.distribution == "lognormal"
    assert bp.mu == pytest.approx(-4.1)  # mean of -4.0 and -4.2
    assert bp.sigma == 0.6  # max of 0.5 and 0.6


def test_resolve_base_prior_explicit():
    """Explicit base_prior is returned as-is."""
    group = ParameterGroup(
        group_id="test",
        base_prior=GroupPrior(distribution="lognormal", mu=-3.0, sigma=0.8),
        between_member_sd=GroupPrior(distribution="half_normal", sigma=0.3),
        members=[
            GroupMember(name="k_a", units="1/day"),
            GroupMember(name="k_b", units="1/day"),
        ],
    )
    bp = group.resolve_base_prior({})  # doesn't need CSV specs
    assert bp.mu == -3.0
    assert bp.sigma == 0.8


def test_load_missing_file_returns_empty():
    """Missing file returns empty config (no error)."""
    config = load_parameter_groups(Path("/nonexistent/submodel_config.yaml"))
    assert config.groups == []
    assert config.all_grouped_params == set()


def test_load_empty_yaml_returns_empty():
    """Empty YAML returns empty config."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("")
        f.flush()
        config = load_parameter_groups(Path(f.name))

    assert config.groups == []


# =============================================================================
# Integration: hierarchical sampling in NumPyro
# =============================================================================


@pytest.mark.skipif(not HAS_JAX, reason="JAX/NumPyro not installed")
def test_hierarchical_sampling_produces_grouped_params():
    """Verify that hierarchical groups produce correlated parameter samples."""
    from unittest.mock import patch

    from qsp_inference.submodel.inference import PriorSpec, run_joint_inference
    from maple.core.calibration.submodel_target import SubmodelTarget

    # Two grouped params, one with a submodel target anchoring it
    # base_prior omitted — derived from CSV priors at runtime
    groups = ParameterGroupsConfig(
        groups=[
            ParameterGroup(
                group_id="test_rates",
                between_member_sd=GroupPrior(distribution="half_normal", sigma=0.3),
                members=[
                    GroupMember(name="k_anchored", units="1/day"),
                    GroupMember(name="k_unanchored", units="1/day"),
                ],
            )
        ]
    )

    # CSV priors for both (needed for validation)
    prior_specs = {
        "k_anchored": PriorSpec(
            name="k_anchored", distribution="lognormal", units="1/day", mu=-4.0, sigma=0.5
        ),
        "k_unanchored": PriorSpec(
            name="k_unanchored", distribution="lognormal", units="1/day", mu=-4.0, sigma=0.5
        ),
    }

    # Simple algebraic target anchoring k_anchored at ~0.02
    target_dict = {
        "target_id": "test_anchor",
        "study_interpretation": "Test",
        "key_assumptions": ["Test"],
        "experimental_context": {"species": "human", "system": "in_vitro"},
        "primary_data_source": {
            "doi": "10.1234/test",
            "source_tag": "Test2024",
            "title": "Test",
            "source_relevance": {
                "indication_match": "exact",
                "indication_match_justification": "Test justification for source relevance with exact indication match for this test.",
                "species_source": "human",
                "species_target": "human",
                "source_quality": "primary_human_in_vitro",
                "perturbation_type": "physiological_baseline",
                "perturbation_relevance": "Baseline measurement under physiological conditions directly applicable.",
                "tme_compatibility": "high",
                "tme_compatibility_notes": "In vitro system closely matches target biology for this test.",
                "measurement_directness": "direct",
                "temporal_resolution": "endpoint_pair",
                "experimental_system": "in_vitro_primary",
            },
        },
        "secondary_data_sources": [],
        "inputs": [
            {
                "name": "obs_mean",
                "value": 0.02,
                "units": "1/day",
                "input_type": "direct_measurement",
                "source_ref": "Test2024",
                "source_location": "Table 1",
                "value_snippet": "rate of 0.02 per day",
            },
            {
                "name": "obs_sd",
                "value": 0.3,
                "units": "dimensionless",
                "input_type": "direct_measurement",
                "source_ref": "Test2024",
                "source_location": "Table 1",
                "value_snippet": "CV of 0.3",
            },
            {
                "name": "n_samples",
                "value": 10,
                "units": "dimensionless",
                "input_type": "direct_measurement",
                "source_ref": "Test2024",
                "source_location": "Methods",
                "value_snippet": "n = 10",
            },
        ],
        "calibration": {
            "parameters": [{"name": "k_anchored", "units": "1/day"}],
            "forward_model": {
                "type": "algebraic",
                "formula": "k = obs",
                "code": "def compute(params, inputs):\n    return params['k_anchored']\n",
                "data_rationale": "Direct measurement",
                "submodel_rationale": "Maps to k_anchored",
            },
            "error_model": [
                {
                    "name": "obs",
                    "units": "1/day",
                    "uses_inputs": ["obs_mean", "obs_sd"],
                    "sample_size_input": "n_samples",
                    "observation_code": (
                        "def derive_observation(inputs, sample_size, rng, n_bootstrap):\n"
                        "    import numpy as np\n"
                        "    mean = inputs['obs_mean']\n"
                        "    cv = inputs['obs_sd']\n"
                        "    return rng.lognormal(np.log(mean), cv / np.sqrt(sample_size), n_bootstrap)\n"
                    ),
                }
            ],
            "identifiability_notes": "Direct observation",
        },
    }

    with patch(
        "maple.core.calibration.validators.resolve_doi",
        return_value={"title": "Test", "year": 2024, "first_author": "Test"},
    ):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            target = SubmodelTarget(**target_dict)

    samples, diagnostics = run_joint_inference(
        prior_specs,
        [target],
        parameter_groups=groups,
        num_warmup=100,
        num_samples=200,
        num_chains=1,
        seed=42,
    )

    # Both params should be in samples (k_anchored as deterministic, k_unanchored as deterministic)
    assert "k_anchored" in samples
    assert "k_unanchored" in samples

    # Hyperparameters should also be present
    assert "test_rates__base" in samples
    # tau is fixed (not sampled) for small groups (<=2 constrained members)
    # z-scores from non-centered parameterization should be present
    assert "k_anchored__z" in samples
    assert "k_unanchored__z" in samples

    # k_anchored should be pulled toward 0.02 by the target
    median_anchored = np.median(samples["k_anchored"])
    assert 0.005 < median_anchored < 0.08, f"Expected near 0.02, got {median_anchored}"

    # k_unanchored should be pulled toward k_anchored by partial pooling
    # (closer to 0.02 than the prior median of exp(-4) ≈ 0.018)
    # Both should be correlated
    corr = np.corrcoef(samples["k_anchored"], samples["k_unanchored"])[0, 1]
    assert corr > 0.1, f"Expected positive correlation from shared base, got {corr}"


@pytest.mark.skipif(not HAS_JAX, reason="JAX/NumPyro not installed")
def test_non_grouped_params_unaffected():
    """Non-grouped params should sample independently, same as before."""

    from qsp_inference.submodel.inference import PriorSpec, run_joint_inference

    groups = ParameterGroupsConfig(
        groups=[
            ParameterGroup(
                group_id="g",
                between_member_sd=GroupPrior(distribution="half_normal", sigma=0.3),
                members=[
                    GroupMember(name="k_a", units="1/day"),
                    GroupMember(name="k_b", units="1/day"),
                ],
            )
        ]
    )

    # k_independent is NOT in any group
    prior_specs = {
        "k_a": PriorSpec(name="k_a", distribution="lognormal", units="1/day", mu=0, sigma=1),
        "k_b": PriorSpec(name="k_b", distribution="lognormal", units="1/day", mu=0, sigma=1),
        "k_independent": PriorSpec(
            name="k_independent", distribution="lognormal", units="1/day", mu=-2, sigma=0.5
        ),
    }

    # No targets — pure prior sampling
    samples, diagnostics = run_joint_inference(
        prior_specs,
        [],
        parameter_groups=groups,
        num_warmup=50,
        num_samples=100,
        num_chains=1,
        seed=0,
    )

    assert "k_independent" in samples
    assert "k_a" in samples
    assert "k_b" in samples

    # k_independent should be uncorrelated with grouped params
    corr_ai = np.corrcoef(samples["k_a"], samples["k_independent"])[0, 1]
    # Allow some sampling noise but shouldn't be highly correlated
    assert abs(corr_ai) < 0.5, f"Expected low correlation, got {corr_ai}"
