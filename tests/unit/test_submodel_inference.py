"""Tests for qsp_inference.submodel.inference."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from qsp_inference.submodel.inference import (
    ErrorModelEntry,
    PriorSpec,
    TargetLikelihood,
    _run_bootstrap,
    build_target_likelihoods,
    load_priors_from_csv,
)
from qsp_inference.submodel.prior import DistFit

try:
    import jax  # noqa: F401
    import numpyro  # noqa: F401

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


# =============================================================================
# Fixtures
# =============================================================================

SIMPLE_CSV = """\
name,median,units,distribution,dist_param1,dist_param2
k_prolif,1.0,1/day,lognormal,0.0,1.0
k_death,0.5,1/day,lognormal,-0.693,0.5
EC50,10.0,nM,lognormal,2.302,0.8
k_ratio,0.3,dimensionless,uniform,0.01,1.0
"""

SIMPLE_OBSERVATION_CODE = """\
def derive_observation(inputs, sample_size, rng, n_bootstrap):
    import numpy as np
    mean = inputs['obs_mean']
    sd = inputs['obs_sd']
    n = int(sample_size)
    return np.exp(rng.normal(np.log(mean), sd / np.sqrt(n), n_bootstrap))
"""


def _make_mock_entry(observation_code=None, name="test", uses_inputs=None):
    """Create a minimal mock ErrorModel entry for bootstrap testing."""
    from unittest.mock import MagicMock

    entry = MagicMock()
    entry.name = name
    entry.observation_code = observation_code or SIMPLE_OBSERVATION_CODE
    entry.sample_size_input = "n_samples"
    entry.n_bootstrap = 10000
    entry.uses_inputs = uses_inputs or ["obs_mean", "obs_sd"]
    entry.x_input = None
    entry.evaluation_points = None
    return entry


DEFAULT_SOURCE_RELEVANCE = {
    "indication_match": "exact",
    "indication_match_justification": "Test justification for source relevance with exact indication match.",
    "species_source": "human",
    "species_target": "human",
    "source_quality": "primary_human_in_vitro",
    "perturbation_type": "physiological_baseline",
    "perturbation_relevance": "Baseline measurement under physiological conditions, directly applicable to model parameter.",
    "tme_compatibility": "high",
    "tme_compatibility_notes": "In vitro system closely recapitulates the target biology for this parameter.",
    "measurement_directness": "direct",
    "temporal_resolution": "endpoint_pair",
    "experimental_system": "in_vitro_primary",
}


def _make_algebraic_target(
    target_id="test_target",
    param_name="k_test",
    formula="k = obs",
    code=None,
):
    """Build a minimal SubmodelTarget dict for algebraic models."""
    if code is None:
        code = "def compute(params, inputs):\n" f"    return params['{param_name}']\n"

    return {
        "target_id": target_id,
        "study_interpretation": "Test interpretation of the study data",
        "key_assumptions": ["Test assumption"],
        "experimental_context": {"species": "human", "system": "in_vitro"},
        "primary_data_source": {
            "doi": "10.1234/test",
            "source_tag": "Test2024",
            "title": "Test paper",
            "source_relevance": DEFAULT_SOURCE_RELEVANCE,
        },
        "secondary_data_sources": [],
        "inputs": [
            {
                "name": "obs_mean",
                "value": 5.0,
                "units": "1/day",
                "input_type": "direct_measurement",
                "source_ref": "Test2024",
                "source_location": "Table 1",
                "value_snippet": "mean value of 5.0 per day",
            },
            {
                "name": "obs_sd",
                "value": 0.3,
                "units": "dimensionless",
                "input_type": "direct_measurement",
                "source_ref": "Test2024",
                "source_location": "Table 1",
                "value_snippet": "SD of 0.3",
            },
            {
                "name": "n_samples",
                "value": 10,
                "units": "dimensionless",
                "input_type": "direct_measurement",
                "source_ref": "Test2024",
                "source_location": "Table 1",
                "value_snippet": "n = 10 patients",
            },
        ],
        "calibration": {
            "parameters": [{"name": param_name, "units": "1/day"}],
            "forward_model": {
                "type": "algebraic",
                "formula": formula,
                "code": code,
                "data_rationale": "Test data rationale",
                "submodel_rationale": "Test submodel rationale",
            },
            "error_model": [
                {
                    "name": "obs",
                    "units": "1/day",
                    "uses_inputs": ["obs_mean", "obs_sd"],
                    "sample_size_input": "n_samples",
                    "observation_code": SIMPLE_OBSERVATION_CODE,
                }
            ],
            "identifiability_notes": "Single parameter, directly observed.",
        },
    }


# =============================================================================
# Mock DOI resolution
# =============================================================================


@pytest.fixture(autouse=True)
def mock_doi_resolution():
    """Mock DOI resolution to avoid network calls."""

    def mock_resolve(doi):
        return {"title": "Test paper", "year": 2024, "first_author": "Test"}

    with patch(
        "maple.core.calibration.validators.resolve_doi",
        side_effect=mock_resolve,
    ):
        yield


# =============================================================================
# Tests: CSV loader
# =============================================================================


class TestLoadPriorsFromCSV:
    def test_load_basic_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(SIMPLE_CSV)
            f.flush()
            specs = load_priors_from_csv(Path(f.name))

        assert len(specs) == 4
        assert "k_prolif" in specs
        assert "k_death" in specs
        assert "EC50" in specs
        assert "k_ratio" in specs

    def test_lognormal_params(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(SIMPLE_CSV)
            f.flush()
            specs = load_priors_from_csv(Path(f.name))

        spec = specs["k_prolif"]
        assert spec.distribution == "lognormal"
        assert spec.mu == pytest.approx(0.0)
        assert spec.sigma == pytest.approx(1.0)
        assert spec.units == "1/day"

    def test_uniform_params(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(SIMPLE_CSV)
            f.flush()
            specs = load_priors_from_csv(Path(f.name))

        spec = specs["k_ratio"]
        assert spec.distribution == "uniform"
        assert spec.lower == pytest.approx(0.01)
        assert spec.upper == pytest.approx(1.0)

    def test_unknown_distribution_raises(self):
        csv = "name,median,units,distribution,dist_param1,dist_param2\nk,1,1/day,cauchy,0,1\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv)
            f.flush()
            with pytest.raises(ValueError, match="Unknown distribution"):
                load_priors_from_csv(Path(f.name))


# =============================================================================
# Tests: Bootstrap runner
# =============================================================================


class TestRunBootstrap:
    def test_basic_bootstrap(self):
        entry = _make_mock_entry()
        inputs_dict = {"obs_mean": 5.0, "obs_sd": 0.3, "n_samples": 10}
        fit = _run_bootstrap(entry, inputs_dict)

        assert isinstance(fit, DistFit)
        assert fit.name in ("lognormal", "gamma", "invgamma")
        assert fit.median > 0
        # Median should be close to obs_mean
        assert fit.median == pytest.approx(5.0, rel=0.1)

    def test_bootstrap_failure_raises(self):
        bad_code = (
            "def derive_observation(inputs, sample_size, rng, n_bootstrap):\n"
            "    import numpy as np\n"
            "    return np.array([0.0] * n_bootstrap)\n"
        )
        entry = _make_mock_entry(observation_code=bad_code)
        inputs_dict = {"obs_mean": 5.0, "obs_sd": 0.3, "n_samples": 10}
        with pytest.raises(RuntimeError, match="Could not fit"):
            _run_bootstrap(entry, inputs_dict)


# =============================================================================
# Tests: Forward function builder
# =============================================================================


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
class TestBuildForwardFns:
    def test_algebraic_forward_fn(self):
        import warnings

        from qsp_inference.submodel.inference import _build_forward_fns
        from maple.core.calibration.submodel_target import SubmodelTarget

        data = _make_algebraic_target()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            target = SubmodelTarget(**data)

        fns = _build_forward_fns(target)
        assert len(fns) == 1

        # Identity forward model: compute returns params['k_test']
        result = fns[0]({"k_test": 3.14})
        assert float(result) == pytest.approx(3.14)

    def test_algebraic_forward_fn_log_transform(self):
        import warnings

        from qsp_inference.submodel.inference import _build_forward_fns
        from maple.core.calibration.submodel_target import SubmodelTarget

        code = "def compute(params, inputs):\n" "    return np.log(2) / params['k_test']\n"
        data = _make_algebraic_target(code=code, formula="t_half = ln(2) / k")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            target = SubmodelTarget(**data)

        fns = _build_forward_fns(target)
        result = fns[0]({"k_test": 1.0})
        assert float(result) == pytest.approx(np.log(2), rel=1e-5)

    def test_unknown_type_raises_value_error(self):
        """Unknown model types should raise ValueError."""

        from unittest.mock import MagicMock

        from qsp_inference.submodel.inference import _build_forward_fns

        target = MagicMock()
        target.calibration.forward_model.type = "totally_unknown_type"
        target.calibration.error_model = [MagicMock()]
        target.inputs = []

        with pytest.raises(ValueError, match="Unknown forward model type"):
            _build_forward_fns(target)


# =============================================================================
# Tests: Target likelihood builder
# =============================================================================


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
class TestBuildTargetLikelihoods:
    def test_basic_build(self):
        import warnings

        from maple.core.calibration.submodel_target import SubmodelTarget

        data = _make_algebraic_target(param_name="k_prolif")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            target = SubmodelTarget(**data)

        prior_specs = {
            "k_prolif": PriorSpec(
                name="k_prolif",
                distribution="lognormal",
                units="1/day",
                mu=0.0,
                sigma=1.0,
            )
        }

        likelihoods = build_target_likelihoods([target], prior_specs)
        assert len(likelihoods) == 1

        tl = likelihoods[0]
        assert tl.target_id == "test_target"
        assert len(tl.entries) == 1

        entry = tl.entries[0]
        assert entry.family == "lognormal"
        assert entry.value > 0
        assert entry.sigma > 0
        assert entry.sigma_trans >= 0.15  # floor sigma

    def test_missing_parameter_raises(self):
        import warnings

        from maple.core.calibration.submodel_target import SubmodelTarget

        data = _make_algebraic_target(param_name="k_missing")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            target = SubmodelTarget(**data)

        prior_specs = {
            "k_other": PriorSpec(
                name="k_other",
                distribution="lognormal",
                units="1/day",
                mu=0.0,
                sigma=1.0,
            )
        }

        with pytest.raises(ValueError, match="not found in priors CSV"):
            build_target_likelihoods([target], prior_specs)


# =============================================================================
# Tests: resolve_per_measurement_sigma
# =============================================================================


class TestResolvePerMeasurementSigma:
    """Test per-measurement translation sigma resolution."""

    def _make_target(self, secondary_sources=None):
        """Build a minimal SubmodelTarget with configurable sources."""
        import warnings

        from maple.core.calibration.submodel_target import SubmodelTarget

        primary_sr = {
            **DEFAULT_SOURCE_RELEVANCE,
            "species_source": "human",
        }

        data = _make_algebraic_target(param_name="k_test")
        data["primary_data_source"]["source_relevance"] = primary_sr

        if secondary_sources:
            data["secondary_data_sources"] = secondary_sources
        else:
            data["secondary_data_sources"] = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SubmodelTarget(**data)

    def test_single_source_all_inputs(self):
        """All inputs from primary → sigma equals primary's translation sigma."""
        from qsp_inference.submodel.inference import (
            resolve_per_measurement_sigma,
        )
        from qsp_inference.submodel.prior import compute_translation_sigma

        target = self._make_target()
        primary_sr = target.primary_data_source.source_relevance
        expected_sigma, _ = compute_translation_sigma(primary_sr)

        sigma, breakdown = resolve_per_measurement_sigma(
            target, target.calibration.error_model[0].uses_inputs
        )
        assert abs(sigma - expected_sigma) < 1e-6
        assert len(breakdown) == 1  # one unique source

    def test_same_source_not_double_counted(self):
        """Two inputs from the same source should count once, not twice."""
        from qsp_inference.submodel.inference import (
            resolve_per_measurement_sigma,
        )
        from qsp_inference.submodel.prior import compute_translation_sigma

        target = self._make_target()
        primary_sr = target.primary_data_source.source_relevance
        expected_sigma, _ = compute_translation_sigma(primary_sr)

        # uses_inputs has two inputs, both from same source_ref
        uses = [
            inp.name
            for inp in target.inputs
            if inp.source_ref == target.primary_data_source.source_tag
        ]
        assert len(uses) >= 2  # obs_mean and obs_sd both from Test2024

        sigma, breakdown = resolve_per_measurement_sigma(target, uses)
        assert abs(sigma - expected_sigma) < 1e-6  # NOT sqrt(2) * expected

    def test_two_sources_quadrature(self):
        """Inputs from two different sources combine in quadrature."""
        import warnings

        import numpy as np

        from qsp_inference.submodel.inference import (
            resolve_per_measurement_sigma,
        )
        from maple.core.calibration.submodel_target import SubmodelTarget
        from qsp_inference.submodel.prior import compute_translation_sigma

        mouse_sr = {
            **DEFAULT_SOURCE_RELEVANCE,
            "species_source": "mouse",
        }

        data = _make_algebraic_target(param_name="k_test")
        data["primary_data_source"]["source_relevance"] = DEFAULT_SOURCE_RELEVANCE
        data["secondary_data_sources"] = [
            {
                "doi": "10.5678/mouse",
                "source_tag": "Mouse2024",
                "title": "Test paper",
                "source_relevance": mouse_sr,
            }
        ]
        # Add an input from the secondary source
        data["inputs"].append(
            {
                "name": "mouse_val",
                "value": 5.0,
                "units": "1/day",
                "input_type": "direct_measurement",
                "source_ref": "Mouse2024",
                "source_location": "Table 1",
                "value_snippet": "value of 5.0",
            }
        )
        # Point error model at both sources
        data["calibration"]["error_model"][0]["uses_inputs"] = ["obs_mean", "mouse_val"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            target = SubmodelTarget(**data)

        sigma_primary, _ = compute_translation_sigma(target.primary_data_source.source_relevance)
        sigma_secondary, _ = compute_translation_sigma(
            target.secondary_data_sources[0].source_relevance
        )
        expected = float(np.sqrt(sigma_primary**2 + sigma_secondary**2))

        sigma, breakdown = resolve_per_measurement_sigma(target, ["obs_mean", "mouse_val"])
        assert abs(sigma - expected) < 1e-6
        assert len(breakdown) == 2  # two unique sources

    def test_no_uses_inputs_falls_back_to_primary(self):
        """Entry with uses_inputs=None falls back to primary source."""
        from qsp_inference.submodel.inference import (
            resolve_per_measurement_sigma,
        )
        from qsp_inference.submodel.prior import compute_translation_sigma

        target = self._make_target()
        primary_sr = target.primary_data_source.source_relevance
        expected_sigma, _ = compute_translation_sigma(primary_sr)

        sigma, breakdown = resolve_per_measurement_sigma(target, None)
        assert abs(sigma - expected_sigma) < 1e-6


# =============================================================================
# Tests: NumPyro model smoke test
# =============================================================================


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
class TestSubmodelJointModel:
    def test_smoke_single_param(self):
        """Minimal smoke test: 1 parameter, 1 likelihood term, short MCMC."""
        import jax
        import jax.random
        from numpyro.infer import MCMC, NUTS

        from qsp_inference.submodel.inference import submodel_joint_model

        prior_specs = {
            "k": PriorSpec(name="k", distribution="lognormal", units="1/day", mu=0.0, sigma=1.0)
        }

        # Observed value=2.0, sigma=0.3, forward model is identity
        def identity_fn(params):
            return params["k"]

        entry = ErrorModelEntry(
            forward_fn=identity_fn,
            value=2.0,
            sigma=0.3,
            family="lognormal",
            fit=DistFit(
                name="lognormal",
                params={"mu": np.log(2.0), "sigma": 0.3},
                aic=0.0,
                ad_stat=0.0,
                ad_crit_5pct=1.0,
                ad_pass=True,
                median=2.0,
                cv=0.3,
            ),
            sigma_trans=0.15,
        )

        tl = TargetLikelihood(
            target_id="test",
            entries=[entry],
        )

        kernel = NUTS(submodel_joint_model)
        mcmc = MCMC(kernel, num_warmup=100, num_samples=200, num_chains=1)
        mcmc.run(
            jax.random.PRNGKey(0),
            prior_specs=prior_specs,
            target_likelihoods=[tl],
        )

        samples = mcmc.get_samples()
        assert "k" in samples
        k_samples = np.asarray(samples["k"])
        assert k_samples.shape == (200,)
        # Posterior should be pulled toward observed value of 2.0
        assert np.median(k_samples) == pytest.approx(2.0, rel=0.5)

    def test_nan_forward_fn_does_not_crash(self):
        """Forward functions that return NaN should not crash MCMC.

        This simulates the scenario where an ODE solver fails for extreme
        parameter values (e.g. diffrax hitting max_steps). The NaN guard
        in submodel_joint_model should assign -inf log-probability so NUTS
        rejects those samples gracefully.
        """
        import jax
        import jax.numpy as jnp
        import jax.random
        from numpyro.infer import MCMC, NUTS

        from qsp_inference.submodel.inference import submodel_joint_model

        prior_specs = {
            "k": PriorSpec(name="k", distribution="lognormal", units="1/day", mu=0.0, sigma=1.0)
        }

        # Forward function that returns NaN when k > 5 (simulating solver failure)
        def unstable_fn(params):
            k = params["k"]
            return jnp.where(k < 5.0, k, jnp.nan)

        entry = ErrorModelEntry(
            forward_fn=unstable_fn,
            value=2.0,
            sigma=0.3,
            family="lognormal",
            fit=DistFit(
                name="lognormal",
                params={"mu": np.log(2.0), "sigma": 0.3},
                aic=0.0,
                ad_stat=0.0,
                ad_crit_5pct=1.0,
                ad_pass=True,
                median=2.0,
                cv=0.3,
            ),
            sigma_trans=0.15,
        )

        tl = TargetLikelihood(
            target_id="nan_test",
            entries=[entry],
        )

        kernel = NUTS(submodel_joint_model)
        mcmc = MCMC(kernel, num_warmup=100, num_samples=200, num_chains=1)
        # This should NOT raise "Normal distribution got invalid loc parameter"
        mcmc.run(
            jax.random.PRNGKey(42),
            prior_specs=prior_specs,
            target_likelihoods=[tl],
        )

        samples = mcmc.get_samples()
        k_samples = np.asarray(samples["k"])
        # All accepted samples should be finite (NaN proposals were rejected)
        assert np.all(np.isfinite(k_samples))
        # Posterior should still be near the observed value
        assert np.median(k_samples) == pytest.approx(2.0, rel=0.5)
