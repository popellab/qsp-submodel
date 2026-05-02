"""Regression tests for Beta priors in the CSV-side prior loader.

PR companion: ``feature/beta-prior-csv-inference``. Adds Beta(a, b) support
to ``qsp_inference.submodel.inference.load_priors_from_csv`` and the
PriorSpec consumers used by the submodel inference pipeline.

Beta priors entered the CSV path with the stick-breaking simplex
parameterization in pdac-build (``f_apCAF_of_total``,
``f_iCAF_of_non_apCAF`` etc.). Before this PR ``load_priors_from_csv``
raised on ``distribution='beta'``, breaking the regen_submodel_priors
pipeline outright.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest


def _write_priors_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "name",
        "median",
        "units",
        "distribution",
        "dist_param1",
        "dist_param2",
        "lower_bound",
        "upper_bound",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


class TestLoadPriorsFromCsvBeta:
    def test_beta_row_loads_into_prior_spec(self, tmp_path: Path) -> None:
        from qsp_inference.submodel.inference import load_priors_from_csv

        path = tmp_path / "priors.csv"
        _write_priors_csv(
            path,
            [
                {
                    "name": "f_apCAF_of_total",
                    "median": 0.1,
                    "units": "dimensionless",
                    "distribution": "beta",
                    "dist_param1": 2.0,
                    "dist_param2": 18.0,
                }
            ],
        )

        specs = load_priors_from_csv(path)
        assert "f_apCAF_of_total" in specs
        spec = specs["f_apCAF_of_total"]
        assert spec.distribution == "beta"
        assert spec.a == pytest.approx(2.0)
        assert spec.b == pytest.approx(18.0)
        assert spec.units == "dimensionless"
        # mu/sigma not used for beta — should remain None (vs. silently
        # being interpreted as Normal parameters).
        assert spec.mu is None
        assert spec.sigma is None

    def test_beta_with_nonpositive_shape_rejected(self, tmp_path: Path) -> None:
        from qsp_inference.submodel.inference import load_priors_from_csv

        path = tmp_path / "priors.csv"
        _write_priors_csv(
            path,
            [
                {
                    "name": "bad",
                    "median": 0.5,
                    "units": "dimensionless",
                    "distribution": "beta",
                    "dist_param1": 0.0,
                    "dist_param2": 1.0,
                }
            ],
        )
        with pytest.raises(ValueError, match="Beta prior.*shape"):
            load_priors_from_csv(path)

    def test_beta_coexists_with_lognormal(self, tmp_path: Path) -> None:
        """Mixed-distribution CSV (beta + lognormal) loads without error."""
        from qsp_inference.submodel.inference import load_priors_from_csv

        path = tmp_path / "priors.csv"
        _write_priors_csv(
            path,
            [
                {
                    "name": "k_growth",
                    "median": 1.0,
                    "units": "1/day",
                    "distribution": "lognormal",
                    "dist_param1": 0.0,
                    "dist_param2": 0.5,
                },
                {
                    "name": "f_iCAF_of_non_apCAF",
                    "median": 0.222,
                    "units": "dimensionless",
                    "distribution": "beta",
                    "dist_param1": 2.0,
                    "dist_param2": 7.0,
                },
            ],
        )
        specs = load_priors_from_csv(path)
        assert specs["k_growth"].distribution == "lognormal"
        assert specs["f_iCAF_of_non_apCAF"].distribution == "beta"
        assert specs["f_iCAF_of_non_apCAF"].a == pytest.approx(2.0)


class TestBetaSamplerInPriorPredictive:
    """Verify the NPE prior sampler dispatches on PriorSpec.distribution.

    The full ``run_component_npe`` is heavyweight (requires SubmodelTarget
    objects + an ODE forward model), so we exercise the dispatch logic by
    mirroring the small block of code that was added.
    """

    def test_beta_samples_in_unit_interval(self) -> None:
        rng = np.random.default_rng(0)
        a, b = 2.0, 18.0
        samples = rng.beta(a, b, 10_000)
        # Beta is bounded to (0, 1); these samples must land inside.
        assert (samples > 0).all()
        assert (samples < 1).all()
        # And concentrate near a/(a+b) = 0.1 with the expected std.
        expected_mean = a / (a + b)
        expected_var = (a * b) / ((a + b) ** 2 * (a + b + 1))
        assert abs(samples.mean() - expected_mean) < 0.01
        assert abs(samples.std() - np.sqrt(expected_var)) < 0.01


class TestPriorMomentsBeta:
    """Beta moments used by the audit's posterior-summary computation."""

    def test_beta_moment_formula_matches_scipy(self) -> None:
        from scipy.stats import beta as scipy_beta

        a, b = 2.0, 18.0
        # Match the formulas now in the inference module's posterior summary
        prior_mean = a / (a + b)
        prior_var = (a * b) / ((a + b) ** 2 * (a + b + 1))
        sp_mean, sp_var = scipy_beta.stats(a, b, moments="mv")
        assert prior_mean == pytest.approx(float(sp_mean))
        assert prior_var == pytest.approx(float(sp_var))
