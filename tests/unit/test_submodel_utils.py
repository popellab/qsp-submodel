#!/usr/bin/env python3
"""
Tests for submodel utility functions.

Tests the standalone functions in submodel_utils.py:
- get_prior_median: Extract median from prior specifications
- build_ode_function: Instantiate ODE from model type
- solve_submodel: Integrate ODE system
- compute_observable: Compute observable from ODE solution
- run_prior_predictive: Full prior predictive workflow
"""

import math
import pytest
from unittest.mock import Mock

from qsp_inference.submodel.utils import (
    get_prior_median,
    build_ode_function,
    solve_submodel,
    compute_observable,
    run_prior_predictive,
    PriorPredictiveError,
)


# ============================================================================
# Tests for get_prior_median
# ============================================================================


class TestGetPriorMedian:
    """Tests for get_prior_median function."""

    def test_lognormal_prior(self):
        """Lognormal prior returns exp(mu)."""
        prior = Mock()
        prior.distribution = "lognormal"
        prior.mu = 0.0
        prior.sigma = 1.0

        result = get_prior_median(prior)
        assert result == pytest.approx(1.0)  # exp(0) = 1

    def test_lognormal_prior_negative_mu(self):
        """Lognormal prior with negative mu."""
        prior = Mock()
        prior.distribution = "lognormal"
        prior.mu = -2.0
        prior.sigma = 0.5

        result = get_prior_median(prior)
        assert result == pytest.approx(math.exp(-2.0))

    def test_lognormal_prior_enum_distribution(self):
        """Lognormal prior with enum distribution."""
        prior = Mock()
        prior.distribution = Mock()
        prior.distribution.value = "lognormal"
        prior.mu = 1.0
        prior.sigma = 0.5

        result = get_prior_median(prior)
        assert result == pytest.approx(math.exp(1.0))

    def test_normal_prior(self):
        """Normal prior returns mu."""
        prior = Mock()
        prior.distribution = "normal"
        prior.mu = 5.0
        prior.sigma = 2.0

        result = get_prior_median(prior)
        assert result == 5.0

    def test_uniform_prior(self):
        """Uniform prior returns midpoint."""
        prior = Mock()
        prior.distribution = "uniform"
        prior.lower = 0.0
        prior.upper = 10.0

        result = get_prior_median(prior)
        assert result == 5.0

    def test_half_normal_prior(self):
        """Half-normal prior returns 0.674 * sigma."""
        prior = Mock()
        prior.distribution = "half_normal"
        prior.sigma = 2.0

        result = get_prior_median(prior)
        assert result == pytest.approx(0.674 * 2.0)

    def test_none_prior(self):
        """None prior returns None."""
        result = get_prior_median(None)
        assert result is None

    def test_lognormal_missing_mu(self):
        """Lognormal prior with missing mu returns None."""
        prior = Mock()
        prior.distribution = "lognormal"
        prior.mu = None
        prior.sigma = 1.0

        result = get_prior_median(prior)
        assert result is None


# ============================================================================
# Tests for build_ode_function
# ============================================================================


class TestBuildOdeFunction:
    """Tests for build_ode_function."""

    def test_first_order_decay(self):
        """First-order decay ODE: dy/dt = -k*y."""
        model = Mock()
        model.type = "first_order_decay"
        model.rate_constant = "k_decay"

        params = {"k_decay": 0.1}
        inputs = {}

        ode = build_ode_function(model, params, inputs)
        assert ode is not None

        # Test at y=100: dy/dt = -0.1 * 100 = -10
        dydt = ode(0.0, [100.0])
        assert dydt == [-10.0]

    def test_exponential_growth(self):
        """Exponential growth ODE: dy/dt = k*y."""
        model = Mock()
        model.type = "exponential_growth"
        model.rate_constant = "k_growth"

        params = {"k_growth": 0.5}
        inputs = {}

        ode = build_ode_function(model, params, inputs)
        assert ode is not None

        # Test at y=10: dy/dt = 0.5 * 10 = 5
        dydt = ode(0.0, [10.0])
        assert dydt == [5.0]

    def test_two_state_model(self):
        """Two-state transition: dA/dt = -k*A, dB/dt = +k*A."""
        model = Mock()
        model.type = "two_state"
        model.forward_rate = "k_activation"

        params = {"k_activation": 0.2}
        inputs = {}

        ode = build_ode_function(model, params, inputs)
        assert ode is not None

        # Test at A=100, B=0: dA/dt = -20, dB/dt = +20
        dydt = ode(0.0, [100.0, 0.0])
        assert dydt == [-20.0, 20.0]

    def test_saturation_model(self):
        """Saturation: dy/dt = k*(1-y)."""
        model = Mock()
        model.type = "saturation"
        model.rate_constant = "k_sat"

        params = {"k_sat": 0.1}
        inputs = {}

        ode = build_ode_function(model, params, inputs)
        assert ode is not None

        # Test at y=0.5: dy/dt = 0.1 * (1 - 0.5) = 0.05
        dydt = ode(0.0, [0.5])
        assert dydt == [pytest.approx(0.05)]

    def test_logistic_model(self):
        """Logistic growth: dy/dt = k*y*(1 - y/K)."""
        model = Mock()
        model.type = "logistic"
        model.rate_constant = "k_growth"
        model.carrying_capacity = "K_max"

        params = {"k_growth": 0.1, "K_max": 1000.0}
        inputs = {}

        ode = build_ode_function(model, params, inputs)
        assert ode is not None

        # Test at y=500: dy/dt = 0.1 * 500 * (1 - 500/1000) = 25
        dydt = ode(0.0, [500.0])
        assert dydt == [pytest.approx(25.0)]

    def test_michaelis_menten(self):
        """Michaelis-Menten: dy/dt = -Vmax*y/(Km+y)."""
        model = Mock()
        model.type = "michaelis_menten"
        model.vmax = "V_max"
        model.km = "K_m"

        params = {"V_max": 10.0, "K_m": 5.0}
        inputs = {}

        ode = build_ode_function(model, params, inputs)
        assert ode is not None

        # Test at y=5: dy/dt = -10 * 5 / (5 + 5) = -5
        dydt = ode(0.0, [5.0])
        assert dydt == [pytest.approx(-5.0)]

    def test_direct_conversion_returns_none(self):
        """Direct conversion model has no ODE."""
        model = Mock()
        model.type = "direct_conversion"

        ode = build_ode_function(model, {}, {})
        assert ode is None

    def test_input_ref_parameter(self):
        """Parameter from input_ref uses input value."""
        model = Mock()
        model.type = "first_order_decay"
        model.rate_constant = Mock()
        model.rate_constant.input_ref = "k_from_data"

        params = {}
        inputs = {"k_from_data": 0.3}

        ode = build_ode_function(model, params, inputs)
        assert ode is not None

        # dy/dt = -0.3 * 100 = -30
        dydt = ode(0.0, [100.0])
        assert dydt == [-30.0]


# ============================================================================
# Tests for solve_submodel
# ============================================================================


class TestSolveSubmodel:
    """Tests for solve_submodel function."""

    def test_first_order_decay_integration(self):
        """Integrate first-order decay: y(t) = y0 * exp(-k*t)."""
        model = Mock()
        model.type = "first_order_decay"
        model.rate_constant = "k"

        params = {"k": 0.1}
        inputs = {}
        y0 = [100.0]
        t_span = (0.0, 10.0)

        result = solve_submodel(model, params, inputs, y0, t_span)

        assert result is not None
        assert "t" in result
        assert "y" in result

        # Check final value: y(10) = 100 * exp(-0.1 * 10) ≈ 36.79
        y_final = result["y"][0, -1]
        expected = 100.0 * math.exp(-0.1 * 10.0)
        assert y_final == pytest.approx(expected, rel=0.01)

    def test_exponential_growth_integration(self):
        """Integrate exponential growth: y(t) = y0 * exp(k*t)."""
        model = Mock()
        model.type = "exponential_growth"
        model.rate_constant = "k"

        params = {"k": 0.2}
        inputs = {}
        y0 = [10.0]
        t_span = (0.0, 5.0)

        result = solve_submodel(model, params, inputs, y0, t_span)

        assert result is not None
        y_final = result["y"][0, -1]
        expected = 10.0 * math.exp(0.2 * 5.0)
        assert y_final == pytest.approx(expected, rel=0.01)

    def test_two_state_integration(self):
        """Integrate two-state: A decays, B grows."""
        model = Mock()
        model.type = "two_state"
        model.forward_rate = "k"

        params = {"k": 0.1}
        inputs = {}
        y0 = [100.0, 0.0]
        t_span = (0.0, 10.0)

        result = solve_submodel(model, params, inputs, y0, t_span)

        assert result is not None
        # A(t) = 100 * exp(-0.1*t), B(t) = 100 * (1 - exp(-0.1*t))
        A_final = result["y"][0, -1]
        B_final = result["y"][1, -1]

        expected_A = 100.0 * math.exp(-0.1 * 10.0)
        expected_B = 100.0 * (1 - math.exp(-0.1 * 10.0))

        assert A_final == pytest.approx(expected_A, rel=0.01)
        assert B_final == pytest.approx(expected_B, rel=0.01)

    def test_with_t_eval(self):
        """Integration with specific evaluation points."""
        model = Mock()
        model.type = "first_order_decay"
        model.rate_constant = "k"

        params = {"k": 0.1}
        inputs = {}
        y0 = [100.0]
        t_span = (0.0, 10.0)
        t_eval = [0.0, 2.5, 5.0, 7.5, 10.0]

        result = solve_submodel(model, params, inputs, y0, t_span, t_eval)

        assert result is not None
        assert len(result["t"]) == 5

    def test_direct_conversion_returns_none(self):
        """Direct conversion model returns None (no ODE)."""
        model = Mock()
        model.type = "direct_conversion"

        result = solve_submodel(model, {}, {}, [1.0], (0.0, 1.0))
        assert result is None


# ============================================================================
# Tests for compute_observable
# ============================================================================


class TestComputeObservable:
    """Tests for compute_observable function."""

    def test_final_value_observable(self):
        """Final value observable returns last state value."""
        import numpy as np

        solution = {
            "t": np.array([0.0, 1.0, 2.0]),
            "y": np.array([[100.0, 80.0, 60.0]]),  # Shape: (n_states, n_timepoints)
        }
        observable = Mock()
        observable.type = "final_value"
        observable.state_variables = ["A"]

        result = compute_observable(solution, observable, [100.0], ["A"])
        assert result == 60.0

    def test_fraction_remaining_observable(self):
        """Fraction remaining: y_final / y0."""
        import numpy as np

        solution = {
            "t": np.array([0.0, 1.0, 2.0]),
            "y": np.array([[100.0, 50.0, 25.0]]),
        }
        observable = Mock()
        observable.type = "fraction_remaining"
        observable.state_variables = ["A"]

        result = compute_observable(solution, observable, [100.0], ["A"])
        assert result == 0.25  # 25 / 100

    def test_fold_change_observable(self):
        """Fold change: y_final / y0."""
        import numpy as np

        solution = {
            "t": np.array([0.0, 1.0, 2.0]),
            "y": np.array([[10.0, 20.0, 40.0]]),
        }
        observable = Mock()
        observable.type = "fold_change"
        observable.state_variables = ["A"]

        result = compute_observable(solution, observable, [10.0], ["A"])
        assert result == 4.0  # 40 / 10

    def test_auc_observable(self):
        """AUC: trapezoidal integration."""
        import numpy as np

        solution = {
            "t": np.array([0.0, 1.0, 2.0]),
            "y": np.array([[10.0, 10.0, 10.0]]),  # Constant = 10
        }
        observable = Mock()
        observable.type = "auc"
        observable.state_variables = ["A"]

        result = compute_observable(solution, observable, [10.0], ["A"])
        assert result == pytest.approx(20.0)  # 10 * 2 = 20

    def test_max_value_observable(self):
        """Max value across time series."""
        import numpy as np

        solution = {
            "t": np.array([0.0, 1.0, 2.0, 3.0]),
            "y": np.array([[10.0, 50.0, 30.0, 20.0]]),
        }
        observable = Mock()
        observable.type = "max_value"
        observable.state_variables = ["A"]

        result = compute_observable(solution, observable, [10.0], ["A"])
        assert result == 50.0

    def test_none_observable_returns_first_state(self):
        """No observable returns first state's final value."""
        import numpy as np

        solution = {
            "t": np.array([0.0, 1.0]),
            "y": np.array([[100.0, 50.0], [0.0, 50.0]]),  # Two states
        }

        result = compute_observable(solution, None, [100.0, 0.0])
        assert result == 50.0  # First state's final value

    def test_multi_state_variable_indexing(self):
        """Observable from second state variable."""
        import numpy as np

        solution = {
            "t": np.array([0.0, 1.0]),
            "y": np.array([[100.0, 50.0], [0.0, 50.0]]),  # A decays, B grows
        }
        observable = Mock()
        observable.type = "final_value"
        observable.state_variables = ["B"]

        result = compute_observable(solution, observable, [100.0, 0.0], ["A", "B"])
        assert result == 50.0  # B's final value


# ============================================================================
# Tests for run_prior_predictive
# ============================================================================


class TestRunPriorPredictive:
    """Tests for run_prior_predictive function."""

    def test_direct_conversion_returns_prior_median(self):
        """Direct conversion: prediction = prior median."""
        model = Mock()
        model.type = "direct_conversion"

        prior = Mock()
        prior.distribution = "lognormal"
        prior.mu = 0.0
        prior.sigma = 1.0

        result = run_prior_predictive(
            model=model,
            prior=prior,
            param_name="k",
            state_variables=None,
            independent_variable=None,
            measurement=None,
            input_values={},
        )

        assert result == pytest.approx(1.0)  # exp(0) = 1

    def test_first_order_decay_prediction(self):
        """First-order decay with fraction_remaining observable."""
        model = Mock()
        model.type = "first_order_decay"
        model.rate_constant = "k_decay"

        prior = Mock()
        prior.distribution = "lognormal"
        prior.mu = math.log(0.1)  # median = 0.1
        prior.sigma = 0.5

        # State variable with fixed IC
        state_var = Mock()
        state_var.name = "A"
        state_var.initial_condition = Mock()
        state_var.initial_condition.value = 100.0

        # Independent variable
        indep_var = Mock()
        indep_var.span = [0.0, 10.0]

        # Observable
        observable = Mock()
        observable.type = "fraction_remaining"
        observable.state_variables = ["A"]

        # Measurement
        measurement = Mock()
        measurement.observable = observable
        measurement.evaluation_points = [10.0]

        result = run_prior_predictive(
            model=model,
            prior=prior,
            param_name="k_decay",
            state_variables=[state_var],
            independent_variable=indep_var,
            measurement=measurement,
            input_values={},
        )

        # Expected: exp(-0.1 * 10) ≈ 0.368
        assert result == pytest.approx(math.exp(-0.1 * 10.0), rel=0.05)

    def test_missing_state_variables_raises_error(self):
        """ODE model without state variables raises PriorPredictiveError."""
        model = Mock()
        model.type = "first_order_decay"
        model.rate_constant = "k"

        prior = Mock()
        prior.distribution = "normal"
        prior.mu = 0.1
        prior.sigma = 0.05

        with pytest.raises(PriorPredictiveError) as exc_info:
            run_prior_predictive(
                model=model,
                prior=prior,
                param_name="k",
                state_variables=None,  # Missing
                independent_variable=Mock(span=[0.0, 10.0]),
                measurement=None,
                input_values={},
            )

        assert "state_variables" in str(exc_info.value)
