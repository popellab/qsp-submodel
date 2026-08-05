"""Target-set resolution and scoring for refit_check."""

import yaml

from qsp_inference.submodel.ppc_audit import ComponentPPC, ObservablePPC
from qsp_inference.submodel.refit_check import (
    resolve_target_set,
    score,
    target_index,
)

GLOB = "*_deriv*.yaml"


def _target(path, target_id, params):
    path.write_text(yaml.safe_dump({
        "target_id": target_id,
        "calibration": {"parameters": [{"name": p} for p in params]},
    }))


def _config(path, cuts):
    path.write_text(yaml.safe_dump({"cascade_cuts": cuts}))


def _obs(name, observed, post_median, covered):
    return ObservablePPC(
        name=name, observed=observed, obs_ci95=(observed / 2, observed * 2),
        prior_median=observed, prior_lo=observed / 10, prior_hi=observed * 10,
        prior_log_width=2.0, post_median=post_median,
        post_ci95=(post_median / 2, post_median * 2), covered=covered,
    )


def test_target_index_maps_ids_and_params(tmp_path):
    _target(tmp_path / "a_deriv001.yaml", "alpha", ["k_a", "k_b"])
    by_id, params = target_index(tmp_path, GLOB)
    assert by_id == {"alpha": "a_deriv001.yaml"}
    assert params["a_deriv001.yaml"] == {"k_a", "k_b"}


def test_resolve_pulls_every_cut_upstream_not_just_triggered_ones(tmp_path):
    # _build_stage_dag walks the whole cut list and raises on any upstream it
    # cannot place, so an unrelated cut's upstream must come along too.
    _target(tmp_path / "a_deriv001.yaml", "alpha", ["k_a"])
    _target(tmp_path / "b_deriv001.yaml", "beta", ["k_b"])
    _target(tmp_path / "c_deriv001.yaml", "gamma", ["k_c"])
    cfg = tmp_path / "submodel_config.yaml"
    _config(cfg, [{"parameter": "k_c", "upstream": ["gamma"]}])

    got = resolve_target_set(tmp_path, ["a_deriv001.yaml"], cfg, GLOB)
    assert got == ["a_deriv001.yaml", "c_deriv001.yaml"]


def test_resolve_closes_transitively(tmp_path):
    _target(tmp_path / "a_deriv001.yaml", "alpha", ["k_a"])
    _target(tmp_path / "b_deriv001.yaml", "beta", ["k_b"])
    _target(tmp_path / "c_deriv001.yaml", "gamma", ["k_c"])
    cfg = tmp_path / "submodel_config.yaml"
    # k_a cuts to beta; beta declares k_b, which cuts to gamma.
    _config(cfg, [
        {"parameter": "k_a", "upstream": ["beta"]},
        {"parameter": "k_b", "upstream": ["gamma"]},
    ])

    got = resolve_target_set(tmp_path, ["a_deriv001.yaml"], cfg, GLOB)
    assert got == ["a_deriv001.yaml", "b_deriv001.yaml", "c_deriv001.yaml"]


def test_resolve_without_config_is_identity(tmp_path):
    _target(tmp_path / "a_deriv001.yaml", "alpha", ["k_a"])
    assert resolve_target_set(tmp_path, ["a_deriv001.yaml"], None, GLOB) == [
        "a_deriv001.yaml"
    ]


def test_score_restricts_to_named_params():
    mine = ComponentPPC(component="c1", params=["k_a"], targets=[],
                        observables=[_obs("o1", 1.0, 1.0, True)])
    # A target pulled in only to satisfy a cascade cut must not move the score.
    theirs = ComponentPPC(component="c2", params=["k_z"], targets=[],
                          observables=[_obs("o2", 1.0, 100.0, False)])

    both = score([mine, theirs])
    assert both.coverage == 0.5
    assert both.n_observables == 2

    own = score([mine, theirs], params={"k_a"})
    assert own.coverage == 1.0
    assert own.n_observables == 1


def test_score_worst_log10_is_the_largest_miss():
    comp = ComponentPPC(component="c1", params=["k_a"], targets=[], observables=[
        _obs("near", 1.0, 1.2, True),
        _obs("far", 1.0, 100.0, False),
    ])
    assert score([comp]).worst_log10 == 2.0


def test_score_of_nothing_is_empty():
    assert score([]).n_observables == 0
