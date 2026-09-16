from types import SimpleNamespace

import pytest
import yaml

from qsp_submodel.submodel import comparison, aggregate, inference


def setup_inputs(tmp_path):
    priors = tmp_path / 'priors.csv'
    priors.write_text('name,median,units,distribution,dist_param1,dist_param2\nk,1,1/day,lognormal,0,1\n')
    targets = tmp_path / 'targets'
    targets.mkdir()
    (targets / 'k_ibocta_robertson2006.yaml').write_text(yaml.safe_dump({
        'target_id': 'robertson', 'calibration': {'parameters': [{'name': 'k', 'nuisance': False}]}
    }))
    return priors, targets


def test_empty_selection_is_error(tmp_path):
    priors, targets = setup_inputs(tmp_path)
    with pytest.raises(ValueError, match='No target YAML'):
        comparison.run_comparison(priors, targets, glob_pattern='*_PDAC_*.yaml')


def test_invalid_target_aborts_and_generated_yaml_is_excluded(tmp_path, monkeypatch):
    from maple.core.calibration.submodel_target import SubmodelTarget
    priors, targets = setup_inputs(tmp_path)
    (targets / 'compare_inference_results.yaml').write_text('parameters: {}')
    (targets / 'submodel_priors.yaml').write_text('parameters: []')
    (targets / 'submodel_config.yaml').write_text('groups: []')
    def fail(*args, **kwargs):
        raise ValueError('invalid evidence')
    monkeypatch.setattr(SubmodelTarget, 'model_validate', fail)
    with pytest.raises(ValueError, match='Failed to validate target k_ibocta_robertson2006.yaml'):
        comparison.run_comparison(priors, targets)
    assert not list((targets / '.compare_cache').glob('comp_*.json'))


def test_failed_component_aborts(tmp_path, monkeypatch):
    from maple.core.calibration.submodel_target import SubmodelTarget
    priors, targets = setup_inputs(tmp_path)
    target = SimpleNamespace(target_id='robertson', calibration=SimpleNamespace(
        forward_model=SimpleNamespace(type='algebraic')))
    monkeypatch.setattr(SubmodelTarget, 'model_validate', lambda *a, **kw: target)
    def fail(*args, **kwargs):
        assert kwargs['target_accept_prob'] == 0.99
        raise RuntimeError('sampler failed')
    monkeypatch.setattr(inference, 'run_joint_inference', fail)
    with pytest.raises(RuntimeError, match='Component 1 failed.*sampler failed'):
        comparison.run_comparison(priors, targets, target_accept_prob=0.99)
    assert not (targets / 'compare_inference_results.yaml').exists()


def test_coverage_includes_non_deriv_names(tmp_path):
    _, targets = setup_inputs(tmp_path)
    assert aggregate.load_submodel_targets(targets) == {'k': ['robertson']}


def test_beta_median_is_not_exponentiated_log_mean():
    spec = inference.PriorSpec(name='f', units='dimensionless', distribution='beta', a=1, b=1, mu=-1, sigma=1)
    assert comparison._prior_median(spec) == pytest.approx(0.5)


def test_invalid_acceptance_probability():
    with pytest.raises(ValueError, match='target_accept_prob'):
        inference.run_joint_inference({}, [], target_accept_prob=1)
