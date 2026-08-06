"""
Parameterize MCMC posterior samples as marginal distributions + Gaussian copula.

Takes the joint posterior samples from submodel_inference.run_joint_inference()
and produces a parametric representation:
  - Per-parameter marginal (lognormal / gamma / inv-gamma, best by AIC)
  - Gaussian copula capturing posterior correlations

Output is written to submodel_priors.yaml for consumption by qsp-sbi.

Usage::

    from qsp_inference.submodel.parameterizer import parameterize_posteriors, write_priors_yaml

    result = parameterize_posteriors(samples, targets)
    write_priors_yaml(result, Path("submodel_priors.yaml"))
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

from qsp_inference.submodel.parameter_groups import ParameterGroupsConfig
from maple.core.calibration.submodel_target import SubmodelTarget
from qsp_inference.submodel.prior import DistFit, fit_distributions


# =============================================================================
# Marginal fitting
# =============================================================================


#: Interior grid points per parameter. Sets the marginal's KS resolution at
#: 1/(N+1), so 199 puts it at 0.005 against a 5% critical value near 0.015.
N_LOG_GRID = 199


def empirical_log_grid(samples: np.ndarray, n_grid: int = N_LOG_GRID) -> dict:
    """A parameter's posterior as quantiles of ``log theta``, against normal scores.

    Returned as ``{"distribution": "empirical_log", "z": [...], "q": [...]}``,
    where ``q[i]`` is the ``Phi(z[i])`` quantile of the log samples.

    Indexed by normal score rather than by probability so that extrapolating
    linearly past the end of the grid is exactly a Gaussian tail. A grid in ``p``
    would have to invent one, and a marginal with no tail returns ``-inf`` for
    any draw outside the sampled range.

    A parametric fit is not used because on this corpus it fails Anderson-Darling
    for 61% of parameters and no candidate family passes for 113 of them.
    """
    s = np.asarray(samples, dtype=float)
    pos = s[s > 0]
    if len(pos) < 100:
        raise ValueError(
            f"need at least 100 positive samples to build a grid, got {len(pos)}"
        )
    logs = np.log(pos)

    p = np.arange(1, n_grid + 1) / (n_grid + 1)
    q = np.quantile(logs, p)
    # Strictly increasing, or the interpolant is not invertible. Ties happen
    # where a chain parked on a bound.
    q = np.maximum.accumulate(q)
    bump = np.arange(n_grid) * np.spacing(max(abs(q[0]), abs(q[-1]), 1.0))
    q = q + bump

    return {
        "distribution": "empirical_log",
        "z": [float(x) for x in stats.norm.ppf(p)],
        "q": [float(x) for x in q],
    }


def split_grids_to_sidecar(parameters: list[dict], yaml_path: Path) -> Optional[dict]:
    """Move the quantile grids out of ``parameters`` into a ``.npz`` beside the YAML.

    Mutates ``parameters`` so each grid marginal carries ``grid`` and ``index``
    instead of its ``z``/``q`` arrays. Returns the sidecar's manifest entry, or
    ``None`` when no parameter carries a grid.

    Inline, the grids make the file 12x larger and 5x slower to parse, and every
    parameter repeats the same ``z``. They are also not what anyone opens the
    YAML to read.
    """
    idx = [i for i, p in enumerate(parameters)
           if p.get("marginal", {}).get("distribution") == "empirical_log"]
    if not idx:
        return None

    z0 = parameters[idx[0]]["marginal"]["z"]
    for i in idx:
        if parameters[i]["marginal"]["z"] != z0:
            raise ValueError(
                f"{parameters[i]['name']} carries a different z grid. The sidecar "
                "stores z once, so every grid has to share it."
            )

    q = np.array([parameters[i]["marginal"]["q"] for i in idx], dtype=float)
    names = [parameters[i]["name"] for i in idx]

    yaml_path = Path(yaml_path)
    npz_path = yaml_path.with_suffix(".npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    # Uncompressed on purpose. Quantile grids of float64 barely compress (268 KB
    # against 281 KB), and this file is version-controlled: a pre-compressed blob
    # is opaque to git's delta, so every regeneration would store the whole thing
    # again. 13 KB buys successive versions that can delta against each other.
    np.savez(npz_path, z=np.asarray(z0, dtype=float), q=q,
             names=np.array(names, dtype=object), allow_pickle=True)

    for slot, i in enumerate(idx):
        m = parameters[i]["marginal"]
        del m["z"], m["q"]
        m["grid"] = npz_path.name
        m["index"] = slot

    return {
        "file": npz_path.name,
        "sha256": _sha256_file(npz_path),
        "n_parameters": len(idx),
        "n_grid": int(q.shape[1]),
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_marginals(samples: dict[str, np.ndarray]) -> dict[str, DistFit]:
    """Fit marginal distributions to each parameter's posterior samples.

    Args:
        samples: {param_name: 1D array of posterior samples}

    Returns:
        {param_name: best DistFit} sorted by AIC
    """
    marginals = {}
    for name, s in samples.items():
        fits = fit_distributions(s)
        if not fits:
            raise RuntimeError(
                f"Could not fit any distribution to posterior samples for '{name}'. "
                f"Got {len(s)} samples, {np.sum(s > 0)} positive."
            )
        marginals[name] = fits[0]
    return marginals


# =============================================================================
# Gaussian copula
# =============================================================================


def _build_marginal_cdf(fit: DistFit):
    """Build a CDF function from a DistFit result."""
    if fit.name == "lognormal":
        mu, sigma = fit.params["mu"], fit.params["sigma"]
        return lambda x: stats.lognorm.cdf(x, s=sigma, scale=np.exp(mu))
    elif fit.name == "gamma":
        shape, scale = fit.params["shape"], fit.params["scale"]
        return lambda x: stats.gamma.cdf(x, shape, scale=scale)
    elif fit.name == "invgamma":
        shape, scale = fit.params["shape"], fit.params["scale"]
        return lambda x: stats.invgamma.cdf(x, shape, scale=scale)
    elif fit.name == "normal":
        mu, sigma = fit.params["mu"], fit.params["sigma"]
        return lambda x: stats.norm.cdf(x, loc=mu, scale=sigma)
    else:
        raise ValueError(f"Unknown distribution: {fit.name}")


def rank_gaussian_copula(samples_matrix: np.ndarray) -> np.ndarray:
    """Gaussian copula correlation from rank pseudo-observations.

    ``u = rank / (n + 1)`` rather than ``F_hat(x)``, so the dependence estimate
    does not depend on how well the marginals were fitted. Going through fitted
    CDFs leaves ``u`` non-uniform wherever a marginal is off, and the correlation
    then absorbs that error as if it were dependence.
    """
    x = np.asarray(samples_matrix, dtype=float)
    n = x.shape[0]
    u = np.empty_like(x)
    for j in range(x.shape[1]):
        u[:, j] = stats.rankdata(x[:, j], method="average") / (n + 1)
    z = stats.norm.ppf(u)
    return np.corrcoef(z, rowvar=False)


def fit_gaussian_copula(
    samples_matrix: np.ndarray,
    marginal_cdfs: list,
) -> np.ndarray:
    """Fit a Gaussian copula to joint posterior samples.

    Args:
        samples_matrix: (N_samples, N_params) array
        marginal_cdfs: list of CDF callables, one per parameter

    Returns:
        (N_params, N_params) correlation matrix of the Gaussian copula
    """
    n_samples, n_params = samples_matrix.shape

    # Probability integral transform: samples -> uniform
    u = np.zeros_like(samples_matrix)
    for j in range(n_params):
        u[:, j] = marginal_cdfs[j](samples_matrix[:, j])

    # Clip to avoid infinities at boundaries
    u = np.clip(u, 1e-6, 1 - 1e-6)

    # Transform to standard normal
    z = stats.norm.ppf(u)

    # Correlation matrix of the copula
    R = np.corrcoef(z, rowvar=False)

    return R


def threshold_copula(
    R: np.ndarray,
    param_names: list[str],
    threshold: float = 0.05,
) -> tuple[np.ndarray, list[str]]:
    """Zero out negligible correlations and identify copula participants.

    Args:
        R: Full correlation matrix
        param_names: Parameter names corresponding to rows/columns
        threshold: Minimum |correlation| to keep (default 0.05)

    Returns:
        (thresholded R, list of param names that participate in copula)
    """
    R_thresh = R.copy()
    n = R.shape[0]

    # Zero out small off-diagonal entries
    for i in range(n):
        for j in range(n):
            if i != j and abs(R[i, j]) < threshold:
                R_thresh[i, j] = 0.0

    # Identify parameters with any nonzero off-diagonal correlation
    participants = []
    for i in range(n):
        has_correlation = any(abs(R_thresh[i, j]) > 0 for j in range(n) if j != i)
        if has_correlation:
            participants.append(param_names[i])

    return R_thresh, participants


# =============================================================================
# Result assembly
# =============================================================================


def parameterize_posteriors(
    samples: dict[str, np.ndarray],
    targets: list[SubmodelTarget],
    translation_sigmas: Optional[dict[str, tuple[float, dict]]] = None,
    copula_threshold: float = 0.05,
    mcmc_config: Optional[dict] = None,
    parameter_groups: Optional[ParameterGroupsConfig] = None,
) -> dict:
    """Parameterize posterior samples as marginals + Gaussian copula.

    Args:
        samples: {param_name: 1D array} from MCMC
        targets: SubmodelTarget objects (for provenance)
        translation_sigmas: {target_id: (total, breakdown)} for provenance
        copula_threshold: Minimum |correlation| to include in copula
        mcmc_config: MCMC settings for metadata
        parameter_groups: Optional hierarchical group config for metadata

    Returns:
        Result dict with keys: metadata, parameters, copula, translation_sigma,
        and optionally parameter_groups
    """
    # Exclude nuisance parameters and group hyperparameters from output
    nuisance_names = set()
    for target in targets:
        for param in target.calibration.parameters:
            if param.nuisance:
                nuisance_names.add(param.name)

    # Group hyperparameters have __base, __tau, or __delta suffixes
    hyper_names = {k for k in samples if "__base" in k or "__tau" in k or "__delta" in k}

    output_samples = {
        k: v for k, v in samples.items() if k not in nuisance_names and k not in hyper_names
    }
    param_names = sorted(output_samples.keys())

    # Fit marginals
    marginals = fit_marginals(output_samples)

    # Build copula
    if len(param_names) > 1:
        samples_matrix = np.column_stack([output_samples[n] for n in param_names])
        marginal_cdfs = [_build_marginal_cdf(marginals[n]) for n in param_names]
        R = fit_gaussian_copula(samples_matrix, marginal_cdfs)
        R_thresh, copula_params = threshold_copula(R, param_names, copula_threshold)
    else:
        R_thresh = np.array([[1.0]])
        copula_params = []

    # Build parameter-to-target mapping (excluding nuisance)
    param_sources = {}
    for target in targets:
        for param in target.calibration.parameters:
            if not param.nuisance:
                param_sources.setdefault(param.name, []).append(target.target_id)

    # Assemble result
    parameters = []
    for name in param_names:
        fit = marginals[name]
        entry = {
            "name": name,
            "marginal": {
                "distribution": fit.name,
                **fit.params,
                "median": float(fit.median),
                "cv": float(fit.cv),
            },
        }
        if name in param_sources:
            entry["source_targets"] = param_sources[name]
        parameters.append(entry)

    # Copula block — only include participants
    copula_block = None
    if copula_params:
        # Extract submatrix for copula participants
        indices = [param_names.index(p) for p in copula_params]
        R_sub = R_thresh[np.ix_(indices, indices)]
        copula_block = {
            "type": "gaussian",
            "parameters": copula_params,
            "correlation": R_sub.tolist(),
        }

    # Translation sigma provenance
    trans_sigma_block = {}
    if translation_sigmas:
        for target_id, (total, breakdown) in translation_sigmas.items():
            trans_sigma_block[target_id] = {
                "total": float(total),
                "breakdown": {k: float(v) for k, v in breakdown.items()},
            }

    n_samples = len(next(iter(output_samples.values()))) if output_samples else 0
    metadata = {
        "generated_by": "qsp_inference.submodel.parameterizer",
        "n_targets": len(targets),
        "n_parameters": len(param_names),
        "n_samples": n_samples,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    if mcmc_config:
        metadata.update(mcmc_config)

    result = {
        "metadata": metadata,
        "parameters": parameters,
    }
    if copula_block:
        result["copula"] = copula_block
    if trans_sigma_block:
        result["translation_sigma"] = trans_sigma_block

    # Group hyperparameter summaries
    if parameter_groups and parameter_groups.groups:
        groups_block = []
        for group in parameter_groups.groups:
            gid = group.group_id
            group_entry = {
                "group_id": gid,
                "description": group.description,
                "members": [m.name for m in group.members],
            }
            # Base rate posterior
            base_key = f"{gid}__base"
            if base_key in samples:
                base_samples = np.asarray(samples[base_key])
                group_entry["base_posterior"] = {
                    "median": float(np.median(base_samples)),
                    "sd": float(np.std(base_samples)),
                    "ci95": [
                        float(np.percentile(base_samples, 2.5)),
                        float(np.percentile(base_samples, 97.5)),
                    ],
                }
            # Tau posterior
            tau_key = f"{gid}__tau"
            if tau_key in samples:
                tau_samples = np.asarray(samples[tau_key])
                group_entry["tau_posterior"] = {
                    "median": float(np.median(tau_samples)),
                    "sd": float(np.std(tau_samples)),
                    "ci95": [
                        float(np.percentile(tau_samples, 2.5)),
                        float(np.percentile(tau_samples, 97.5)),
                    ],
                }
            groups_block.append(group_entry)
        result["parameter_groups"] = groups_block

    return result


# =============================================================================
# YAML writer
# =============================================================================


def write_priors_yaml(result: dict, path: Path) -> None:
    """Write parameterized posteriors to submodel_priors.yaml.

    Uses ruamel.yaml for clean block formatting.
    """
    import json

    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 120

    # Convert numpy types to native Python via JSON roundtrip
    result = json.loads(json.dumps(result, default=_json_default))

    # Round correlation matrix for readability
    if "copula" in result:
        result["copula"]["correlation"] = [
            [round(v, 4) for v in row] for row in result["copula"]["correlation"]
        ]

    # Round marginal params to 6 SIGNIFICANT FIGURES, not 6 decimal places.
    # Decimal rounding silently annihilates any marginal whose parameters live
    # far below 1: a gamma fitted to a rate in nmol/cell/hour has scale ~1e-11,
    # and round(1.47e-11, 6) is 0.0. A gamma with scale=0 is degenerate, so the
    # log-domain fit in priors/copula_prior.py raises and the parameter silently
    # falls back to its CSV lognormal -- losing the submodel anchor it was
    # inferred to carry. Significant figures keep the same readability without
    # being scale-dependent. (Lognormals were spared only by accident: their
    # mu/sigma are log-space and O(1), so only their unused `median` field
    # underflowed.)
    for param in result["parameters"]:
        m = param["marginal"]
        for key in m:
            if isinstance(m[key], float):
                m[key] = float(f"{m[key]:.6g}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(result, f)


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
