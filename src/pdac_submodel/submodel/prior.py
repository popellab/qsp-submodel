"""
Convert SubmodelTarget YAMLs to prior distributions for SBI.

Pipeline (per-target via process_yaml):
  1. Load YAML + priors CSV
  2. Run single-target NumPyro MCMC (same engine as joint pipeline)
  3. Fit candidate distributions to posterior samples
  4. Report translation sigma breakdown (applied inside MCMC likelihood)

Pipeline (joint via process_targets):
  1. Load all YAMLs + priors CSV
  2. Run joint NumPyro MCMC across all targets (shared parameters sampled once)
  3. Fit marginal distributions + Gaussian copula to posterior
  4. Output submodel_priors.yaml

Translation sigma inflation rubric (8 axes, additive in quadrature, floor=0.15):
  indication_match:        exact=0, related=0.2, proxy=0.5, unrelated=1.0
  species mismatch:        same=0, different=0.3
  source_quality:          primary_human_clinical=0, primary_human_in_vitro=0.1,
                           primary_animal_in_vivo=0.3, primary_animal_in_vitro=0.4,
                           review_article=0.2, textbook=0.3, non_peer_reviewed=0.5
  perturbation_type:       pathological_state=0, physiological_baseline=0.1,
                           pharmacological=0.25, genetic_perturbation=0.4
  tme_compatibility:       high=0, moderate=0.15, low=0.5
  measurement_directness:  direct=0, single_inversion=0.15,
                           steady_state_inversion=0.3, proxy_observable=0.5
  temporal_resolution:     timecourse=0, endpoint_pair=0.1,
                           snapshot_or_equilibrium=0.2
  experimental_system:     clinical_in_vivo=0, animal_in_vivo=0.15, ex_vivo=0.1,
                           in_vitro_coculture=0.15, in_vitro_primary=0.2,
                           in_vitro_cell_line=0.3

These add in quadrature: sigma_total = max(sqrt(sigma_data^2 + sum(sigma_i^2)), 0.15)

Usage (as library)::

    from qsp_inference.submodel.prior import process_yaml
    results = process_yaml(Path("target.yaml"))  # list of dicts, one per parameter

Usage (CLI)::

    python -m qsp_inference.submodel.prior target.yaml
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

from qsp_inference.submodel.parameter_groups import load_parameter_groups
from maple.core.calibration.submodel_target import SubmodelTarget


# ============================================================================
# Translation sigma inflation rubric
# ============================================================================

FLOOR_SIGMA = 0.15  # irreducible model-abstraction uncertainty

INDICATION_SIGMA = {
    "exact": 0.0,
    "related": 0.2,
    "proxy": 0.5,
    "unrelated": 1.0,
}

SPECIES_MISMATCH_SIGMA = 0.3  # added when source != target

SOURCE_QUALITY_SIGMA = {
    "primary_human_clinical": 0.0,
    "primary_human_in_vitro": 0.1,
    "primary_animal_in_vivo": 0.3,
    "primary_animal_in_vitro": 0.4,
    "review_article": 0.2,
    "textbook": 0.3,
    "non_peer_reviewed": 0.5,
}

TME_SIGMA = {
    "high": 0.0,
    "moderate": 0.15,
    "low": 0.5,
}

PERTURBATION_SIGMA = {
    "pathological_state": 0.0,
    "physiological_baseline": 0.1,
    "pharmacological": 0.25,
    "genetic_perturbation": 0.4,
}

DIRECTNESS_SIGMA = {
    "direct": 0.0,
    "single_inversion": 0.15,
    "steady_state_inversion": 0.3,
    "proxy_observable": 0.5,
}

TEMPORAL_SIGMA = {
    "timecourse": 0.0,
    "endpoint_pair": 0.1,
    "snapshot_or_equilibrium": 0.2,
}

SYSTEM_SIGMA = {
    "clinical_in_vivo": 0.0,
    "animal_in_vivo": 0.15,
    "ex_vivo": 0.1,
    "in_vitro_coculture": 0.15,
    "in_vitro_primary": 0.2,
    "in_vitro_cell_line": 0.3,
}


def compute_translation_sigma(sr) -> tuple[float, dict[str, float]]:
    """Compute translation sigma inflation from source_relevance fields.

    Returns (total_sigma, breakdown_dict) where breakdown shows each component.
    Components add in quadrature with a floor of FLOOR_SIGMA.
    """
    components = {}

    components["indication"] = INDICATION_SIGMA.get(sr.indication_match.value, 0.5)

    if sr.species_source != sr.species_target:
        components["species"] = SPECIES_MISMATCH_SIGMA
    else:
        components["species"] = 0.0

    components["quality"] = SOURCE_QUALITY_SIGMA.get(sr.source_quality.value, 0.3)

    components["tme"] = TME_SIGMA.get(sr.tme_compatibility.value, 0.15)

    components["perturbation"] = PERTURBATION_SIGMA.get(sr.perturbation_type.value, 0.2)

    # New axes (None-safe for YAMLs that haven't been updated yet)
    if sr.measurement_directness is not None:
        components["directness"] = DIRECTNESS_SIGMA.get(sr.measurement_directness.value, 0.15)

    if sr.temporal_resolution is not None:
        components["temporal"] = TEMPORAL_SIGMA.get(sr.temporal_resolution.value, 0.1)

    if sr.experimental_system is not None:
        components["system"] = SYSTEM_SIGMA.get(sr.experimental_system.value, 0.15)

    raw = np.sqrt(sum(v**2 for v in components.values()))
    total = max(raw, FLOOR_SIGMA)
    return total, components


@dataclass
class DistFit:
    """Result of fitting a single distribution to samples."""

    name: str  # e.g. "lognormal", "gamma", "invgamma"
    params: dict  # distribution-specific parameters
    aic: float  # Akaike information criterion (lower = better)
    ad_stat: float  # Anderson-Darling statistic (absolute GOF)
    ad_crit_5pct: float  # AD critical value at 5% significance
    ad_pass: bool  # AD stat < critical value at 5%
    median: float  # fitted median
    cv: float  # fitted CV


def _ad_test_samples(
    samples: np.ndarray, cdf_func, dist_name: str = "unknown"
) -> tuple[float, float]:
    """Anderson-Darling test against a fitted CDF.

    Returns (ad_stat, critical_value_5pct).
    Uses subsample of 2000 to give reasonable power without being overpowered.
    Critical values from Stephens (1986) for parameters estimated from data.
    """
    rng = np.random.default_rng(123)
    if len(samples) > 2000:
        samples = rng.choice(samples, 2000, replace=False)
    n = len(samples)
    s = np.sort(samples)
    u = cdf_func(s)
    u = np.clip(u, 1e-15, 1 - 1e-15)
    i = np.arange(1, n + 1)
    ad = -n - np.sum((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1]))) / n
    crit_table = {"lognormal": 0.752, "gamma": 0.786, "invgamma": 0.786}
    crit_5 = crit_table.get(dist_name, 0.752)
    return ad, crit_5


def fit_distributions(samples: np.ndarray) -> list[DistFit]:
    """Fit candidate distributions to bootstrap samples.

    Always tries Normal. For all-positive samples, also tries lognormal,
    gamma, and inverse-gamma. AIC selects the best fit.

    Returns list of DistFit sorted by AIC (best first).
    """
    if len(samples) < 100:
        return []

    fits = []

    # --- Normal (always attempted) ---
    mu_n = np.mean(samples)
    sig_n = np.std(samples, ddof=1)
    if sig_n > 0:
        ll = np.sum(stats.norm.logpdf(samples, loc=mu_n, scale=sig_n))
        aic = -2 * ll + 2 * 2
        ad, crit = _ad_test_samples(
            samples,
            lambda x: stats.norm.cdf(x, loc=mu_n, scale=sig_n),
            "normal",
        )
        fits.append(
            DistFit(
                name="normal",
                params={"mu": mu_n, "sigma": sig_n},
                aic=aic,
                ad_stat=ad,
                ad_crit_5pct=crit,
                ad_pass=ad < crit,
                median=mu_n,
                cv=abs(sig_n / mu_n) if mu_n != 0 else float("inf"),
            )
        )

    # Positive-only distributions require sufficient positive mass
    positive = samples[samples > 0]
    if len(positive) < 100:
        fits.sort(key=lambda f: f.aic)
        return fits

    # --- Lognormal ---
    log_s = np.log(positive)
    mu_ln = np.mean(log_s)
    sig_ln = np.std(log_s, ddof=1)
    if sig_ln > 0:
        ll = np.sum(stats.lognorm.logpdf(positive, s=sig_ln, scale=np.exp(mu_ln)))
        aic = -2 * ll + 2 * 2
        median = np.exp(mu_ln)
        cv = np.sqrt(np.exp(sig_ln**2) - 1)
        ad, crit = _ad_test_samples(
            positive,
            lambda x: stats.lognorm.cdf(x, s=sig_ln, scale=np.exp(mu_ln)),
            "lognormal",
        )
        fits.append(
            DistFit(
                name="lognormal",
                params={"mu": mu_ln, "sigma": sig_ln},
                aic=aic,
                ad_stat=ad,
                ad_crit_5pct=crit,
                ad_pass=ad < crit,
                median=median,
                cv=cv,
            )
        )

    # --- Gamma ---
    try:
        a_gam, _, scale_gam = stats.gamma.fit(positive, floc=0)
        if a_gam > 0 and scale_gam > 0:
            ll = np.sum(stats.gamma.logpdf(positive, a_gam, scale=scale_gam))
            aic = -2 * ll + 2 * 2
            median = stats.gamma.ppf(0.5, a_gam, scale=scale_gam)
            cv = 1.0 / np.sqrt(a_gam)
            ad, crit = _ad_test_samples(
                positive,
                lambda x: stats.gamma.cdf(x, a_gam, scale=scale_gam),
                "gamma",
            )
            fits.append(
                DistFit(
                    name="gamma",
                    params={"shape": a_gam, "scale": scale_gam},
                    aic=aic,
                    ad_stat=ad,
                    ad_crit_5pct=crit,
                    ad_pass=ad < crit,
                    median=median,
                    cv=cv,
                )
            )
    except Exception:
        pass

    # --- Inverse Gamma ---
    try:
        inv_s = 1.0 / positive
        a_ig, _, scale_ig = stats.gamma.fit(inv_s, floc=0)
        if a_ig > 0 and scale_ig > 0:
            ll = np.sum(stats.invgamma.logpdf(positive, a_ig, scale=1.0 / scale_ig))
            aic = -2 * ll + 2 * 2
            median = stats.invgamma.ppf(0.5, a_ig, scale=1.0 / scale_ig)
            cv_ig = np.sqrt(1.0 / (a_ig - 2)) if a_ig > 2 else float("inf")
            ad, crit = _ad_test_samples(
                positive,
                lambda x: stats.invgamma.cdf(x, a_ig, scale=1.0 / scale_ig),
                "invgamma",
            )
            fits.append(
                DistFit(
                    name="invgamma",
                    params={"shape": a_ig, "scale": 1.0 / scale_ig},
                    aic=aic,
                    ad_stat=ad,
                    ad_crit_5pct=crit,
                    ad_pass=ad < crit,
                    median=median,
                    cv=cv_ig,
                )
            )
    except Exception:
        pass

    fits.sort(key=lambda f: f.aic)
    return fits


def process_yaml(yaml_path: Path, priors_csv: Path) -> list[dict]:
    """Process a single SubmodelTarget YAML into prior specifications via MCMC.

    Uses the same NumPyro MCMC engine as the joint pipeline (run_joint_inference)
    with lightweight settings for fast single-target validation. Returns a result
    dict for each parameter in the target.

    Args:
        yaml_path: Path to SubmodelTarget YAML file.
        priors_csv: Path to priors CSV (e.g., pdac_priors.csv).

    Returns a list of dicts, one per parameter in the target. Each dict has keys:
    name, units, target_id, best_dist, all_fits, param_samples, median_data,
    sigma_data, translation_sigma, translation_breakdown, median_prior,
    sigma_prior, mu_prior, cv_data, cv_prior.

    On failure, returns a single-element list with a dict containing: name, error.
    """
    import warnings

    import yaml

    from qsp_inference.submodel.inference import (
        load_priors_from_csv,
        run_joint_inference,
    )

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        target = SubmodelTarget(**data)

    params = target.calibration.parameters
    first_name = params[0].name

    # Load priors, filter to only this target's non-nuisance parameters for fast MCMC
    # (nuisance parameters carry inline priors and are injected by build_target_likelihoods)
    all_priors = load_priors_from_csv(priors_csv)
    non_nuisance_names = {p.name for p in params if not p.nuisance}
    prior_specs = {k: v for k, v in all_priors.items() if k in non_nuisance_names}

    missing = non_nuisance_names - set(prior_specs.keys())
    if missing:
        return [
            {
                "name": first_name,
                "error": f"Parameters not found in priors CSV: {sorted(missing)}",
            }
        ]

    # Run single-target MCMC with lightweight settings
    try:
        samples, diagnostics = run_joint_inference(
            prior_specs,
            [target],
            num_warmup=200,
            num_samples=500,
            num_chains=1,
        )
    except Exception as e:
        return [{"name": first_name, "error": f"MCMC inference failed: {e}"}]

    # Translation sigma: collect per-entry, report the max for diagnostics
    from qsp_inference.submodel.inference import resolve_per_measurement_sigma

    entry_sigmas = []
    for entry in target.calibration.error_model:
        s, bd = resolve_per_measurement_sigma(target, entry.uses_inputs)
        entry_sigmas.append((s, bd))
    if entry_sigmas:
        # Pick the entry with the largest sigma for reporting
        trans_sigma, breakdown = max(entry_sigmas, key=lambda x: x[0])
    else:
        trans_sigma, breakdown = 0.0, {}

    # Fit candidate distributions to each non-nuisance parameter's marginal posterior
    results = []
    for param in params:
        if param.nuisance:
            continue
        pname = param.name
        punits = param.units
        param_samples = samples[pname]

        fits = fit_distributions(param_samples)
        if not fits:
            results.append({"name": pname, "error": "All distribution fits failed"})
            continue

        best = fits[0]

        # Extract lognormal parameters from the posterior fit
        # Note: posterior already includes translation sigma from the MCMC likelihood,
        # so we report the fitted posterior directly without post-hoc inflation.
        if best.name == "lognormal":
            sigma_data = best.params["sigma"]
            mu_data = best.params["mu"]
        else:
            mu_data = np.log(best.median)
            sigma_data = np.sqrt(np.log(1 + best.cv**2))

        # MCMC diagnostics for this parameter
        param_diag = diagnostics["per_param"].get(pname, {})

        results.append(
            {
                "name": pname,
                "units": punits,
                "target_id": target.target_id,
                "best_dist": best,
                "all_fits": fits,
                "param_samples": param_samples,
                "median_data": best.median,
                "sigma_data": sigma_data,
                "translation_sigma": trans_sigma,
                "translation_breakdown": breakdown,
                "median_prior": best.median,
                "sigma_prior": sigma_data,
                "mu_prior": mu_data,
                "cv_data": best.cv,
                "cv_prior": best.cv,
                "mcmc_diagnostics": {
                    "num_divergences": diagnostics["num_divergences"],
                    **param_diag,
                },
            }
        )

    return results


def format_report(result: dict) -> str:
    """Format a process_yaml result dict as a human-readable report string."""
    if "error" in result:
        return f"ERROR: {result['name']}: {result['error']}"

    best = result["best_dist"]
    all_fits = result["all_fits"]

    lines = [
        f"Parameter: {result['name']} ({result['units']})",
        "",
        f"Distribution fits (n={len(all_fits)}):",
        f"  {'dist':<12} {'AIC':>10} {'dAIC':>6} {'AD':>7} {'crit5%':>7} {'AD?':>5} "
        f"{'median':>10} {'CV':>6}",
    ]

    best_aic = all_fits[0].aic
    for f in all_fits:
        daic = f.aic - best_aic
        ad_flag = "PASS" if f.ad_pass else "FAIL"
        lines.append(
            f"  {f.name:<12} {f.aic:>10.1f} {daic:>+6.1f} {f.ad_stat:>7.3f} "
            f"{f.ad_crit_5pct:>7.3f} {ad_flag:>5} {f.median:>10.4g} {f.cv:>6.2f}"
        )

    lines.extend(
        [
            "",
            f"Best: {best.name} (median={best.median:.4g}, CV={best.cv:.2f})",
            f"Translation sigma: {result['translation_sigma']:.3f}",
        ]
    )
    for k, v in result["translation_breakdown"].items():
        if v > 0:
            lines.append(f"  {k}: +{v:.2f}")

    lines.append(
        f"Prior: median={result['median_prior']:.4g}, "
        f"sigma={result['sigma_prior']:.3f}, "
        f"CV={result['cv_prior']:.2f}"
    )

    # MCMC diagnostics
    diag = result.get("mcmc_diagnostics")
    if diag:
        lines.append(
            f"MCMC: n_eff={diag.get('n_eff', 0):.0f}, "
            f"r_hat={diag.get('r_hat', 0):.3f}, "
            f"divergences={diag.get('num_divergences', 0)}"
        )
        contraction = diag.get("contraction")
        z_score = diag.get("z_score")
        if contraction is not None and z_score is not None:
            lines.append(f"  contraction={contraction:.2f}, " f"z_score={z_score:.2f}")

    return "\n".join(lines)


# ============================================================================
# CSV merge & sync
# ============================================================================


def merge_into_priors_csv(results: list[dict], priors_csv: Path) -> tuple[int, int]:
    """Merge YAML-derived priors into a priors CSV.

    Updates existing rows by name; appends new parameters not yet in the CSV.
    Returns (n_updated, n_added).
    """
    import pandas as pd

    df = pd.read_csv(priors_csv)
    n_updated = 0
    n_added = 0

    for r in results:
        mask = df["name"] == r["name"]
        updates = {
            "median": r["median_prior"],
            "units": r["units"],
            "distribution": "lognormal",
            "dist_param1": r["mu_prior"],
            "dist_param2": r["sigma_prior"],
        }
        if mask.any():
            for col, val in updates.items():
                df.loc[mask, col] = val
            n_updated += 1
        else:
            row_data = {
                **updates,
                "name": r["name"],
                "lower_bound": np.nan,
                "upper_bound": np.nan,
            }
            df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
            n_added += 1

    df.to_csv(priors_csv, index=False)
    return n_updated, n_added


def sync_submodel_priors(
    submodel_dir: Path,
    priors_csv: Path,
    glob_pattern: str = "*.yaml",
    verbose: bool = True,
) -> list[dict]:
    """Glob submodel target YAMLs, process each, and merge into priors CSV.

    Also loads submodel_config.yaml from submodel_dir if present.
    Note: parameter groups are only applied during joint inference
    (process_targets), not during per-target sync. The sync pipeline
    processes targets individually for fast validation.

    Returns the list of successfully processed result dicts.
    """
    yaml_files = sorted(submodel_dir.glob(glob_pattern))
    # Exclude submodel_config.yaml from target list
    yaml_files = [f for f in yaml_files if f.name != "submodel_config.yaml"]
    if not yaml_files:
        if verbose:
            print(f"  No submodel target YAMLs matching {glob_pattern} in {submodel_dir}")
        return []

    results = []
    for yf in yaml_files:
        try:
            param_results = process_yaml(yf, priors_csv=priors_csv)
            for r in param_results:
                if "error" not in r:
                    results.append(r)
                elif verbose:
                    print(f"  SKIP {yf.name} ({r['name']}): {r['error']}")
        except Exception as e:
            if verbose:
                print(f"  SKIP {yf.name}: {e}")

    if results:
        n_upd, n_add = merge_into_priors_csv(results, priors_csv)
        if verbose:
            names = [r["name"] for r in results]
            print(
                f"  Merged {len(results)} submodel priors into {priors_csv}: "
                f"{n_upd} updated, {n_add} added ({', '.join(names)})"
            )

    return results


# ============================================================================
# Plotting
# ============================================================================


def _dist_pdf(fit: DistFit, x: np.ndarray) -> np.ndarray:
    """Evaluate the fitted PDF at x values."""
    if fit.name == "lognormal":
        return stats.lognorm.pdf(x, s=fit.params["sigma"], scale=np.exp(fit.params["mu"]))
    elif fit.name == "gamma":
        return stats.gamma.pdf(x, fit.params["shape"], scale=fit.params["scale"])
    elif fit.name == "invgamma":
        return stats.invgamma.pdf(x, fit.params["shape"], scale=fit.params["scale"])
    return np.zeros_like(x)


# ============================================================================
# Joint inference orchestrator
# ============================================================================


def process_targets(
    priors_csv: Path,
    yaml_paths: list[Path],
    reference_db: Optional[dict] = None,
    output_dir: Optional[Path] = None,
    plot: bool = False,
    export_csv: Optional[Path] = None,
    parameter_groups_path: Optional[Path] = None,
    **mcmc_kwargs,
) -> dict:
    """Run joint inference across SubmodelTargets and produce parameterized priors.

    Args:
        priors_csv: Path to starting priors CSV (pdac_priors.csv)
        yaml_paths: Paths to SubmodelTarget YAML files
        reference_db: Optional reference values for ReferenceRef resolution
        output_dir: Where to write submodel_priors.yaml and plots
        plot: Whether to generate diagnostic plots
        export_csv: Optional path to write updated CSV
        parameter_groups_path: Optional path to submodel_config.yaml for hierarchical pooling
        **mcmc_kwargs: Passed to run_joint_inference (num_warmup, num_samples, etc.)

    Returns:
        Result dict from parameterize_posteriors
    """
    import warnings

    import yaml as pyyaml

    from qsp_inference.submodel.parameterizer import (
        parameterize_posteriors,
        write_priors_yaml,
    )
    from qsp_inference.submodel.inference import (
        load_priors_from_csv,
        run_joint_inference,
    )

    # 1. Load starting priors
    all_prior_specs = load_priors_from_csv(priors_csv)
    print(f"Loaded {len(all_prior_specs)} priors from {priors_csv}")

    # 1b. Load parameter groups if provided
    param_groups = None
    if parameter_groups_path is not None:
        param_groups = load_parameter_groups(parameter_groups_path)
        if param_groups.groups:
            print(
                f"Loaded {len(param_groups.groups)} parameter groups "
                f"({len(param_groups.all_grouped_params)} params)"
            )

    # 2. Load and validate targets
    targets = []
    for p in yaml_paths:
        with open(p) as f:
            data = pyyaml.safe_load(f)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            target = SubmodelTarget(**data)
        targets.append(target)
    print(f"Loaded {len(targets)} SubmodelTargets")

    # 3. Filter priors to only non-nuisance parameters referenced by targets
    #    (nuisance parameters carry inline priors and are injected by build_target_likelihoods)
    target_params = set()
    for target in targets:
        for param in target.calibration.parameters:
            if not param.nuisance:
                target_params.add(param.name)
    # Also include grouped params — they may not appear in any target but
    # need CSV priors for resolve_base_prior and hierarchical sampling
    if param_groups:
        target_params |= param_groups.all_grouped_params
    prior_specs = {k: v for k, v in all_prior_specs.items() if k in target_params}
    n_nuisance = sum(
        1
        for t in targets
        for p in t.calibration.parameters
        if p.nuisance and p.name not in prior_specs
    )
    print(
        f"Fitting {len(prior_specs) + n_nuisance} parameters "
        f"({len(prior_specs)} from CSV, {n_nuisance} nuisance)"
    )

    # 4. Run joint inference
    samples, _diagnostics = run_joint_inference(
        prior_specs,
        targets,
        reference_db,
        parameter_groups=param_groups,
        **mcmc_kwargs,
    )

    # 5. Collect translation sigmas for provenance (max per-entry per target)
    from qsp_inference.submodel.inference import resolve_per_measurement_sigma

    translation_sigmas = {}
    for target in targets:
        entry_sigmas = [
            resolve_per_measurement_sigma(target, entry.uses_inputs)
            for entry in target.calibration.error_model
        ]
        if entry_sigmas:
            translation_sigmas[target.target_id] = max(entry_sigmas, key=lambda x: x[0])
        else:
            translation_sigmas[target.target_id] = (0.0, {})

    # 6. Parameterize posteriors
    mcmc_config = {
        "num_warmup": mcmc_kwargs.get("num_warmup", 1000),
        "num_samples": mcmc_kwargs.get("num_samples", 5000),
        "num_chains": mcmc_kwargs.get("num_chains", 4),
    }
    result = parameterize_posteriors(
        samples,
        targets,
        translation_sigmas,
        mcmc_config=mcmc_config,
        parameter_groups=param_groups,
    )

    # 7. Write output
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = output_dir / "submodel_priors.yaml"
        write_priors_yaml(result, yaml_path)
        print(f"\nPriors written to {yaml_path}")

    # 8. Export CSV if requested
    if export_csv:
        _export_marginals_csv(result, export_csv)
        print(f"CSV exported to {export_csv}")

    # 9. Plot if requested
    if plot and output_dir:
        plot_joint_posteriors(samples, result, output_dir)

    return result


def _export_marginals_csv(result: dict, csv_path: Path):
    """Export marginal fits to CSV format compatible with pdac_priors.csv."""
    import pandas as pd

    rows = []
    for param in result["parameters"]:
        m = param["marginal"]
        dist = m["distribution"]
        if dist == "lognormal":
            p1, p2 = m["mu"], m["sigma"]
        elif dist == "gamma":
            p1, p2 = m["shape"], m["scale"]
        elif dist == "invgamma":
            p1, p2 = m["shape"], m["scale"]
        else:
            p1, p2 = 0, 0

        rows.append(
            {
                "name": param["name"],
                "median": m["median"],
                "units": "",
                "distribution": dist,
                "dist_param1": p1,
                "dist_param2": p2,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)


def plot_joint_posteriors(
    samples: dict[str, np.ndarray],
    result: dict,
    output_dir: Path,
):
    """Plot posterior diagnostics: marginal histograms + corner plot."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    param_entries = result["parameters"]
    n = len(param_entries)

    # --- Marginal plots ---
    fig, axes = plt.subplots(n, 1, figsize=(8, 3.0 * n), squeeze=False)

    for i, entry in enumerate(param_entries):
        ax = axes[i, 0]
        name = entry["name"]
        s = samples[name]
        m = entry["marginal"]

        # Histogram
        positive = s[s > 0]
        ax.hist(positive, bins=80, density=True, alpha=0.35, color="#888", edgecolor="none")

        # Fitted marginal PDF
        lo, hi = np.percentile(positive, [0.5, 99.5])
        x = np.linspace(lo, hi, 500)

        fit = DistFit(
            name=m["distribution"],
            params={k: v for k, v in m.items() if k not in ("distribution", "median", "cv")},
            aic=0,
            ad_stat=0,
            ad_crit_5pct=0,
            ad_pass=True,
            median=m["median"],
            cv=m["cv"],
        )
        ax.plot(x, _dist_pdf(fit, x), "-", lw=2, color="#2563eb", label=f'{m["distribution"]}')

        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.set_title(f"{name} — {m['distribution']} (median={m['median']:.3g})")
        ax.legend(fontsize=8)
        ax.set_xlim(lo, hi)

    fig.tight_layout()
    path = output_dir / "posterior_marginals.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Marginal plots saved to {path}")

    # --- Corner plot for copula participants ---
    if "copula" in result and len(result["copula"]["parameters"]) >= 2:
        cop_params = result["copula"]["parameters"]
        nc = len(cop_params)
        fig, axes = plt.subplots(nc, nc, figsize=(3 * nc, 3 * nc))

        for i in range(nc):
            for j in range(nc):
                ax = axes[i, j] if nc > 1 else axes
                si = samples[cop_params[i]]
                sj = samples[cop_params[j]]

                if i == j:
                    ax.hist(si, bins=50, density=True, alpha=0.5, color="#2563eb")
                    ax.set_xlabel(cop_params[i] if i == nc - 1 else "")
                elif i > j:
                    ax.scatter(sj, si, s=1, alpha=0.1, color="#888")
                    ax.set_xlabel(cop_params[j] if i == nc - 1 else "")
                    ax.set_ylabel(cop_params[i] if j == 0 else "")
                else:
                    ax.axis("off")

        fig.suptitle("Posterior correlations (copula participants)")
        fig.tight_layout()
        path = output_dir / "posterior_corner.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Corner plot saved to {path}")


# ============================================================================
# CLI
# ============================================================================


def main():
    """CLI entry point for yaml_to_prior."""
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  maple-yaml-to-prior --priors CSV [--output DIR] [--plot] "
            "[--export-csv PATH] [--reference-db YAML] <yaml_files_or_dirs...>"
        )
        sys.exit(1)

    # Parse flags
    argv = sys.argv[1:]
    do_plot = "--plot" in argv
    argv = [a for a in argv if a != "--plot"]

    priors_csv = None
    if "--priors" in argv:
        idx = argv.index("--priors")
        priors_csv = Path(argv[idx + 1])
        argv = argv[:idx] + argv[idx + 2 :]

    output_dir = None
    if "--output" in argv:
        idx = argv.index("--output")
        output_dir = Path(argv[idx + 1])
        argv = argv[:idx] + argv[idx + 2 :]

    export_csv = None
    if "--export-csv" in argv:
        idx = argv.index("--export-csv")
        export_csv = Path(argv[idx + 1])
        argv = argv[:idx] + argv[idx + 2 :]

    reference_db_path = None
    if "--reference-db" in argv:
        idx = argv.index("--reference-db")
        reference_db_path = Path(argv[idx + 1])
        argv = argv[:idx] + argv[idx + 2 :]

    parameter_groups_path = None
    if "--parameter-groups" in argv:
        idx = argv.index("--parameter-groups")
        parameter_groups_path = Path(argv[idx + 1])
        argv = argv[:idx] + argv[idx + 2 :]

    if not priors_csv:
        print("ERROR: --priors CSV is required.")
        sys.exit(1)

    # Collect YAML files
    yaml_files = []
    for arg in argv:
        p = Path(arg)
        if p.is_dir():
            yaml_files.extend(sorted(p.glob("*.yaml")))
        else:
            yaml_files.append(p)

    # Load reference DB if provided
    reference_db = None
    if reference_db_path:
        import yaml as pyyaml

        with open(reference_db_path) as f:
            reference_db = pyyaml.safe_load(f)

    print("=" * 80)
    print("JOINT SUBMODEL PRIOR INFERENCE")
    print("=" * 80)
    process_targets(
        priors_csv=priors_csv,
        yaml_paths=yaml_files,
        reference_db=reference_db,
        output_dir=output_dir,
        plot=do_plot,
        export_csv=export_csv,
        parameter_groups_path=parameter_groups_path,
    )
