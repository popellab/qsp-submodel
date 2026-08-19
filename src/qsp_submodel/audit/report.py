#!/usr/bin/env python3
"""Parameter coverage audit for QSP models.

Auto-detects available data and reports:
- Stage 0: Prior coverage, submodel target coverage, PRCC-weighted priority
- Stage 1: Submodel posterior contraction (when inference results exist)
- Stage 2: NPE posterior shifts vs submodel posteriors (when SBI results exist)
- Clinical predictive uncertainty: CI95 widths for PPC clinical endpoints
  (OS/DFS landmarks, RECIST, tumor size, TTP) and Spearman attribution of
  remaining spread back to parameters — surfaces the extraction priorities
  that would shrink predictive uncertainty. Emits only when the promoted
  stage-2 run has ``posterior_predictive_clinical.csv`` alongside its
  posterior samples.

Usage:
    python -m qsp_submodel.audit.report --project-root /path/to/project
    python -m qsp_submodel.audit.report --project-root . --output report.md
"""

import csv
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module=r"maple\.core\.calibration")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic")
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml



@dataclass
class AuditConfig:
    """Paths configuration for the parameter audit report.

    All paths are resolved relative to project_root if not absolute.
    """

    project_root: Path
    priors_csv: Path | None = None
    cross_model_csv: Path | None = None
    submodel_dir: Path | None = None
    param_groups: Path | None = None
    prcc_csv: Path | None = None
    prcc_matrix_csv: Path | None = None
    prcc_pvalue_matrix_csv: Path | None = None
    compare_cache: Path | None = None
    sbi_dir: Path | None = None
    sbi_run_path: Path | None = None
    submodel_priors_yaml: Path | None = None
    model_names: list[str] = field(
        default_factory=lambda: ["TNBC", "CRC", "UM", "NSCLC", "HCC", "PDAC"]
    )

    def __post_init__(self):
        root = Path(self.project_root).resolve()
        self.project_root = root
        if self.priors_csv is None:
            self.priors_csv = root / "parameters" / "pdac_priors.csv"
        if self.cross_model_csv is None:
            self.cross_model_csv = root / "parameters" / "cross_model_parameters.csv"
        if self.submodel_dir is None:
            self.submodel_dir = root / "calibration_targets" / "submodel_targets"
        if self.param_groups is None:
            self.param_groups = self.submodel_dir / "submodel_config.yaml"
        if self.prcc_csv is None:
            self.prcc_csv = root / "results" / "prcc_sensitivity" / "aggregate_parameter_ranking.csv"
        if self.prcc_matrix_csv is None:
            self.prcc_matrix_csv = self.prcc_csv.parent / "prcc_matrix.csv"
        if self.prcc_pvalue_matrix_csv is None:
            self.prcc_pvalue_matrix_csv = self.prcc_csv.parent / "prcc_pvalue_matrix.csv"
        if self.compare_cache is None:
            self.compare_cache = self.submodel_dir / ".compare_cache"
        if self.sbi_dir is None:
            self.sbi_dir = root / "figures"
        if self.sbi_run_path is not None:
            p = Path(self.sbi_run_path)
            self.sbi_run_path = p if p.is_absolute() else (root / p).resolve()
        if self.submodel_priors_yaml is None:
            self.submodel_priors_yaml = self.submodel_dir / "submodel_priors.yaml"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_priors(path: Path) -> dict:
    """Load pdac_priors.csv → {name: {median, sigma, mu, units, distribution}}."""
    params = {}
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6 or not row[0].strip():
                continue
            name = row[0].strip()
            try:
                median = float(row[1])
                mu = float(row[4])
                sigma = float(row[5])
            except (ValueError, IndexError):
                continue
            params[name] = {
                "median": median,
                "units": row[2].strip(),
                "distribution": row[3].strip(),
                "mu": mu,
                "sigma": sigma,
            }
    return params






def load_submodel_targets(submodel_dir: Path) -> dict:
    """Scan submodel target YAMLs → {param_name: [target_ids]}."""
    coverage = defaultdict(list)
    for yaml_path in sorted(submodel_dir.glob("*_deriv*.yaml")):
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
        except Exception:
            continue
        if not data or "calibration" not in data:
            continue
        target_id = data.get("target_id", yaml_path.stem)
        for p in data["calibration"].get("parameters", []):
            name = p.get("name", "")
            if not p.get("nuisance", False) and name:
                coverage[name].append(target_id)
    return dict(coverage)




def load_parameter_groups(path: Path) -> dict:
    """Load submodel_config.yaml → {param_name: group_id}."""
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    membership = {}
    for group in data.get("groups", []):
        gid = group.get("group_id", "unknown")
        for member in group.get("members", []):
            membership[member["name"]] = gid
    return membership









def load_compare_results(cache_dir: Path, priors: dict | None = None) -> dict:
    """Build compare_results directly from component cache files.

    All statistics (sigma, contraction, z_score, median, cv) are computed
    from raw posterior samples — no parametric fits involved. This avoids
    pathological values (inf sigma, -inf contraction) from failed fits.

    Args:
        cache_dir: Path to .compare_cache/ directory.
        priors: Prior dict from load_priors(). Needed for contraction and
            z_score computation. If None, contraction/z_score are omitted.
    """
    import numpy as np

    if not cache_dir.exists():
        raise FileNotFoundError(
            f"{cache_dir} not found. Run inference first:\n"
            f"  python scripts/parameter_audit.py"
        )

    parameters = {}
    total_divergences = 0
    ppc_observables = []
    ppc_n_total = 0
    ppc_n_covered = 0

    for path in sorted(cache_dir.glob("comp_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        samples = data.get("samples", {})
        diag = data.get("diag", {})
        per_param = diag.get("per_param", {})

        total_divergences += diag.get("num_divergences", 0)
        ppc_n_total += diag.get("ppc_n_total", 0)
        ppc_n_covered += diag.get("ppc_n_covered", 0)
        if "ppc_observables" in diag:
            ppc_observables.extend(diag["ppc_observables"])

        for pname, samps in samples.items():
            arr = np.array(samps, dtype=float)
            pos = arr[arr > 0]
            if len(pos) < 10:
                continue

            # All stats from raw samples
            median = float(np.median(pos))
            log_samps = np.log(pos)
            post_sigma = float(np.std(log_samps))
            post_mu = float(np.mean(log_samps))
            mean = float(np.mean(pos))
            cv = float(np.std(pos) / mean) if mean > 0 else None

            # Contraction and z_score require prior info
            contraction = None
            z_score = None
            if priors and pname in priors:
                prior_sigma = priors[pname]["sigma"]
                prior_mu = priors[pname]["mu"]
                if prior_sigma > 0:
                    contraction = 1.0 - (post_sigma / prior_sigma) ** 2
                    z_score = abs(post_mu - prior_mu) / prior_sigma

            pp = per_param.get(pname, {})
            entry = {
                "joint": {
                    "median": median,
                    "sigma": post_sigma,
                    "cv": cv,
                    "contraction": contraction,
                    "z_score": z_score,
                },
                "mcmc_diagnostics": {
                    "n_eff": pp.get("n_eff"),
                    "r_hat": pp.get("r_hat"),
                },
            }

            flags = []
            if contraction is not None and contraction < 0.1:
                flags.append("weak_contraction")
            if z_score is not None and z_score > 2:
                flags.append("high_z_score")
            if flags:
                entry["flags"] = flags

            parameters[pname] = entry

    return {
        "metadata": {
            "num_divergences": total_divergences,
            "ppc_n_total": ppc_n_total,
            "ppc_n_covered": ppc_n_covered,
        },
        "parameters": parameters,
        "ppc_observables": ppc_observables,
    }






# ---------------------------------------------------------------------------
# MCMC sample loading and plotting
# ---------------------------------------------------------------------------




def load_joint_samples(cache_dir: Path) -> dict | None:
    """Load posterior samples from component cache files.

    Merges samples across all component_*.json files.
    Returns {param_name: list[float]} or None if no samples available.

    NOTE: This flat view loses component-membership information. Callers
    that need to group params by their inference component should use
    :func:`load_joint_samples_by_component` instead — see the
    docstring there for the bug this fixes.
    """
    if not cache_dir.exists():
        return None
    all_samples = {}
    # Load from component caches (current format)
    for path in sorted(cache_dir.glob("comp_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        samples = data.get("samples", {})
        for pname, samps in samples.items():
            all_samples[pname] = samps
    # Also check legacy joint_*.json format
    for path in sorted(cache_dir.glob("joint_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        samples = data.get("joint_samples", {})
        for pname, samps in samples.items():
            if pname not in all_samples:
                all_samples[pname] = samps
    return all_samples if all_samples else None


def load_joint_samples_by_component(cache_dir: Path) -> dict[str, dict] | None:
    """Load posterior samples grouped by inference component.

    Returns ``{component_id: {param_name: list[float]}}`` where
    ``component_id`` is the cache file basename (e.g. ``"comp_a3f2b1c4"``)
    or ``"joint_<id>"`` for legacy joint caches. Each top-level entry
    corresponds to one independent submodel inference run, so params from
    different entries are statistically independent given the prior — the
    audit's composite copula must therefore be block-diagonal across
    components.

    The previous flat :func:`load_joint_samples` collapsed this structure,
    which led ``_write_submodel_priors`` to bucket params by raw sample
    length as a (broken) proxy for component membership and fit a Gaussian
    copula across row-aligned independent samples. When NPE samplings
    share RNG state across components, that row-alignment is comonotonic
    coupling and the inferred copula entries spuriously approach |r| ≈ 1.

    Returns ``None`` if no caches are present.
    """
    if not cache_dir.exists():
        return None
    by_component: dict[str, dict] = {}
    for path in sorted(cache_dir.glob("comp_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        samples = data.get("samples", {})
        if not samples:
            continue
        by_component[path.stem] = {k: list(v) for k, v in samples.items()}
    # Legacy joint_*.json format: each file is its own joint inference,
    # so it's a single component with multiple correlated params.
    for path in sorted(cache_dir.glob("joint_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        samples = data.get("joint_samples", {})
        if not samples:
            continue
        by_component[path.stem] = {k: list(v) for k, v in samples.items()}
    return by_component if by_component else None


def load_population_samples_by_component(cache_dir: Path) -> dict[str, dict] | None:
    """Load the population-scale posterior samples grouped by component.

    Mirrors :func:`load_joint_samples_by_component` but reads the
    ``population_samples`` block written by ``run_comparison(
    run_population_pass=True)`` — the Option-A second-pass posteriors that
    condition on genuine across-unit spread rather than the SEM-scale
    center. These feed the audit's parallel population (ω) copula block.

    Only components produced with the population pass carry this block, so a
    ``comp_*.json`` without ``population_samples`` is skipped. Returns
    ``None`` if no component carries population samples (i.e. the population
    pass was never run) — the audit then emits only the center block.
    """
    if not cache_dir.exists():
        return None
    by_component: dict[str, dict] = {}
    for path in sorted(cache_dir.glob("comp_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        samples = data.get("population_samples", {})
        if not samples:
            continue
        by_component[path.stem] = {k: list(v) for k, v in samples.items()}
    return by_component if by_component else None


def load_freshness_by_component(cache_dir: Path) -> dict[str, dict]:
    """Load the freshness manifest stamped onto each ``comp_*.json``.

    Returns ``{component_id: freshness_dict}``. Components written before
    the freshness feature was added are simply absent from the result —
    callers should treat that as "unknown freshness" and re-run inference.
    """
    if not cache_dir.exists():
        return {}
    out: dict[str, dict] = {}
    for path in sorted(cache_dir.glob("comp_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        freshness = data.get("freshness")
        if freshness:
            out[path.stem] = freshness
    return out


def export_posterior_medians(cache_dir: Path) -> dict:
    """Export posterior medians from component cache files.

    Reads all comp_*.json in *cache_dir*, filters out nuisance parameters
    (sigma_*, *__z, *__base), and computes the median of each QSP
    parameter's posterior samples.

    Returns a dict suitable for JSON serialization::

        {
            "_meta": {
                "generated_at": "2026-04-10T...",
                "source_hash": "a1b2c3...",
                "n_params": 189,
            },
            "medians": {"param_name": float_value, ...},
        }

    The source_hash is a SHA-256 digest (16 hex chars) of the cache file
    contents, allowing downstream consumers to detect staleness.
    """
    import hashlib
    from datetime import datetime, timezone

    import numpy as np

    all_samples = load_joint_samples(cache_dir)
    if not all_samples:
        raise FileNotFoundError(
            f"No posterior samples found in {cache_dir}. "
            "Run parameter audit first: python scripts/parameter_audit.py"
        )

    # Filter nuisance params
    def _is_qsp_param(name: str) -> bool:
        if name.startswith("sigma_"):
            return False
        if name.endswith("__z"):
            return False
        if "__base" in name:
            return False
        return True

    medians = {}
    for pname, samps in sorted(all_samples.items()):
        if not _is_qsp_param(pname):
            continue
        arr = np.array(samps, dtype=float)
        pos = arr[arr > 0]
        if len(pos) < 10:
            continue
        medians[pname] = float(np.median(pos))

    # Source hash: deterministic digest of cache file contents
    h = hashlib.sha256()
    for path in sorted(cache_dir.glob("comp_*.json")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    source_hash = h.hexdigest()[:16]

    return {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_hash": source_hash,
            "n_params": len(medians),
        },
        "medians": medians,
    }


# ---------------------------------------------------------------------------
# Priority scoring
# ---------------------------------------------------------------------------


_PARAM_TYPE_PATTERNS: list[tuple[str, str]] = [
    (r"_50|EC50|ec50",                           "EC50"),
    (r"^k_.*sec$|^k_.*Sec$|^k_.*_sec_",         "secretion"),
    (r"^k_.*deg$|^k_.*_deg$|^k_.*clear$",       "degradation"),
    (r"^k_.*death$|^k_.*_death$",                "death"),
    (r"^k_.*pro$|^k_.*prolif$|^k_.*_pro_",      "proliferation"),
    (r"^k_.*rec$|^k_.*_rec$|^q_",               "trafficking"),
    (r"^k_.*pol$|^k_.*_pol$",                    "polarization"),
    (r"^k_.*act$|^k_.*_act$|^k_.*activation",   "activation"),
    (r"^k_.*kill$|^k_.*phago$",                  "killing"),
    (r"^k_.*exh",                                "exhaustion"),
    (r"^f_|^rho_|^r_",                           "fraction"),
    (r"^n_|^N_|^N_div_base$",                    "count"),
    (r"^d_|^phi_|^E_",                           "biophysical"),
]










# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------














# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------














# ---------------------------------------------------------------------------
# Submodel priors export
# ---------------------------------------------------------------------------


def _marginals_and_copula(
    samples_by_component: dict[str, dict],
    targets: dict[str, list[str]],
    groups: dict[str, str],
    copula_threshold: float,
    cascade_cuts: dict[str, list[str]] | None = None,
    comp_targets: dict[str, set] | None = None,
) -> tuple[list[dict], dict | None, int]:
    """Fit per-param marginals + a block-diagonal Gaussian copula from component samples.

    Shared by the center pass and the population pass so both produce identically
    structured ``parameters`` / ``copula`` blocks. Returns
    ``(parameters, copula_block_or_None, n_samples)``; ``parameters`` is empty when
    there are no usable samples. Copula is block-diagonal by inference component
    (params from different components are independent — off-diagonal forced to 0).
    """
    import numpy as np

    from qsp_submodel.submodel.parameterizer import (
        empirical_log_grid,
        rank_gaussian_copula,
        threshold_copula,
    )

    cascade_cuts = cascade_cuts or {}
    comp_targets = comp_targets or {}
    if cascade_cuts and not comp_targets:
        raise ValueError(
            "cascade_cuts needs comp_targets to resolve against. Without the "
            "per-component target list the cut cannot be told from the view it "
            "exists to discard, and the filter would silently pass everything."
        )

    output_samples: dict[str, np.ndarray] = {}
    param_to_component: dict[str, str] = {}
    for comp_id, comp_samples in samples_by_component.items():
        for k, v in comp_samples.items():
            # "__" strips the non-centered internals (__z, __base, __tau, __delta).
            # Keep a param if a target names it directly OR it is a member of a
            # hierarchical group: a group member with no target of its own is
            # precisely the case partial pooling exists to serve — its posterior is
            # the group mean shrunk by the ANCHORED members. Requiring `k in targets`
            # dropped those members here, so they silently fell back to their bare
            # CSV priors and the pooling was computed and then thrown away.
            if "__" in k or (k not in targets and k not in groups):
                continue
            # A cut parameter appears in several components by construction, so
            # plain assignment kept whichever came last in dict order. That was
            # the downstream view for 3 of the 7 cuts, including one the config
            # cut for SBC bias. Only the declared upstream is taken now.
            if k in cascade_cuts:
                if not (comp_targets.get(comp_id, set()) & set(cascade_cuts[k])):
                    continue
            output_samples[k] = np.asarray(v)
            param_to_component[k] = comp_id

    unresolved = [k for k in cascade_cuts if k in
                  {p for s in samples_by_component.values() for p in s}
                  and k not in output_samples]
    if unresolved:
        print(f"WARNING: cascade cuts with no upstream component: {sorted(unresolved)}")

    if not output_samples:
        return [], None, 0

    param_names = sorted(output_samples.keys())
    grids = {n: empirical_log_grid(output_samples[n]) for n in param_names}

    if len(param_names) > 1:
        from collections import defaultdict

        component_groups: dict[str, list[str]] = defaultdict(list)
        for n in param_names:
            component_groups[param_to_component[n]].append(n)

        name_to_idx = {n: i for i, n in enumerate(param_names)}
        R = np.eye(len(param_names))

        for group_names in component_groups.values():
            if len(group_names) < 2:
                continue
            # Ranks, not the fitted CDFs. A marginal that fits poorly makes
            # F(x) non-uniform, and the correlation then absorbs the marginal's
            # error as if it were dependence.
            block_matrix = np.column_stack([output_samples[n] for n in group_names])
            R_block = rank_gaussian_copula(block_matrix)
            for bi, ni in enumerate(group_names):
                for bj, nj in enumerate(group_names):
                    R[name_to_idx[ni], name_to_idx[nj]] = R_block[bi, bj]

        R_thresh, copula_params = threshold_copula(R, param_names, copula_threshold)
    else:
        R_thresh = np.array([[1.0]])
        copula_params = []

    parameters = []
    for name in param_names:
        s_pos = np.asarray(output_samples[name], dtype=float)
        s_pos = s_pos[s_pos > 0]
        # Summaries from the samples, not from a fitted family, so a reader that
        # wants one number gets the posterior's own.
        entry = {
            "name": name,
            "marginal": {
                **grids[name],
                "median": float(np.median(s_pos)),
                "cv": float(np.std(s_pos, ddof=1) / np.mean(s_pos)),
            },
        }
        if name in targets:
            entry["source_targets"] = targets[name]
        if name in groups:
            entry["group"] = groups[name]
        parameters.append(entry)

    copula_block = None
    if copula_params:
        indices = [param_names.index(p) for p in copula_params]
        R_sub = R_thresh[np.ix_(indices, indices)]
        copula_block = {
            "type": "gaussian",
            "parameters": copula_params,
            "correlation": R_sub.tolist(),
        }

    n_samples = len(next(iter(output_samples.values())))
    return parameters, copula_block, n_samples


def _write_submodel_priors(
    joint_samples_by_component: dict[str, dict],
    targets: dict[str, list[str]],
    groups: dict[str, str],
    output_path: Path,
    copula_threshold: float = 0.05,
    freshness_by_component: dict[str, dict] | None = None,
    population_samples_by_component: dict[str, dict] | None = None,
    cascade_cuts: dict[str, list[str]] | None = None,
    comp_targets: dict[str, set] | None = None,
) -> None:
    """Write submodel_priors.yaml, plus the .npz sidecar holding its grids.

    Marginals are the posterior's own shape as a quantile grid on ``log theta``.
    The grids live in ``<output_path stem>.npz`` and the YAML keeps the part a
    person reads, with a sha256 of the sidecar so drift is detectable.

    The copula block is constructed **block-diagonally per inference
    component**: params produced by the same ``comp_*.json`` cache file
    share a copula block (their joint samples are genuine paired draws
    from the same NPE/MCMC posterior); params from different cache files
    are forced to zero off-diagonal correlation, since they were inferred
    independently and their per-row alignment in the cache files is an
    artifact of RNG state, not a real dependency.

    Args:
        joint_samples_by_component: ``{component_id: {param: list[float]}}``
            from :func:`load_joint_samples_by_component`. Each top-level
            key is a single submodel inference run.
        targets: {param_name: [target_ids]} from load_submodel_targets
        groups: {param_name: group_id} from load_parameter_groups
        output_path: Where to write submodel_priors.yaml
        copula_threshold: Minimum |correlation| to include in copula
    """
    from qsp_submodel.submodel.parameterizer import (
        split_grids_to_sidecar,
        write_priors_yaml,
    )

    # Center-scale marginals + copula (the flat-SBI prior). See _marginals_and_copula
    # for the block-diagonal-by-component construction.
    parameters, copula_block, n_samples = _marginals_and_copula(
        joint_samples_by_component, targets, groups, copula_threshold,
        cascade_cuts, comp_targets
    )
    if not parameters:
        print("No posterior samples to parameterize — skipping submodel_priors.yaml")
        return

    param_names = [p["name"] for p in parameters]

    metadata: dict = {
        "generated_by": "qsp_submodel.audit.report",
        "n_parameters": len(param_names),
        "n_samples": n_samples,
    }
    if freshness_by_component:
        # Embed every component fingerprint we have on disk, not just those
        # whose params won the ``param_to_component`` last-write-wins race.
        # When a parameter appears in multiple cache files via cascade
        # propagation, only one comp_id ends up in ``param_to_component``
        # and the upstream cascade comps would otherwise be silently
        # dropped — but their target YAML / prior CSV / config slice
        # still drives the final posterior, so a checker needs to see
        # them. ``_cleanup_orphaned_caches`` has already pruned anything
        # that isn't an active component, so the dict is the right
        # superset.
        metadata["freshness"] = {
            cid: freshness_by_component[cid]
            for cid in sorted(freshness_by_component)
        }
    sidecar = split_grids_to_sidecar(parameters, output_path)
    if sidecar:
        metadata["grid_sidecar"] = sidecar

    result = {
        "metadata": metadata,
        "parameters": parameters,
    }
    if copula_block:
        result["copula"] = copula_block

    # Population-scale block (Option A): a parallel marginals + copula fit from the
    # population pass (run_joint_inference(population_scale=True)), where params with a
    # population observed_distribution are fit against the declared spread. This is the
    # omega prior for hierarchical / virtual-patient inference; the center block above
    # stays the SEM-scale flat-SBI prior. Only params that actually differ appear here;
    # the hierarchical runner falls back to the center marginal / a wide default otherwise.
    if population_samples_by_component is not None:
        pop_parameters, pop_copula, pop_n = _marginals_and_copula(
            population_samples_by_component, targets, groups, copula_threshold,
            cascade_cuts, comp_targets
        )
        if pop_parameters:
            population_block: dict = {"n_samples": pop_n, "parameters": pop_parameters}
            if pop_copula:
                population_block["copula"] = pop_copula
            result["population"] = population_block
            metadata["has_population_block"] = True

    write_priors_yaml(result, output_path)
    print(f"Submodel priors written to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run_inference(config: AuditConfig, invalidate_params=None, run_population_pass=False):
    """Run component-wise inference to generate/update cached results."""
    from qsp_submodel.submodel.comparison import run_comparison

    if invalidate_params:
        print(f"Invalidating components containing: {invalidate_params}")
    print("Running inference comparison...")
    run_comparison(
        priors_csv=str(config.priors_csv),
        submodel_dir=str(config.submodel_dir),
        num_samples=4000,
        # Decided here, where the config lives, rather than discovered inside
        # run_comparison: absence has to be a statement someone made.
        parameter_groups_path=(
            config.param_groups if config.param_groups.exists() else None
        ),
        invalidate_params=invalidate_params,
        run_population_pass=run_population_pass,
    )
    print("Done.")






if __name__ == "__main__":
    main()
