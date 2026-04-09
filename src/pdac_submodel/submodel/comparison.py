"""Compare single-target vs joint inference results for submodel targets.

Runs MCMC inference on each target individually, then jointly across all
targets, and produces a comparison report showing how joint inference
shifts parameter estimates relative to single-target posteriors and the
original CSV priors.

Can be run standalone::

    python -m qsp_inference.submodel.comparison \\
        --priors-csv parameters/pdac_priors.csv \\
        --submodel-dir calibration_targets/submodel_targets/ \\
        --output results/inference_comparison/report.md

Or called programmatically::

    from qsp_inference.submodel.comparison import run_comparison
    report = run_comparison(priors_csv, submodel_dir)
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# MCMC result caching
# =============================================================================


def _compute_hash(*contents: str | bytes) -> str:
    """Compute a short SHA256 hash from concatenated contents."""
    h = hashlib.sha256()
    for c in contents:
        if isinstance(c, str):
            c = c.encode()
        h.update(c)
    return h.hexdigest()[:16]


def _cache_dir(submodel_dir: Path) -> Path:
    d = submodel_dir / ".compare_cache"
    d.mkdir(exist_ok=True)
    return d


def _save_cache(path: Path, data: dict) -> None:
    """Save MCMC results to JSON cache file."""

    def _convert(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Cannot serialize {type(obj)}")

    with open(path, "w") as f:
        json.dump(data, f, default=_convert)


def _load_cache(path: Path) -> dict | None:
    """Load cached MCMC results if they exist."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _contraction(prior_sigma: float, posterior_sigma: float) -> float:
    """Compute contraction: 1 - (posterior_sigma / prior_sigma)^2.

    +1 = fully constrained (posterior collapsed to a point).
     0 = no information gained.
    <0 = posterior wider than prior (data conflicts or model issues).
    """
    if prior_sigma <= 0:
        return 0.0
    return 1.0 - (posterior_sigma / prior_sigma) ** 2


def _z_score(prior_mu: float, posterior_mu: float, prior_sigma: float) -> float:
    """Shift in log-space medians, normalized by prior sigma."""
    if prior_sigma <= 0:
        return 0.0
    return abs(posterior_mu - prior_mu) / prior_sigma


def _build_structured_results(
    csv_priors: dict,
    joint_fits: dict,
    single_results: dict,
    joint_diag: dict,
    all_param_names: set,
    n_targets: int,
    num_samples: int,
) -> dict:
    """Build structured dict for YAML serialization."""
    from datetime import datetime, timezone

    parameters = {}
    for pname in sorted(all_param_names):
        prior_spec = csv_priors.get(pname)
        prior = None
        if prior_spec:
            median = float(np.exp(prior_spec.mu)) if prior_spec.mu is not None else None
            prior = {
                "median": median,
                "sigma": float(prior_spec.sigma) if prior_spec.sigma is not None else None,
            }

        joint = None
        if pname in joint_fits:
            jf = joint_fits[pname]
            joint = {
                "median": float(jf["median"]),
                "sigma": float(jf["sigma"]),
                "cv": float(jf["cv"]),
                "contraction": round(float(jf["contraction"]), 4),
                "z_score": round(float(jf["z_score"]), 4),
                "distribution": jf["dist"],
            }

        singles = []
        if pname in single_results:
            for entry in single_results[pname]:
                singles.append(
                    {
                        "target_id": entry["target_id"],
                        "median": float(entry["median"]),
                        "sigma": float(entry["sigma"]),
                        "cv": float(entry["cv"]),
                        "contraction": round(float(entry["contraction"]), 4),
                        "z_score": round(float(entry["z_score"]), 4),
                    }
                )

        # Flags
        flags = []
        if prior and joint and prior["sigma"] > 0:
            if joint["sigma"] / prior["sigma"] > 0.8:
                flags.append("weak_contraction")
        if joint and joint["z_score"] > 2.0:
            flags.append("high_z_score")

        # Tension: single-target medians disagree by >3x
        tension = False
        if len(singles) >= 2:
            medians = [s["median"] for s in singles]
            if min(medians) > 0:
                tension = max(medians) / min(medians) > 3.0

        diag = joint_diag.get("per_param", {}).get(pname)
        mcmc_diag = None
        if diag and "n_eff" in diag and "r_hat" in diag:
            mcmc_diag = {
                "n_eff": round(float(diag["n_eff"])),
                "r_hat": round(float(diag["r_hat"]), 4),
            }

        param_entry = {"prior": prior, "joint": joint}
        if singles:
            param_entry["single_targets"] = singles
        if tension:
            param_entry["tension"] = True
        if flags:
            param_entry["flags"] = flags
        if mcmc_diag:
            param_entry["mcmc_diagnostics"] = mcmc_diag

        # SBC results (NPE components only)
        sbc = joint_diag.get("sbc", {}).get(pname)
        if sbc:
            param_entry["sbc"] = sbc

        parameters[pname] = param_entry

    # Aggregate PPC stats
    ppc_covered = joint_diag.get("ppc_n_covered", 0)
    ppc_total = joint_diag.get("ppc_n_total", 0)

    return {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_targets": n_targets,
            "n_parameters": len(all_param_names),
            "method": "component_wise",
            "num_samples": num_samples,
            "num_divergences": joint_diag.get("num_divergences", 0),
            "ppc_coverage": float(ppc_covered / ppc_total) if ppc_total else None,
            "ppc_n_covered": ppc_covered,
            "ppc_n_total": ppc_total,
        },
        "parameters": parameters,
        "ppc_observables": joint_diag.get("ppc_observables", []),
    }


def _lightweight_parse(raw_yaml_or_dict: str | dict) -> dict | None:
    """Extract target_id and QSP parameter names without Pydantic.

    Accepts either a raw YAML string or an already-parsed dict.

    Returns dict with 'target_id' and 'qsp_params' (set of non-nuisance param
    names), or None if parsing fails.
    """
    if isinstance(raw_yaml_or_dict, dict):
        data = raw_yaml_or_dict
    else:
        import yaml

        try:
            data = yaml.safe_load(raw_yaml_or_dict)
        except Exception:
            return None
    if not isinstance(data, dict):
        return None
    target_id = data.get("target_id")
    cal = data.get("calibration", {})
    params = cal.get("parameters", [])
    qsp_params = set()
    for p in params:
        if not p.get("nuisance", False):
            qsp_params.add(p["name"])
    return {"target_id": target_id, "qsp_params": qsp_params}


def _find_components_lightweight(
    lightweight_targets: list[dict],
    param_groups,
    cascade_cut_params: frozenset[str] = frozenset(),
) -> list[dict]:
    """Find connected components using lightweight target data.

    Args:
        lightweight_targets: List of dicts from _lightweight_parse, each with
            'target_id', 'qsp_params', and 'filename'.
        param_groups: Optional ParameterGroupsConfig.
        cascade_cut_params: Parameter names that should NOT merge components
            during BFS. Each component still includes these params in its
            ``params`` set, but the BFS does not follow edges through them.

    Returns list of dicts: [{"params": set[str], "target_filenames": list[str]}]
    """
    from collections import defaultdict, deque

    # Build param -> target mapping (QSP params only)
    param_to_targets = defaultdict(list)
    target_to_params = {}
    for lt in lightweight_targets:
        fname = lt["filename"]
        target_to_params[fname] = lt["qsp_params"]
        for p in lt["qsp_params"]:
            param_to_targets[p].append(fname)

    # Build group membership edges
    group_edges = defaultdict(set)  # param -> set of group-linked params
    if param_groups:
        for g in param_groups.groups:
            members = {m.name for m in g.members}
            for m in members:
                group_edges[m] = group_edges[m] | members

    # All params (from targets + groups)
    all_params = set(param_to_targets.keys())
    if param_groups:
        all_params |= param_groups.all_grouped_params

    # BFS to find connected components
    # Cascade cut params are added to comp_params when encountered but do NOT
    # propagate the BFS to other targets/groups — they act as severed edges.
    # They are NOT marked visited so they can appear in multiple components.
    visited = set()
    components = []
    for start_p in sorted(all_params):
        if start_p in visited or start_p in cascade_cut_params:
            continue
        comp_params = set()
        comp_filenames = set()
        queue = deque([start_p])
        while queue:
            p = queue.popleft()
            if p in cascade_cut_params:
                # Include in this component but don't mark visited (so other
                # components can also claim it) and don't propagate edges.
                comp_params.add(p)
                continue
            if p in visited:
                continue
            visited.add(p)
            comp_params.add(p)
            # Link via shared targets
            for fname in param_to_targets.get(p, []):
                comp_filenames.add(fname)
                for p2 in target_to_params[fname]:
                    if p2 not in visited:
                        queue.append(p2)
            # Link via group membership
            for p2 in group_edges.get(p, set()):
                if p2 not in visited:
                    queue.append(p2)

        components.append({"params": comp_params, "target_filenames": sorted(comp_filenames)})

    return components


# =============================================================================
# Cascade cuts — staged inference DAG
# =============================================================================


def _build_stage_dag(
    components: list[dict],
    cascade_cuts: list,
    lightweight_targets: list[dict],
) -> tuple[list[list[int]], dict[str, dict]]:
    """Build a DAG of component stages from cascade cuts.

    For each cascade cut, the component containing the upstream targets
    gets a directed edge to every other component that references the
    cut parameter. Components are then topologically sorted into stages.

    Args:
        components: From ``_find_components_lightweight``, each with
            ``params`` (set[str]) and ``target_filenames`` (list[str]).
        cascade_cuts: List of ``CascadeCut`` objects.
        lightweight_targets: For mapping target_id → component index.

    Returns:
        stages: List of lists of component indices, topologically sorted.
            ``stages[0]`` has no upstream dependencies, etc.
        cascade_edges: ``{param_name: {"upstream_comp": int, "downstream_comps": [int]}}``

    Raises:
        ValueError: If a cycle is detected or upstream targets are not found.
    """
    from collections import defaultdict, deque

    if not cascade_cuts:
        return [list(range(len(components)))], {}

    # Map target_id → component index
    target_id_to_comp: dict[str, int] = {}
    for lt in lightweight_targets:
        tid = lt["target_id"]
        fname = lt["filename"]
        for ci, comp in enumerate(components):
            if fname in comp["target_filenames"]:
                target_id_to_comp[tid] = ci
                break

    # Build cascade edges
    cascade_edges: dict[str, dict] = {}
    adj: dict[int, set[int]] = defaultdict(set)  # upstream → {downstream}
    in_degree: dict[int, int] = {i: 0 for i in range(len(components))}

    for cut in cascade_cuts:
        # Find upstream component (must contain ALL listed upstream targets)
        upstream_comps = set()
        for tid in cut.upstream:
            if tid not in target_id_to_comp:
                raise ValueError(
                    f"Cascade cut for '{cut.parameter}': upstream target "
                    f"'{tid}' not found in any component"
                )
            upstream_comps.add(target_id_to_comp[tid])
        if len(upstream_comps) != 1:
            raise ValueError(
                f"Cascade cut for '{cut.parameter}': upstream targets "
                f"must all be in the same component, but found components "
                f"{upstream_comps}"
            )
        upstream_ci = upstream_comps.pop()

        # Find downstream components (all others that have this param)
        downstream_cis = []
        for ci, comp in enumerate(components):
            if ci != upstream_ci and cut.parameter in comp["params"]:
                downstream_cis.append(ci)

        cascade_edges[cut.parameter] = {
            "upstream_comp": upstream_ci,
            "downstream_comps": downstream_cis,
        }

        for dci in downstream_cis:
            adj[upstream_ci].add(dci)
            in_degree[dci] = in_degree.get(dci, 0) + 1

    # Warn if multiple cascade params share the same upstream component —
    # their posterior correlation will be lost (each gets an independent
    # lognormal prior in the downstream). A joint copula-aware prior would
    # be needed to preserve it.
    upstream_to_params: dict[int, list[str]] = defaultdict(list)
    for cut in cascade_cuts:
        info = cascade_edges[cut.parameter]
        if info["downstream_comps"]:
            upstream_to_params[info["upstream_comp"]].append(cut.parameter)
    for uci, params in upstream_to_params.items():
        if len(params) > 1:
            logger.warning(
                "Cascade cuts %s share upstream component %d — their posterior "
                "correlation will be lost in downstream priors. Consider a joint "
                "copula-aware prior if these parameters are correlated.",
                params,
                uci,
            )

    # Topological sort (Kahn's algorithm)
    queue = deque(ci for ci in range(len(components)) if in_degree.get(ci, 0) == 0)
    stages: list[list[int]] = []
    processed = set()

    while queue:
        # All nodes with in_degree 0 form the current stage
        current_stage = sorted(queue)
        queue.clear()
        stages.append(current_stage)
        for ci in current_stage:
            processed.add(ci)
            for dci in adj.get(ci, set()):
                in_degree[dci] -= 1
                if in_degree[dci] == 0:
                    queue.append(dci)

    if len(processed) < len(components):
        unprocessed = set(range(len(components))) - processed
        raise ValueError(
            f"Cycle detected in cascade cuts: components {unprocessed} " f"form a dependency cycle"
        )

    return stages, cascade_edges


def _posterior_to_prior_spec(
    samples: np.ndarray,
    param_name: str,
    original_spec,
) -> object:
    """Convert posterior samples to a PriorSpec for downstream injection.

    Fits distributions to the samples and converts the best fit to a
    lognormal PriorSpec (the universal prior format for QSP parameters).

    Args:
        samples: 1D array of posterior samples.
        param_name: Parameter name.
        original_spec: Original PriorSpec from CSV (for units).

    Returns:
        PriorSpec with lognormal distribution fitted to the posterior.
    """
    from qsp_inference.submodel.inference import PriorSpec
    from qsp_inference.submodel.prior import fit_distributions

    fits = fit_distributions(samples)
    if not fits:
        logger.warning(
            "Cascade: could not fit distribution to %s, using original prior",
            param_name,
        )
        return original_spec

    best = fits[0]
    if best.name == "lognormal":
        mu = best.params["mu"]
        sigma = best.params["sigma"]
    else:
        # Convert any distribution to lognormal approximation via moments
        log_samples = np.log(samples[samples > 0])
        if len(log_samples) < 10:
            logger.warning(
                "Cascade: too few positive samples for %s, using original prior",
                param_name,
            )
            return original_spec
        mu = float(np.mean(log_samples))
        sigma = float(np.std(log_samples))

    return PriorSpec(
        name=param_name,
        distribution="lognormal",
        units=original_spec.units,
        mu=mu,
        sigma=max(sigma, 0.01),  # floor to prevent degenerate priors
    )


def _cascade_invalidation(
    comp_cache_info: list[dict],
    cascade_edges: dict[str, dict],
) -> None:
    """Propagate cache invalidation through the cascade DAG.

    If an upstream component has no cache, all downstream components
    must also be invalidated (their priors depend on the upstream posterior).
    """
    from collections import deque

    if not cascade_edges:
        return

    # Build forward adjacency from cascade edges
    adj: dict[int, set[int]] = {}
    for edge_info in cascade_edges.values():
        uci = edge_info["upstream_comp"]
        if uci not in adj:
            adj[uci] = set()
        adj[uci].update(edge_info["downstream_comps"])

    # Find initially invalidated components (cache miss)
    invalidated = set()
    for i, cci in enumerate(comp_cache_info):
        if cci["cached"] is None:
            invalidated.add(i)

    # BFS forward through DAG
    queue = deque(invalidated)
    while queue:
        ci = queue.popleft()
        for dci in adj.get(ci, set()):
            if dci not in invalidated:
                invalidated.add(dci)
                cache_path = comp_cache_info[dci]["cache_path"]
                if cache_path.exists():
                    cache_path.unlink()
                    logger.info(
                        "Cascade-invalidated cache: %s",
                        cache_path.name,
                    )
                comp_cache_info[dci]["cached"] = None
                queue.append(dci)


def _cleanup_orphaned_caches(cache_dir: Path, active_components: list[dict]) -> int:
    """Delete cache files that don't match any active component.

    When cascade cuts change the component structure, old mega-component
    caches persist with stale results. This removes them so the aggregation
    step doesn't load conflicting values.

    Returns the number of orphaned files deleted.
    """
    active_hashes = set()
    for comp in active_components:
        active_hashes.add(_compute_hash("\n".join(sorted(comp["params"]))))
    deleted = 0
    for orphan in cache_dir.glob("comp_*.json"):
        comp_hash = orphan.stem.replace("comp_", "")
        if comp_hash not in active_hashes:
            orphan.unlink()
            logger.info("Deleted orphaned cache: %s", orphan.name)
            deleted += 1
    return deleted


def run_comparison(
    priors_csv: str | Path,
    submodel_dir: str | Path,
    glob_pattern: str = "*_PDAC_deriv*.yaml",
    num_samples: int = 4000,
    parameter_groups_path: str | Path | None = None,
    invalidate_params: list[str] | None = None,
) -> str:
    """Run component-wise NPE inference, return comparison report.

    Args:
        priors_csv: Path to priors CSV.
        submodel_dir: Directory containing SubmodelTarget YAMLs.
        glob_pattern: Glob for YAML files.
        num_samples: Number of posterior samples per component.
        parameter_groups_path: Optional path to submodel_config.yaml.
        invalidate_params: Optional list of parameter names. Any cached
            component containing at least one of these parameters will be
            deleted and re-run.

    Returns:
        Markdown-formatted comparison report.
    """
    import yaml

    from qsp_inference.submodel.inference import (
        load_priors_from_csv,
    )
    from maple.core.calibration.submodel_target import SubmodelTarget
    from qsp_inference.submodel.prior import (
        fit_distributions,
    )

    from qsp_inference.submodel.parameter_groups import load_parameter_groups

    priors_csv = Path(priors_csv)
    submodel_dir = Path(submodel_dir)

    # Load parameter groups if provided (or auto-discover in submodel_dir)
    param_groups = None
    if parameter_groups_path is not None:
        param_groups = load_parameter_groups(Path(parameter_groups_path))
    else:
        auto_path = submodel_dir / "submodel_config.yaml"
        if auto_path.exists():
            param_groups = load_parameter_groups(auto_path)
    if param_groups and param_groups.groups:
        logger.info(
            "Loaded %d parameter groups (%d params)",
            len(param_groups.groups),
            len(param_groups.all_grouped_params),
        )

    yaml_files = sorted(submodel_dir.glob(glob_pattern))
    # Exclude submodel_config.yaml from target list
    yaml_files = [f for f in yaml_files if f.name != "submodel_config.yaml"]
    if not yaml_files:
        return f"No YAML files found matching {glob_pattern} in {submodel_dir}"

    csv_priors = load_priors_from_csv(priors_csv)
    cache = _cache_dir(submodel_dir)

    # ── Lightweight parse: extract param names without Pydantic validation ──
    yaml_contents: dict[str, str] = {}  # {filename: raw_content}
    lightweight_targets: list[dict] = []
    for yf in yaml_files:
        try:
            raw = yf.read_text()
            yaml_contents[yf.name] = raw
            lt = _lightweight_parse(raw)
            if lt is not None:
                lt["filename"] = yf.name
                lightweight_targets.append(lt)
        except Exception as e:
            logger.warning("Failed to read %s: %s", yf.name, e)

    all_param_names = set()
    for lt in lightweight_targets:
        all_param_names |= lt["qsp_params"]
    if param_groups:
        all_param_names |= param_groups.all_grouped_params

    # ── Phase 1: Component-wise joint inference ──
    # Find connected components using lightweight data (no Pydantic needed)
    cascade_cut_params = frozenset(param_groups.cascade_cut_params if param_groups else ())
    components = _find_components_lightweight(lightweight_targets, param_groups, cascade_cut_params)
    logger.info(
        "Phase 1: %d components (largest: %d params, %d targets)",
        len(components),
        max(len(c["params"]) for c in components) if components else 0,
        max(len(c["target_filenames"]) for c in components) if components else 0,
    )

    joint_fits: dict[str, dict] = {}
    joint_diag: dict = {"num_divergences": 0, "per_param": {}}
    joint_samples_all: dict[str, list] = {}

    # Filter out components with no targets (e.g., group-only with no data)
    active_components = [c for c in components if c["target_filenames"]]
    logger.info("Phase 1: %d active components", len(active_components))

    # Invalidate cached components containing specified parameters
    if invalidate_params:
        invalidate_set = set(invalidate_params)
        for comp in active_components:
            if comp["params"] & invalidate_set:
                comp_id = _compute_hash("\n".join(sorted(comp["params"])))
                cache_file = cache / f"comp_{comp_id}.json"
                if cache_file.exists():
                    cache_file.unlink()
                    logger.info(
                        "Invalidated cache: %s (matched params: %s)",
                        cache_file.name,
                        comp["params"] & invalidate_set,
                    )

    # ── Clean up orphaned caches ──
    _cleanup_orphaned_caches(cache, active_components)

    # ── Pre-check caches to skip expensive Pydantic validation ──
    # Cache is keyed only by component identity (sorted parameter names).
    # No content hashing — results persist until manually invalidated via
    # invalidate_params or by deleting .compare_cache/ files.
    comp_cache_info: list[dict] = []
    for comp in active_components:
        comp_params = comp["params"]
        comp_id = _compute_hash("\n".join(sorted(comp_params)))
        comp_cache_path = cache / f"comp_{comp_id}.json"

        comp_cache_info.append(
            {
                "comp": comp,
                "comp_id": comp_id,
                "cache_path": comp_cache_path,
                "cached": _load_cache(comp_cache_path),
            }
        )

    # Only validate targets whose components have cache misses
    filenames_needing_validation = set()
    for cci in comp_cache_info:
        if cci["cached"] is None:
            filenames_needing_validation.update(cci["comp"]["target_filenames"])

    # ── Build cascade stage DAG ──
    if param_groups and param_groups.cascade_cuts:
        stages, cascade_edges = _build_stage_dag(
            active_components, param_groups.cascade_cuts, lightweight_targets
        )
        logger.info(
            "Cascade: %d stages, %d cut params (%s)",
            len(stages),
            len(cascade_edges),
            ", ".join(sorted(cascade_edges)),
        )
        # Propagate cache invalidation through the DAG
        _cascade_invalidation(comp_cache_info, cascade_edges)
        # Re-check which files need validation after cascade invalidation
        filenames_needing_validation = set()
        for cci in comp_cache_info:
            if cci["cached"] is None:
                filenames_needing_validation.update(cci["comp"]["target_filenames"])
    else:
        stages = [list(range(len(active_components)))]
        cascade_edges = {}

    n_cached = sum(1 for cci in comp_cache_info if cci["cached"] is not None)
    n_total_targets = sum(len(c["target_filenames"]) for c in active_components)
    logger.info(
        "Cache: %d/%d components cached — validating %d/%d targets",
        n_cached,
        len(comp_cache_info),
        len(filenames_needing_validation),
        n_total_targets,
    )

    # ── Validate only the targets we actually need ──
    validated_targets: dict[str, object] = {}  # {filename: SubmodelTarget}
    for fname in filenames_needing_validation:
        raw = yaml_contents[fname]
        try:
            data = yaml.safe_load(raw)
            target = SubmodelTarget.model_validate(data)
            validated_targets[fname] = target
        except Exception as e:
            logger.warning("Failed to validate %s: %s", fname, e)

    # ── Staged execution ──
    # Components run stage-by-stage. After each stage, cascade cut posteriors
    # are fitted and injected as priors for downstream stages.
    cascade_priors: dict[str, object] = {}  # param_name → PriorSpec from upstream

    for stage_idx, stage_comp_indices in enumerate(stages):
        if len(stages) > 1:
            logger.info(
                "Stage %d/%d: %d components",
                stage_idx,
                len(stages) - 1,
                len(stage_comp_indices),
            )

        for ci in stage_comp_indices:
            cci = comp_cache_info[ci]
            comp = cci["comp"]
            comp_params = comp["params"]
            comp_cache_path = cci["cache_path"]

            cached_comp = cci["cached"]
            if cached_comp is not None:
                for k, v in cached_comp.get("fits", {}).items():
                    joint_fits[k] = v
                comp_diag = cached_comp.get("diag", {})
                joint_diag["num_divergences"] += comp_diag.get("num_divergences", 0)
                for k, v in comp_diag.get("per_param", {}).items():
                    joint_diag["per_param"][k] = v
                for k, v in cached_comp.get("samples", {}).items():
                    joint_samples_all[k] = v
                # Forward PPC observables from cached components
                if "ppc_observables" in comp_diag:
                    if "ppc_observables" not in joint_diag:
                        joint_diag["ppc_observables"] = []
                    joint_diag["ppc_observables"].extend(comp_diag["ppc_observables"])
                    joint_diag["ppc_n_total"] = joint_diag.get("ppc_n_total", 0) + comp_diag.get(
                        "ppc_n_total", 0
                    )
                    joint_diag["ppc_n_covered"] = joint_diag.get(
                        "ppc_n_covered", 0
                    ) + comp_diag.get("ppc_n_covered", 0)
                continue

            # Resolve validated SubmodelTarget objects for this component
            comp_targets = [
                validated_targets[fname]
                for fname in comp["target_filenames"]
                if fname in validated_targets
            ]
            if not comp_targets:
                logger.warning("  Component %d: no valid targets after validation", ci + 1)
                continue

            # Build prior specs for this component
            comp_prior_specs = {k: v for k, v in csv_priors.items() if k in comp_params}

            # Inject cascade priors from upstream stages
            for pname in comp_params:
                if pname in cascade_priors:
                    comp_prior_specs[pname] = cascade_priors[pname]
                    logger.info(
                        "    Injected cascade prior for %s (mu=%.3f, sigma=%.3f)",
                        pname,
                        cascade_priors[pname].mu,
                        cascade_priors[pname].sigma,
                    )

            # Find relevant parameter groups for this component
            comp_groups = None
            if param_groups:
                from qsp_inference.submodel.parameter_groups import (
                    ParameterGroupsConfig,
                )

                relevant = [
                    g for g in param_groups.groups if any(m.name in comp_params for m in g.members)
                ]
                if relevant:
                    comp_groups = ParameterGroupsConfig(groups=relevant)

            n_p = len(comp_params)
            n_t = len(comp_targets)
            logger.info(
                "  Component %d/%d: %d params, %d targets",
                ci + 1,
                len(comp_cache_info),
                n_p,
                n_t,
            )

            has_ode = any(t.calibration.forward_model.type == "custom_ode" for t in comp_targets)

            # Check for per-target inference method overrides
            use_method = None
            if param_groups:
                comp_target_ids = [t.target_id for t in comp_targets]
                use_method = param_groups.get_inference_method_for_targets(comp_target_ids)

            use_npe = has_ode if use_method is None else (use_method == "npe")

            try:
                if use_npe:
                    # NPE for ODE components (gives SBC diagnostics)
                    from qsp_inference.submodel.inference import (
                        run_component_npe,
                    )

                    logger.info("    (NPE — ODE)")
                    comp_samples, comp_diag = run_component_npe(
                        comp_prior_specs,
                        comp_targets,
                        parameter_groups=comp_groups,
                        num_posterior_samples=num_samples,
                    )
                else:
                    # Multi-target, no ODE: NUTS is fast and exact
                    from qsp_inference.submodel.inference import (
                        run_joint_inference,
                    )

                    logger.info("    (joint MCMC — %d targets)", n_t)
                    comp_samples, comp_diag = run_joint_inference(
                        comp_prior_specs,
                        comp_targets,
                        parameter_groups=comp_groups,
                        num_warmup=2000,
                        num_samples=num_samples,
                        num_chains=2,
                    )
            except Exception as e:
                logger.warning("  Component %d failed: %s", ci + 1, e)
                continue

            # Fit distributions and accumulate
            comp_fits = {}
            for pname in sorted(comp_params):
                if pname not in comp_samples:
                    continue
                fits = fit_distributions(comp_samples[pname])
                if not fits:
                    continue
                best = fits[0]

                if best.name == "lognormal":
                    post_sigma = best.params["sigma"]
                else:
                    post_sigma = np.sqrt(np.log(1 + best.cv**2))

                prior_sigma = csv_priors[pname].sigma if pname in csv_priors else 1.0
                prior_mu = csv_priors[pname].mu if pname in csv_priors else 0.0
                post_mu = np.log(best.median)

                comp_fits[pname] = {
                    "median": best.median,
                    "cv": best.cv,
                    "sigma": post_sigma,
                    "dist": best.name,
                    "contraction": _contraction(prior_sigma, post_sigma),
                    "z_score": _z_score(prior_mu, post_mu, prior_sigma),
                }
                joint_fits[pname] = comp_fits[pname]

            # Run PPC for non-NPE components (NPE does its own PPC internally)
            if not has_ode and "ppc_coverage" not in comp_diag:
                from qsp_inference.submodel.inference import (
                    build_numpy_forward_fns,
                    build_target_likelihoods,
                )

                ppc_fns = []
                ppc_obs = []
                ppc_obs_ci = []
                ppc_obs_fits = []  # Store fit info for bootstrap sampling
                comp_tls = build_target_likelihoods(comp_targets, comp_prior_specs)
                for target, tl_entry in zip(comp_targets, comp_tls):
                    fns = build_numpy_forward_fns(target)
                    for fn, le in zip(fns, tl_entry.entries):
                        ppc_fns.append(fn)
                        fit = le.fit
                        ppc_obs.append(float(fit.median))
                        if fit.name == "lognormal" and "sigma" in fit.params:
                            from scipy.stats import lognorm as _lognorm

                            d = _lognorm(s=fit.params["sigma"], scale=fit.median)
                            ppc_obs_ci.append([float(d.ppf(0.025)), float(d.ppf(0.975))])
                            ppc_obs_fits.append(("lognormal", fit.median, fit.params["sigma"]))
                        elif fit.cv and fit.cv > 0:
                            sd = abs(fit.median) * fit.cv
                            ppc_obs_ci.append(
                                [float(fit.median - 1.96 * sd), float(fit.median + 1.96 * sd)]
                            )
                            ppc_obs_fits.append(("normal", fit.median, sd))
                        else:
                            ppc_obs_ci.append(None)
                            ppc_obs_fits.append(None)

                # Build observable names
                ppc_obs_names = []
                for target in comp_targets:
                    for entry in target.calibration.error_model:
                        ppc_obs_names.append(f"{target.target_id}__{entry.name}")

                n_ppc = min(200, len(comp_samples.get(next(iter(comp_samples), ""), [])))
                if n_ppc > 0 and ppc_fns:
                    nuisance = {}
                    for t in comp_targets:
                        for p in t.calibration.parameters:
                            if p.nuisance and p.prior:
                                nuisance[p.name] = (p.prior.mu, p.prior.sigma)
                    rng = np.random.default_rng(42)

                    # Prior predictive (sample from CSV priors)
                    prior_preds_all = [[] for _ in ppc_fns]
                    for i in range(n_ppc):
                        pd = {}
                        for pn in comp_prior_specs:
                            sp = comp_prior_specs[pn]
                            pd[pn] = float(rng.lognormal(sp.mu, sp.sigma))
                        for nn, (mu, sig) in nuisance.items():
                            pd[nn] = float(rng.lognormal(mu, sig))
                        for obs_idx, fn in enumerate(ppc_fns):
                            try:
                                prior_preds_all[obs_idx].append(float(fn(pd)))
                            except Exception:
                                pass

                    n_covered = 0
                    ppc_observables = []
                    for obs_idx, fn in enumerate(ppc_fns):
                        # Posterior predictive
                        preds = []
                        for i in range(n_ppc):
                            pd = {
                                pn: float(comp_samples[pn][i])
                                for pn in comp_samples
                                if i < len(comp_samples[pn])
                            }
                            for nn, (mu, sig) in nuisance.items():
                                pd[nn] = float(rng.lognormal(mu, sig))
                            try:
                                preds.append(float(fn(pd)))
                            except Exception:
                                pass
                        entry = {
                            "name": (
                                ppc_obs_names[obs_idx]
                                if obs_idx < len(ppc_obs_names)
                                else f"obs_{obs_idx}"
                            ),
                            "observed": ppc_obs[obs_idx],
                        }
                        if obs_idx < len(ppc_obs_ci) and ppc_obs_ci[obs_idx]:
                            entry["obs_ci95"] = ppc_obs_ci[obs_idx]
                        # Prior predictive samples + CI
                        pp = prior_preds_all[obs_idx]
                        pp_valid = [v for v in pp if np.isfinite(v)]
                        if len(pp_valid) >= 10:
                            entry["prior_median"] = float(np.median(pp_valid))
                            entry["prior_ci95"] = [
                                float(np.percentile(pp_valid, 2.5)),
                                float(np.percentile(pp_valid, 97.5)),
                            ]
                            entry["prior_samples"] = [float(v) for v in pp_valid]
                        # Posterior predictive samples + CI
                        preds_valid = [v for v in preds if np.isfinite(v)]
                        if len(preds_valid) >= 10:
                            lo, hi = np.percentile(preds_valid, [2.5, 97.5])
                            entry["post_median"] = float(np.median(preds_valid))
                            entry["post_ci95"] = [float(lo), float(hi)]
                            entry["post_samples"] = [float(v) for v in preds_valid]
                            entry["covered"] = bool(lo <= ppc_obs[obs_idx] <= hi)
                            if entry["covered"]:
                                n_covered += 1
                        # Observed bootstrap samples from fit distribution
                        if obs_idx < len(ppc_obs_fits) and ppc_obs_fits[obs_idx]:
                            fit_type, fit_med, fit_param = ppc_obs_fits[obs_idx]
                            if fit_type == "lognormal":
                                obs_samps = rng.lognormal(np.log(fit_med), fit_param, size=n_ppc)
                            else:  # normal
                                obs_samps = rng.normal(fit_med, fit_param, size=n_ppc)
                            obs_samps_valid = [
                                float(v) for v in obs_samps if np.isfinite(v)
                            ]
                            if obs_samps_valid:
                                entry["obs_samples"] = obs_samps_valid
                        ppc_observables.append(entry)
                    comp_diag["ppc_coverage"] = float(n_covered / len(ppc_fns)) if ppc_fns else 0
                    comp_diag["ppc_n_covered"] = n_covered
                    comp_diag["ppc_n_total"] = len(ppc_fns)
                    comp_diag["ppc_observables"] = ppc_observables
                    logger.info("    PPC: %d/%d covered", n_covered, len(ppc_fns))

            joint_diag["num_divergences"] += comp_diag.get("num_divergences", 0)
            for k, v in comp_diag.get("per_param", {}).items():
                joint_diag["per_param"][k] = v
            # Accumulate SBC results
            if "sbc" in comp_diag:
                if "sbc" not in joint_diag:
                    joint_diag["sbc"] = {}
                joint_diag["sbc"].update(comp_diag["sbc"])
            # Accumulate PPC
            joint_diag["ppc_n_covered"] = joint_diag.get("ppc_n_covered", 0) + comp_diag.get(
                "ppc_n_covered", 0
            )
            joint_diag["ppc_n_total"] = joint_diag.get("ppc_n_total", 0) + comp_diag.get(
                "ppc_n_total", 0
            )
            if "ppc_observables" in comp_diag:
                if "ppc_observables" not in joint_diag:
                    joint_diag["ppc_observables"] = []
                joint_diag["ppc_observables"].extend(comp_diag["ppc_observables"])

            comp_samples_list = {k: v for k, v in comp_samples.items()}
            for k, v in comp_samples_list.items():
                joint_samples_all[k] = v

            _save_cache(
                comp_cache_path,
                {
                    "fits": comp_fits,
                    "diag": comp_diag,
                    "samples": comp_samples_list,
                },
            )

        # After each stage: extract posteriors for cascade params and build
        # priors for downstream stages
        for param_name, edge_info in cascade_edges.items():
            if edge_info["upstream_comp"] in stage_comp_indices:
                if param_name in joint_samples_all:
                    cascade_priors[param_name] = _posterior_to_prior_spec(
                        np.array(joint_samples_all[param_name]),
                        param_name,
                        csv_priors[param_name],
                    )
                    logger.info(
                        "  Cascade: %s posterior → prior (mu=%.3f, sigma=%.3f)",
                        param_name,
                        cascade_priors[param_name].mu,
                        cascade_priors[param_name].sigma,
                    )

    logger.info("Phase 1: done (%d params fitted)", len(joint_fits))

    # Single-target results no longer computed separately — NPE handles
    # everything component-wise. Keep empty dicts for report compatibility.
    single_results: dict[str, list[dict]] = {}

    # ── Save structured results ──
    structured = _build_structured_results(
        csv_priors=csv_priors,
        joint_fits=joint_fits,
        single_results=single_results,
        joint_diag=joint_diag,
        all_param_names=all_param_names,
        n_targets=len(lightweight_targets),
        num_samples=num_samples,
    )
    results_path = submodel_dir / "compare_inference_results.yaml"
    with open(results_path, "w") as f:
        yaml.dump(structured, f, default_flow_style=False, sort_keys=False)
    logger.info("Structured results saved to %s", results_path)

    # ── Build comparison report ──
    lines = [
        "# Inference Comparison Report",
        "",
        f"**Targets:** {len(lightweight_targets)}",
        f"**Parameters:** {len(all_param_names)}",
        f"**Method:** Component-wise NPE ({num_samples} posterior samples)",
        "",
    ]

    # Per-parameter detailed comparison
    for pname in sorted(all_param_names):
        lines.append(f"### `{pname}`")
        lines.append("")

        # CSV prior
        if pname in csv_priors:
            spec = csv_priors[pname]
            csv_median = np.exp(spec.mu)
            lines.append(
                f"**CSV prior:** median={csv_median:.4g}, "
                f"sigma={spec.sigma:.2f}, "
                f"CV={np.sqrt(np.exp(spec.sigma**2) - 1):.2f}"
            )
        else:
            lines.append("**CSV prior:** not in CSV")

        lines.append("")

        # Single-target results
        if pname in single_results:
            lines.append("| Source | Median | CV | Sigma | Contraction | z-score | Dist |")
            lines.append("|--------|--------|----|-------|-------------|---------|------|")

            for entry in single_results[pname]:
                short_tid = entry["target_id"]
                # Truncate long target IDs
                if len(short_tid) > 35:
                    short_tid = short_tid[:32] + "..."
                lines.append(
                    f"| {short_tid} | {entry['median']:.4g} | "
                    f"{entry['cv']:.2f} | {entry['sigma']:.3f} | "
                    f"{entry['contraction']:+.2f} | {entry['z_score']:.2f} | "
                    f"{entry['dist']} |"
                )

            # Joint row
            if pname in joint_fits:
                jf = joint_fits[pname]
                lines.append(
                    f"| **JOINT** | **{jf['median']:.4g}** | "
                    f"**{jf['cv']:.2f}** | **{jf['sigma']:.3f}** | "
                    f"**{jf['contraction']:+.2f}** | **{jf['z_score']:.2f}** | "
                    f"**{jf['dist']}** |"
                )
        else:
            lines.append("No single-target results (parameter only in joint model).")
            if pname in joint_fits:
                jf = joint_fits[pname]
                lines.append(
                    f"**Joint:** median={jf['median']:.4g}, CV={jf['cv']:.2f}, "
                    f"contraction={jf['contraction']:+.2f}"
                )

        lines.append("")

    # ── Summary table ──
    lines.extend(
        [
            "## Summary",
            "",
            "| Parameter | CSV Median | Joint Median | Shift | Joint Contraction |",
            "|-----------|------------|--------------|-------|-------------------|",
        ]
    )

    for pname in sorted(all_param_names):
        csv_med_str = "—"
        shift_str = "—"
        joint_str = "—"
        contr_str = "—"

        if pname in csv_priors:
            csv_median = np.exp(csv_priors[pname].mu)
            csv_med_str = f"{csv_median:.4g}"

        if pname in joint_fits:
            jf = joint_fits[pname]
            joint_str = f"{jf['median']:.4g}"
            contr_str = f"{jf['contraction']:+.2f}"

            if pname in csv_priors:
                ratio = jf["median"] / csv_median
                if ratio > 2:
                    shift_str = f"{ratio:.1f}x up"
                elif ratio < 0.5:
                    shift_str = f"{1/ratio:.1f}x down"
                else:
                    shift_str = f"{ratio:.2f}x"

        lines.append(f"| `{pname}` | {csv_med_str} | {joint_str} | {shift_str} | {contr_str} |")

    # ── Consistency check ──
    lines.extend(["", "## Consistency Check", ""])
    lines.append("Parameters where single-target estimates disagree by >3x:")
    lines.append("")

    any_disagreement = False
    for pname in sorted(single_results.keys()):
        entries = single_results[pname]
        if len(entries) < 2:
            continue
        medians = [e["median"] for e in entries]
        ratio = max(medians) / min(medians) if min(medians) > 0 else float("inf")
        if ratio > 3:
            any_disagreement = True
            vals = ", ".join(f"{m:.4g}" for m in medians)
            lines.append(f"- **`{pname}`**: {ratio:.1f}x spread ({vals})")
            for e in entries:
                lines.append(f"  - {e['target_id']}: {e['median']:.4g}")

    if not any_disagreement:
        lines.append("None — all multi-target parameters are consistent (<3x spread).")

    # ── Joint MCMC diagnostics ──
    lines.extend(["", "## Joint MCMC Diagnostics", ""])
    lines.append(f"- Divergences: {joint_diag['num_divergences']}")
    for pname in sorted(all_param_names):
        pd = joint_diag["per_param"].get(pname, {})
        if pd:
            neff = pd.get("n_eff", 0)
            rhat = pd.get("r_hat", 0)
            flag = " ⚠" if neff < 100 or rhat > 1.05 else ""
            lines.append(f"- `{pname}`: n_eff={neff:.0f}, r_hat={rhat:.3f}{flag}")

    return "\n".join(lines)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare single-target vs joint submodel inference"
    )
    parser.add_argument("--priors-csv", required=True, help="Path to priors CSV")
    parser.add_argument("--submodel-dir", required=True, help="Directory with SubmodelTarget YAMLs")
    parser.add_argument("--glob-pattern", default="*_PDAC_deriv*.yaml")
    parser.add_argument("--num-samples", type=int, default=4000)
    parser.add_argument("--output", help="Optional output file (markdown)")
    parser.add_argument(
        "--parameter-groups",
        help="Path to submodel_config.yaml (auto-discovered in submodel-dir if not set)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    report = run_comparison(
        priors_csv=args.priors_csv,
        submodel_dir=args.submodel_dir,
        glob_pattern=args.glob_pattern,
        num_samples=args.num_samples,
        parameter_groups_path=args.parameter_groups,
    )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
