#!/usr/bin/env python3
"""Parameter coverage audit for QSP models.

Auto-detects available data and reports:
- Stage 0: Prior coverage, submodel target coverage, PRCC-weighted priority
- Stage 1: Submodel posterior contraction (when inference results exist)
- Stage 2: NPE posterior shifts vs submodel posteriors (when SBI results exist)

Usage:
    python -m qsp_inference.audit.report --project-root /path/to/project
    python -m qsp_inference.audit.report --project-root . --output report.md
"""

import argparse
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

from qsp_inference.audit.plots import plot_inference_dag, plot_marginals, plot_ppc_histograms


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


def load_prcc(path: Path) -> dict:
    """Load PRCC rankings → {param_name: {rank, mean_abs_prcc, significant, n_sig_obs}}."""
    if not path.exists():
        return {}
    rankings = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("parameter", "").strip()
            if not name:
                continue
            try:
                n_sig = int(float(row.get("n_significant_prcc", 0)))
            except (ValueError, TypeError):
                n_sig = 0
            rankings[name] = {
                "rank": int(row.get("rank_prcc", 999)),
                "mean_abs_prcc": float(row.get("mean_abs_prcc", 0)),
                "significant": row.get("significant", "").lower() == "true",
                "n_sig_obs": n_sig,
            }
    return rankings


def load_cross_model(path: Path, model_names: list[str] | None = None) -> dict:
    """Load cross_model_parameters.csv → {prior_name: {model: value, ..., ref_name}}.

    Uses the ``alias`` column to map xlsx parameter names to priors CSV
    names.  When an alias is present the entry is keyed by the alias (i.e. the
    current prior name); the original xlsx name is stored as ``ref_name``.
    """
    if not path.exists():
        return {}
    models = model_names or ["TNBC", "CRC", "UM", "NSCLC", "HCC", "PDAC"]
    result = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_name = row.get("name", "").strip()
            if not raw_name:
                continue
            alias = row.get("alias", "").strip()
            key = alias if alias else raw_name
            values = {"ref_name": raw_name}
            n_present = 0
            for m in models:
                v = row.get(m, "").strip()
                values[m] = v
                if v and v.upper() != "NA":
                    n_present += 1
            values["unit"] = row.get("unit", "")
            values["description"] = row.get("description", "")
            values["n_models"] = n_present
            result[key] = values
    return result


def _cross_model_summary(cross_entry: dict) -> str:
    """One-line summary of cross-model values for a parameter."""
    models = ["TNBC", "CRC", "UM", "NSCLC", "HCC"]
    vals = []
    for m in models:
        v = cross_entry.get(m, "")
        if v and v.upper() != "NA":
            # Truncate long annotations
            short = v.split("(")[0].strip()[:12]
            vals.append(f"{m}:{short}")
    if not vals:
        return ""
    return ", ".join(vals)



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


def find_sbi_posteriors(sbi_dir: Path) -> list:
    """Find SBI posterior summary CSVs."""
    results = []
    for d in sorted(sbi_dir.glob("sbi_*")):
        summary = d / "posterior_summary.csv"
        samples = d / "posterior_samples.csv"
        if summary.exists() and samples.exists():
            results.append(
                {
                    "name": d.name,
                    "summary_path": summary,
                    "samples_path": samples,
                }
            )
    return results


def load_posterior_summary(path: Path) -> dict:
    """Load posterior_summary.csv → {param: {mean, std, median, ci_lo, ci_hi}}."""
    posteriors = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("", row.get("Unnamed: 0", "")).strip()
            if not name:
                continue
            try:
                posteriors[name] = {
                    "mean": float(row.get("mean", 0)),
                    "std": float(row.get("std", 0)),
                    "median": float(row.get("50%", 0)),
                    "ci_lo": float(row.get("2.5%", 0)),
                    "ci_hi": float(row.get("97.5%", 0)),
                }
            except (ValueError, TypeError):
                continue
    return posteriors


# ---------------------------------------------------------------------------
# MCMC sample loading and plotting
# ---------------------------------------------------------------------------


def load_component_diagnostics(cache_dir: Path) -> list[dict] | None:
    """Load per-component diagnostics from cache files."""
    if not cache_dir.exists():
        return None
    components = []
    for path in sorted(cache_dir.glob("comp_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        diag = data.get("diag", {})
        samples = data.get("samples", {})
        params = sorted(samples.keys())
        if not params:
            params = sorted(data.get("fits", {}).keys())
        if not params:
            continue
        per_param = diag.get("per_param", {})
        # Detect method
        method = (diag.get("method") or "").upper()
        if not method:
            n_effs = [v.get("n_eff", 0) for v in per_param.values() if v.get("n_eff")]
            method = "NPE" if (len(set(n_effs)) <= 1 and len(n_effs) > 1) else "NUTS"

        components.append({
            "params": params,
            "num_divergences": diag.get("num_divergences", 0),
            "ppc_n_covered": diag.get("ppc_n_covered", 0),
            "ppc_n_total": diag.get("ppc_n_total", 0),
            "per_param": per_param,
            "method": method,
            "sbc": diag.get("sbc", {}),
        })
    return components if components else None


def load_joint_samples(cache_dir: Path) -> dict | None:
    """Load posterior samples from component cache files.

    Merges samples across all component_*.json files.
    Returns {param_name: list[float]} or None if no samples available.
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


def classify_param(name: str) -> str:
    """Classify parameter by type based on naming conventions."""
    for pattern, label in _PARAM_TYPE_PATTERNS:
        if re.search(pattern, name):
            return label
    return "rate"


def _fmt_sigma(val) -> str:
    """Format sigma values consistently."""
    if val is None:
        return "---"
    return f"{float(val):.2f}"


def _contraction_bar(contraction: float) -> str:
    """Visual contraction indicator."""
    if contraction >= 0.5:
        return f"**{contraction:.0%}**"
    if contraction >= 0.2:
        return f"{contraction:.0%}"
    if contraction >= 0:
        return f"~{contraction:.0%}~"
    return f"_{contraction:.0%}_"


def compute_priority(
    sigma: float,
    prcc_rank: int | None,
    n_targets: int,
    in_group: bool,
    stage1_effectiveness: float = 1.0,
) -> float:
    """Higher score = higher priority for extraction.

    Factors:
    - High sigma (unconstrained) → high priority
    - Low PRCC rank (high sensitivity) → high priority
    - No *effective* existing targets → high priority
    - Not in a parameter group (no partial pooling fallback) → higher priority

    ``stage1_effectiveness`` is ``1 - (sigma_S1 / sigma_prior)**2`` clipped
    to [0, 1]; it weights the target count so that an existing-but-
    ineffective target does not discount priority (the param still needs
    data). When no stage-1 posterior is available, callers default to 1.0
    (preserves legacy behavior of counting targets at face value).
    """
    # Sigma contribution (0-1, normalized to typical range 0.3-1.5)
    sigma_score = min(sigma / 1.0, 1.5)

    # PRCC contribution (0-1, rank 1 = highest)
    if prcc_rank is not None:
        prcc_score = max(0, 1.0 - (prcc_rank - 1) / 194)
    else:
        prcc_score = 0.3  # unknown sensitivity → moderate

    # Target coverage penalty, weighted by how much existing targets
    # actually contracted the stage-1 posterior.
    effective_coverage = n_targets * max(0.0, min(1.0, stage1_effectiveness))
    coverage_penalty = 1.0 / (1 + effective_coverage)

    # Group membership discount (grouped params get partial pooling)
    group_factor = 0.7 if in_group else 1.0

    return sigma_score * prcc_score * coverage_penalty * group_factor


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def _score_params(priors, targets, groups, prcc, compare_results, cross_model, sbi_run=None):
    """Compute priority-scored list for all parameters.

    Sigma preference (tightest wins): stage-2 shell-local implied sigma →
    stage-1 joint posterior sigma → pdac_priors.csv prior sigma. The
    stage-2 shell-local sigma is derived from the mean contraction across
    `theta_test_local` in `local_calibration.csv` via
    ``sigma_shell = sigma_prior * sqrt(max(1 - contraction_local, 0))`` —
    this is the trustworthy SBC-style estimate of remaining uncertainty
    (the single-point posterior at x_obs can be tight as an NPE artifact,
    see the `artifact` flag in the stage-2 section).
    """
    import math

    local_map = {}
    if sbi_run is not None and sbi_run.get("local_calibration") is not None:
        lc_df = sbi_run["local_calibration"]
        if "contraction_local" in lc_df.columns:
            local_map = {
                row["parameter"]: row["contraction_local"]
                for _, row in lc_df.iterrows()
            }

    scored = []
    for name, info in priors.items():
        prcc_info = prcc.get(name, {})
        prcc_rank = prcc_info.get("rank") if prcc_info else None
        n_tgt = len(targets.get(name, []))

        sigma_val = info["sigma"]
        sigma_src = "prior"
        if name in compare_results.get("parameters", {}):
            joint = compare_results["parameters"][name].get("joint") or {}
            if joint.get("sigma") is not None:
                sigma_val = float(joint["sigma"])
                sigma_src = "S1"

        contr = local_map.get(name)
        if contr is not None and not math.isnan(float(contr)):
            sigma_shell = info["sigma"] * math.sqrt(max(1.0 - float(contr), 0.0))
            if sigma_shell < sigma_val:
                sigma_val = sigma_shell
                sigma_src = "S2"

        # Stage-1 effectiveness: how much did submodel-target inference
        # actually contract this param's posterior from the prior?
        # Missing S1 posterior → treat as fully effective (neutral) so
        # coverage_penalty reduces to the legacy (1+n_targets) form.
        s1_joint = (compare_results.get("parameters", {}).get(name, {}) or {}).get("joint") or {}
        s1_sigma_val = s1_joint.get("sigma")
        if s1_sigma_val is not None and info["sigma"] > 0:
            s1_effectiveness = max(
                0.0,
                min(1.0, 1.0 - (float(s1_sigma_val) / info["sigma"]) ** 2),
            )
        else:
            s1_effectiveness = 1.0

        xmodel = (cross_model or {}).get(name)
        scored.append({
            "name": name,
            "score": compute_priority(
                sigma_val, prcc_rank, n_tgt, name in groups, s1_effectiveness
            ),
            "sigma": sigma_val,
            "sigma_src": sigma_src,
            "units": info["units"],
            "prcc_rank": prcc_rank,
            "prcc_significant": prcc_info.get("significant", False),
            "n_sig_obs": prcc_info.get("n_sig_obs", 0),
            "group": groups.get(name),
            "type": classify_param(name),
            "n_targets": n_tgt,
            "s1_effectiveness": s1_effectiveness,
            "s1_sigma": float(s1_sigma_val) if s1_sigma_val is not None else None,
            "cross_model": _cross_model_summary(xmodel) if xmodel else "",
            "cross_n_models": xmodel["n_models"] if xmodel else 0,
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def _section_header(n_covered, n_total, has_prcc):
    """Report header with progress bar."""
    lines = ["# Parameter Coverage Audit\n"]
    pct = 100 * n_covered / n_total
    filled = round(pct / 5)
    bar = "=" * filled + "-" * (20 - filled)
    lines.append(f"**[{bar}] {n_covered}/{n_total} parameters covered ({pct:.0f}%)**\n")
    if not has_prcc:
        lines.append("> PRCC rankings are stale or missing. Run `python workflows/prcc_sensitivity.py` to update.\n")
    return lines


def _section_extraction_priority(scored, groups, targets, cross_model):
    """Section 1: What should I extract next?

    Splits candidates into two lists driven by different follow-ups:
      * **New target** — no existing submodel target; needs literature
        extraction.
      * **Improve existing** — has at least one target but stage-1
        inference barely moved the posterior from the prior; check
        whether the target is wrong, the forward-model submodel code is
        wrong, or the prior needs widening.
    """
    lines = ["---\n", "## What should I extract next?\n"]

    def _sigma_str(s):
        out = f"{s['sigma']:.2f}"
        if s["sigma_src"] != "prior":
            out += f" ({s['sigma_src']})"
        return out

    def _sig_obs_str(s):
        n = s.get("n_sig_obs", 0) or 0
        return str(n) if n else "---"

    def _render_table(rows, limit, include_effectiveness):
        has_xmodel = cross_model and any(r["cross_n_models"] > 0 for r in rows[:limit])
        hdr_cols = ["#", "Parameter", "Type", "Sigma", "PRCC", "#obs"]
        sep_cols = ["--:", "-----------", "------", "------:", "-----:", "----:"]
        if include_effectiveness:
            hdr_cols += ["Targets", "S1 eff"]
            sep_cols += [":-------:", "-----:"]
        else:
            hdr_cols += ["Targets"]
            sep_cols += [":-------:"]
        hdr_cols += ["Score"]
        sep_cols += ["------:"]
        if has_xmodel:
            hdr_cols += ["Models"]
            sep_cols += ["-------:"]
        lines.append("| " + " | ".join(hdr_cols) + " |")
        lines.append("|" + "|".join(sep_cols) + "|")
        for i, s in enumerate(rows[:limit], 1):
            rank = f"#{s['prcc_rank']}" if s["prcc_rank"] is not None else ""
            tgt_str = str(s["n_targets"]) if s["n_targets"] > 0 else "---"
            cells = [
                str(i),
                f"`{s['name']}`",
                s["type"],
                _sigma_str(s),
                rank,
                _sig_obs_str(s),
            ]
            if include_effectiveness:
                eff = f"{s['s1_effectiveness']:.0%}" if s["n_targets"] > 0 else "---"
                cells += [tgt_str, eff]
            else:
                cells += [tgt_str]
            cells += [f"{s['score']:.2f}"]
            if has_xmodel:
                cells += [f"{s['cross_n_models']}/6" if s["cross_n_models"] else "---"]
            lines.append("| " + " | ".join(cells) + " |")
        if len(rows) > limit:
            lines.append(f"| | *...{len(rows) - limit} more* | | | | | |")
        lines.append("")

    if scored:
        new_target = [s for s in scored if s["n_targets"] == 0]
        improve = [
            s for s in scored
            if s["n_targets"] > 0 and s["s1_effectiveness"] < 0.3
        ]
        improve.sort(key=lambda s: (s["s1_effectiveness"], -s["score"]))

        lines.append(
            "Ranked by impact (sigma × PRCC × effective-coverage). "
            "**#obs** = count of observables where this param is PRCC-significant "
            "(informs where to look for constraining data). "
            "**S1 eff** = how much submodel inference actually contracted the "
            "posterior from the prior; low values mean the existing target is "
            "ineffective and the param re-enters the priority list.\n"
        )

        if new_target:
            lines.append("### New target — literature extraction\n")
            _render_table(new_target, 20, include_effectiveness=False)

        if improve:
            lines.append("### Improve existing — ineffective target (S1 eff < 30%)\n")
            lines.append(
                "These params have at least one submodel target, but stage-1 "
                "inference barely moved the posterior. Likely causes: target "
                "is wrong, forward-model code is wrong, or the prior needs "
                "widening. Extract new data, re-audit existing target, or "
                "widen the prior — not the same action as the *new target* "
                "list above.\n"
            )
            _render_table(improve, 15, include_effectiveness=True)

    # Low-hanging fruit
    easy = [s for s in scored if s["type"] in ("EC50", "secretion") and s["sigma"] > 0.4]
    if easy:
        lines.append("### Low-hanging fruit\n")
        lines.append("EC50s → `direct_fit` (Hill curve from dose-response). Secretion rates → `batch_accumulation` (ELISA).\n")
        lines.append("| Parameter | Type | Sigma | PRCC | Units |")
        lines.append("|-----------|------|------:|-----:|-------|")
        for s in sorted(easy, key=lambda x: x["score"], reverse=True)[:15]:
            rank = f"#{s['prcc_rank']}" if s["prcc_rank"] is not None else ""
            lines.append(f"| `{s['name']}` | {s['type']} | {s['sigma']:.2f} | {rank} | {s['units']} |")
        lines.append("")

    # Parameter group gaps
    if groups:
        group_members = defaultdict(list)
        for name, gid in groups.items():
            group_members[gid].append((name, name in targets))
        incomplete = {
            gid: members for gid, members in group_members.items()
            if any(not c for _, c in members)
        }
        if incomplete:
            lines.append("### Parameter group gaps\n")
            lines.append("Extracting one member constrains the whole group via partial pooling.\n")
            lines.append("| Group | Coverage | Still needed |")
            lines.append("|-------|:--------:|-------------|")
            for gid, members in sorted(incomplete.items()):
                n_cov = sum(1 for _, c in members if c)
                needed = ", ".join(f"`{n}`" for n, c in members if not c)
                lines.append(f"| {gid} | {n_cov}/{len(members)} | {needed} |")
            lines.append("")

    # Type summary
    type_counts = defaultdict(int)
    for s in scored:
        type_counts[s["type"]] += 1
    type_summary = " | ".join(f"**{t}** {c}" for t, c in sorted(type_counts.items(), key=lambda x: -x[1]))
    lines.append(f"*Remaining by type: {type_summary}*\n")

    return lines


def _section_target_health(priors, targets, compare_results, output_dir, joint_samples, submodel_dir=None):
    """Section 2: Are my existing targets working?"""
    import math

    lines = ["---\n", "## Are my existing targets working?\n"]

    # Inference DAG
    if output_dir and submodel_dir:
        fig_dir = output_dir / "parameter_audit_report_files"
        dag_path = plot_inference_dag(submodel_dir, fig_dir)
        if dag_path:
            rel = dag_path.relative_to(output_dir)
            lines.append("### Inference DAG\n")
            lines.append(f"![Inference DAG]({rel})\n")

    has_problems = False
    cr_params = compare_results["parameters"]

    # Conflicting targets
    tension_params = {k: v for k, v in cr_params.items() if v.get("tension")}
    if tension_params:
        has_problems = True
        lines.append("### Conflicting targets\n")
        lines.append("These parameters get >3x different estimates from different targets. The targets may have incompatible data, missing Hill functions, or unit errors.\n")
        for pname, pdata in sorted(tension_params.items()):
            singles = pdata.get("single_targets", [])
            medians = [s["median"] for s in singles]
            ratio = max(medians) / min(medians) if min(medians) > 0 else float("inf")
            lines.append(f"**`{pname}`** ({ratio:.1f}x spread)\n")
            lines.append("| Target | Median | Contraction |")
            lines.append("|--------|-------:|:-----------:|")
            for s in singles:
                lines.append(f"| {s['target_id']} | {s['median']:.4g} | {s['contraction']:.0%} |")
            lines.append("")

    # Weak contraction
    weak_params = [
        (k, v) for k, v in cr_params.items()
        if v.get("flags") and "weak_contraction" in v["flags"]
    ]
    if weak_params:
        has_problems = True
        lines.append("### Weak targets\n")
        lines.append("These have submodel targets but the data barely moved the prior. The target may be uninformative, or the translation sigma is too large.\n")
        lines.append("| Parameter | Contraction | Targets |")
        lines.append("|-----------|:-----------:|---------|")
        for pname, pdata in sorted(weak_params, key=lambda x: x[1].get("joint", {}).get("contraction", 0)):
            contr = pdata.get("joint", {}).get("contraction", 0)
            tgt_list = ", ".join(targets.get(pname, []))
            lines.append(f"| `{pname}` | {contr:.0%} | {tgt_list} |")
        lines.append("")

    # MCMC warnings
    diag_warnings = [
        (pname, pdata.get("mcmc_diagnostics", {}))
        for pname, pdata in cr_params.items()
        if (pdata.get("mcmc_diagnostics", {}).get("n_eff") or 999) < 100
        or (pdata.get("mcmc_diagnostics", {}).get("r_hat") or 1.0) > 1.05
    ]
    if diag_warnings:
        has_problems = True
        lines.append("### MCMC issues\n")
        for pname, diag in diag_warnings:
            lines.append(f"- `{pname}`: n_eff={diag.get('n_eff', '?')}, r_hat={diag.get('r_hat', '?')}")
        lines.append("")

    n_div = compare_results["metadata"].get("num_divergences", 0)
    if n_div > 0:
        lines.append(f"> Joint MCMC had **{n_div} divergences** — may indicate prior-model conflict.\n")

    # SBC diagnostics
    sbc_params = {k: v for k, v in cr_params.items() if v.get("sbc")}
    if sbc_params:
        sbc_fail = {k: v for k, v in sbc_params.items() if not v["sbc"]["calibrated"]}
        sbc_pass = {k: v for k, v in sbc_params.items() if v["sbc"]["calibrated"]}
        n_sbc = len(sbc_params)
        lines.append("### SBC calibration (NPE components)\n")
        lines.append(f"{len(sbc_pass)}/{n_sbc} params well-calibrated (KS p > 0.01)\n")
        lines.append("| Parameter | KS p | Bias | Dispersion | Interpretation |")
        lines.append("|-----------|:----:|:----:|:----------:|----------------|")
        for pname in sorted(sbc_params):
            sbc = sbc_params[pname]["sbc"]
            ranks = sbc.get("ranks", [])
            expected = sbc.get("expected_rank", 100)
            mean_rank = sbc.get("mean_rank", expected)
            ks_p = sbc.get("ks_p", 1.0)

            bias_pct = (mean_rank - expected) / expected * 100
            n_post = int(expected * 2)
            uniform_std = n_post / math.sqrt(12)
            if ranks:
                import numpy as _np
                dispersion = float(_np.std(ranks)) / uniform_std
            else:
                dispersion = 1.0

            parts = []
            if abs(bias_pct) > 5:
                parts.append("biased " + ("low" if bias_pct > 0 else "high"))
            if dispersion > 1.15:
                parts.append("overconfident")
            elif dispersion < 0.85:
                parts.append("underconfident")
            interp = ", ".join(parts) if parts else ("OK" if ks_p > 0.01 else "marginal")

            lines.append(
                f"| `{pname}` | {ks_p:.4f} | {bias_pct:+.1f}% | {dispersion:.2f} | {interp} |"
            )
        lines.append("")
        lines.append(
            "> **Bias**: (mean_rank - expected) / expected. "
            "Positive = posterior biased low. "
            "**Dispersion**: std(ranks) / std(uniform). "
            ">1 = overconfident (too narrow), <1 = underconfident (too wide).\n"
        )
        if sbc_fail:
            has_problems = True

    # PPC
    ppc_obs_list = compare_results.get("ppc_observables", [])
    meta = compare_results.get("metadata", {})
    ppc_n = meta.get("ppc_n_total", 0)
    ppc_cov = meta.get("ppc_n_covered", 0)
    if ppc_n > 0:
        lines.append("### Posterior predictive coverage\n")
        lines.append(f"**{ppc_cov}/{ppc_n}** observables covered by 95% CI ({100*ppc_cov/ppc_n:.0f}%)\n")
        if output_dir and ppc_obs_list:
            fig_dir = output_dir / "parameter_audit_report_files"
            ppc_plot = plot_ppc_histograms(ppc_obs_list, fig_dir)
            if ppc_plot:
                rel = ppc_plot.relative_to(output_dir)
                lines.append(f"![PPC Histograms]({rel})\n")

    if not has_problems:
        lines.append("No issues detected. All targets are constraining their parameters.\n")

    # Contraction summary
    lines.append("### Contraction summary\n")
    all_constrained = []
    for name in sorted(cr_params.keys()):
        pdata = cr_params[name]
        prior_sigma = priors.get(name, {}).get("sigma")
        joint = pdata.get("joint") or {}
        contr = joint.get("contraction")
        if contr is None:
            continue
        all_constrained.append((name, prior_sigma, joint.get("sigma"), joint.get("median"), contr))

    all_constrained.sort(key=lambda x: -x[4])
    lines.append("| Parameter | Prior Sigma | Post Sigma | Median | Contraction | n_eff | r_hat |")
    lines.append("|-----------|:----------:|:---------:|-------:|:-----------:|------:|------:|")
    for name, ps, post_s, med, contr in all_constrained:
        pdata = cr_params.get(name, {})
        mcmc = pdata.get("mcmc_diagnostics", {})
        neff = f"{mcmc['n_eff']:.0f}" if mcmc.get("n_eff") else "---"
        rhat = f"{mcmc['r_hat']:.3f}" if mcmc.get("r_hat") and mcmc["r_hat"] != 1.0 else "---"
        lines.append(
            f"| `{name}` | {_fmt_sigma(ps)} | {_fmt_sigma(post_s)} | "
            f"{f'{med:.4g}' if med is not None else '---'} | "
            f"{_contraction_bar(contr)} | {neff} | {rhat} |"
        )
    lines.append("")

    # Marginals plot
    if output_dir:
        fig_dir = output_dir / "parameter_audit_report_files"
        marginals_path = plot_marginals(priors, compare_results, joint_samples, fig_dir)
        if marginals_path:
            rel = marginals_path.relative_to(output_dir)
            lines.append("### Prior vs posterior marginals\n")
            lines.append(f"![Prior vs posterior marginals]({rel})\n")

    return lines


def _section_whats_left(priors, targets, groups, prcc, sbi_runs):
    """Section 3: What's left before stage 2?"""
    lines = ["---\n", "## What's left before stage 2?\n"]

    n_total = len(priors)
    n_covered = sum(1 for p in priors if p in targets)
    pct = 100 * n_covered / n_total
    lines.append(f"**{n_covered}/{n_total}** parameters have submodel targets ({pct:.0f}%).\n")

    if prcc:
        sig_params = [k for k, v in prcc.items() if v.get("significant")]
        sig_covered = sum(1 for p in sig_params if p in targets)
        lines.append(
            f"Of **{len(sig_params)} PRCC-significant** parameters, "
            f"**{sig_covered}** have targets ({100*sig_covered/max(len(sig_params),1):.0f}%).\n"
        )

    lines.append(f"### Covered ({n_covered})\n")
    lines.append("| Parameter | Type | Targets | Group | PRCC |")
    lines.append("|-----------|------|:-------:|-------|-----:|")
    for name in sorted(targets.keys()):
        if name not in priors:
            continue
        n_t = len(targets[name])
        group = groups.get(name, "")
        rank = prcc.get(name, {}).get("rank")
        rank_str = f"#{rank}" if rank else ""
        lines.append(f"| `{name}` | {classify_param(name)} | {n_t} | {group} | {rank_str} |")
    lines.append("")

    if sbi_runs:
        lines.append("### Stage 2 runs available\n")
        for run in sbi_runs:
            posteriors = load_posterior_summary(run["summary_path"])
            n_post = len(posteriors) if posteriors else 0
            lines.append(f"- **{run['name']}** — {n_post} parameters")
        lines.append("")

    return lines


def _section_cross_model(priors, targets, compare_results, cross_model):
    """Appendix: Cross-model parameter comparison."""
    if not cross_model:
        return []

    matched = [(name, cross_model[name]) for name in sorted(priors) if name in cross_model]
    if not matched:
        return []

    lines = ["---\n", "## Appendix: Cross-model parameter comparison\n"]
    lines.append(
        "Values used for the same parameter across QSPIO cancer models "
        "(TNBC, CRC, UM, NSCLC, HCC, PDAC). "
        "Parameters where our PDAC value differs substantially from other models "
        "may warrant validation.\n"
    )
    lines.append(
        "| Parameter | Ref name | TNBC | CRC | UM | NSCLC | HCC | PDAC (ref) | PDAC (current) | Unit |"
    )
    lines.append(
        "|-----------|----------|------|-----|-----|-------|-----|------------|:--------------:|------|"
    )
    for name, xm in matched:
        ref_name = xm.get("ref_name", name)
        alias_col = f"`{ref_name}`" if ref_name != name else ""
        cols = [xm.get(m, "") or "---" for m in ["TNBC", "CRC", "UM", "NSCLC", "HCC"]]
        cols = [c[:18] if len(c) > 18 else c for c in cols]
        pdac_ref = xm.get("PDAC", "") or "---"
        if len(pdac_ref) > 18:
            pdac_ref = pdac_ref[:18]
        post_median = None
        if name in targets and name in compare_results.get("parameters", {}):
            joint = compare_results["parameters"][name].get("joint") or {}
            post_median = joint.get("median")
        current_str = f"**{post_median:.4g}**" if post_median is not None else f"{priors[name]['median']:.4g}"
        unit = xm.get("unit", "")
        lines.append(
            f"| `{name}` | {alias_col} | {cols[0]} | {cols[1]} | {cols[2]} | "
            f"{cols[3]} | {cols[4]} | {pdac_ref} | {current_str} | {unit} |"
        )
    lines.append("")
    lines.append(
        "> **Bold** PDAC (current) values are posterior medians from submodel inference; "
        "plain values are prior medians.\n"
    )

    # Fixed parameters
    fixed = [(name, cross_model[name]) for name in sorted(cross_model) if name not in priors]
    if fixed:
        lines.append("### Fixed / derived parameters\n")
        lines.append(
            "Cancer-type-specific parameters not varied in SBI "
            "(constants, PK params, or derived quantities).\n"
        )
        lines.append(
            "| Parameter | TNBC | CRC | UM | NSCLC | HCC | PDAC (ref) | Unit |"
        )
        lines.append(
            "|-----------|------|-----|-----|-------|-----|------------|------|"
        )
        for name, xm in fixed:
            cols = [xm.get(m, "") or "---" for m in ["TNBC", "CRC", "UM", "NSCLC", "HCC"]]
            cols = [c[:18] if len(c) > 18 else c for c in cols]
            pdac_ref = xm.get("PDAC", "") or "---"
            if len(pdac_ref) > 18:
                pdac_ref = pdac_ref[:18]
            unit = xm.get("unit", "")
            lines.append(
                f"| `{name}` | {cols[0]} | {cols[1]} | {cols[2]} | "
                f"{cols[3]} | {cols[4]} | {pdac_ref} | {unit} |"
            )
        lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _section_component_diagnostics(comp_diags):
    """Section: Per-component inference diagnostics."""
    if not comp_diags:
        return []

    lines = ["---\n", "## Component diagnostics\n"]

    # Summary
    n_comps = len(comp_diags)
    n_nuts = sum(1 for c in comp_diags if c["method"] == "NUTS")
    n_npe = n_comps - n_nuts
    total_div = sum(c["num_divergences"] for c in comp_diags)
    total_ppc_cov = sum(c["ppc_n_covered"] for c in comp_diags)
    total_ppc_tot = sum(c["ppc_n_total"] for c in comp_diags)
    lines.append(
        f"**{n_comps} components** ({n_nuts} NUTS, {n_npe} NPE) — "
        f"**{total_div} total divergences** — "
        f"**{total_ppc_cov}/{total_ppc_tot}** PPC observables covered\n"
    )

    # Classify components by health status
    def _has_issues(c):
        if c["num_divergences"] > 0:
            return True
        if c["ppc_n_total"] > 0 and c["ppc_n_covered"] < c["ppc_n_total"]:
            return True
        # NPE: check SBC failures
        if c["method"] == "NPE":
            sbc = c.get("sbc", {})
            if any(not v.get("calibrated", True) for v in sbc.values()):
                return True
        return False

    problem_comps = [c for c in comp_diags if _has_issues(c)]
    healthy_comps = [c for c in comp_diags if not _has_issues(c)]

    if problem_comps:
        lines.append("### Components with issues\n")

        # NUTS problems
        nuts_problems = [c for c in problem_comps if c["method"] == "NUTS"]
        if nuts_problems:
            lines.append("**NUTS components:**\n")
            lines.append("| Params | Divergences | PPC | Parameters |")
            lines.append("|:------:|:-----------:|:---:|-----------|")
            for c in sorted(nuts_problems, key=lambda x: -x["num_divergences"]):
                ppc = f"{c['ppc_n_covered']}/{c['ppc_n_total']}" if c["ppc_n_total"] else "---"
                param_str = ", ".join(f"`{p}`" for p in c["params"][:6])
                if len(c["params"]) > 6:
                    param_str += f", +{len(c['params']) - 6} more"
                div_str = f"**{c['num_divergences']}**" if c["num_divergences"] > 0 else "0"
                lines.append(f"| {len(c['params'])} | {div_str} | {ppc} | {param_str} |")
            lines.append("")

            # Detail tables for NUTS components with divergences
            for c in sorted(nuts_problems, key=lambda x: -x["num_divergences"]):
                if c["num_divergences"] == 0 or not c["per_param"]:
                    continue
                param_list = ", ".join(f"`{p}`" for p in c["params"][:4])
                if len(c["params"]) > 4:
                    param_list += ", ..."
                lines.append(f"<details><summary><b>{c['num_divergences']} divergences</b> — {param_list}</summary>\n")
                lines.append("| Parameter | n_eff | r_hat | Contraction | z_score |")
                lines.append("|-----------|------:|------:|:-----------:|--------:|")
                for pname in c["params"]:
                    pp = c["per_param"].get(pname, {})
                    n_eff = pp.get("n_eff")
                    r_hat = pp.get("r_hat")
                    contr = pp.get("contraction")
                    z = pp.get("z_score")
                    n_eff_str = f"{n_eff:.0f}" if n_eff else "---"
                    r_hat_str = f"{r_hat:.4f}" if r_hat else "---"
                    contr_str = f"{contr:.0%}" if contr is not None else "---"
                    z_str = f"{z:.2f}" if z is not None else "---"
                    lines.append(f"| `{pname}` | {n_eff_str} | {r_hat_str} | {contr_str} | {z_str} |")
                lines.append("\n</details>\n")

        # NPE problems
        npe_problems = [c for c in problem_comps if c["method"] == "NPE"]
        if npe_problems:
            lines.append("**NPE components:**\n")
            for c in npe_problems:
                param_list = ", ".join(f"`{p}`" for p in c["params"][:6])
                if len(c["params"]) > 6:
                    param_list += f", +{len(c['params']) - 6} more"
                ppc = f"{c['ppc_n_covered']}/{c['ppc_n_total']}" if c["ppc_n_total"] else "---"
                lines.append(f"**{len(c['params'])} params** ({param_list}) — PPC {ppc}\n")

                # SBC results
                sbc = c.get("sbc", {})
                if sbc:
                    failed_sbc = {k: v for k, v in sbc.items() if not v.get("calibrated", True)}
                    if failed_sbc:
                        lines.append("SBC failures:\n")
                        lines.append("| Parameter | Issue | KS p-value |")
                        lines.append("|-----------|-------|:----------:|")
                        for pname, sv in sorted(failed_sbc.items()):
                            issue = sv.get("issue", "?")
                            ks_p = sv.get("ks_p", 0)
                            lines.append(f"| `{pname}` | {issue} | {ks_p:.2e} |")
                        lines.append("")


    # Healthy summary
    if healthy_comps:
        lines.append(f"### Healthy components ({len(healthy_comps)})\n")
        lines.append(
            f"{len(healthy_comps)} components with 0 divergences, full PPC coverage"
            + (", and passing SBC." if any(c["method"] == "NPE" for c in healthy_comps) else ".")
            + "\n"
        )

    return lines


def load_submodel_priors(path: Path) -> dict:
    """Load submodel_priors.yaml → {name: {mu, sigma, median}} (log-space marginals).

    Only lognormal marginals are used; others are skipped (no comparable σ).
    """
    if path is None or not Path(path).exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    out = {}
    for entry in data.get("parameters", []) or []:
        name = entry.get("name")
        m = entry.get("marginal", {}) or {}
        if not name or m.get("distribution") != "lognormal":
            continue
        try:
            import math
            mu = float(m["mu"])
            sigma = float(m["sigma"])
            median = m.get("median")
            # Fall back to exp(mu) when YAML dumped zero/None for tiny lognormals.
            median_f = float(median) if median else math.exp(mu)
            out[name] = {"mu": mu, "sigma": sigma, "median": median_f}
        except (KeyError, TypeError, ValueError):
            continue
    return out


def load_sbi_run(run_path: Path) -> dict | None:
    """Load SBI run artifacts (posterior samples + diagnostics + metadata).

    Returns None when the pinned run directory is missing required files.
    """
    import numpy as np
    import pandas as pd

    if run_path is None or not run_path.exists():
        return None

    post_samples_path = run_path / "posterior_samples.csv"
    diag_dir = run_path / "diagnostics"
    zc_path = diag_dir / "z_score_contraction.csv"
    summary_path = diag_dir / "summary.json"
    local_path = run_path / "local_calibration.csv"

    if not post_samples_path.exists():
        print(f"[stage2] posterior_samples.csv not found in {run_path}")
        return None

    post_df = pd.read_csv(post_samples_path)

    zc_df = pd.read_csv(zc_path) if zc_path.exists() else None
    local_df = pd.read_csv(local_path) if local_path.exists() else None

    meta = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        meta = summary.get("training", {})

    # Observed-data posterior stats (log-space on positive samples), mirroring
    # the stage-1 contraction formula in load_compare_results.
    observed: dict[str, dict] = {}
    for pname in post_df.columns:
        if pname.startswith("Unnamed:"):
            continue
        arr = post_df[pname].to_numpy(dtype=float)
        pos = arr[arr > 0]
        if len(pos) < 10:
            continue
        log_samps = np.log(pos)
        observed[pname] = {
            "median": float(np.median(pos)),
            "post_mu": float(np.mean(log_samps)),
            "post_sigma": float(np.std(log_samps)),
        }

    return {
        "run_path": run_path,
        "observed": observed,
        "z_score_contraction": zc_df,
        "local_calibration": local_df,
        "metadata": meta,
    }


def _section_stage2_npe(
    priors: dict,
    stage1_priors: dict,
    sbi_run: dict | None,
) -> list[str]:
    """Section: Stage 2 NPE posterior shifts vs pre-stage-1 and post-stage-1 priors.

    Reports per-parameter contraction and z-score at the observed data
    (same formulas as the submodel stage-1 section) against two references:
      - pre: pdac_priors.csv (total calibration effect)
      - post-S1: submodel_priors.yaml fit marginals (stage 2 marginal effect)
    Also surfaces the SBC-style median contraction from z_score_contraction.csv.
    """
    if sbi_run is None:
        return []

    observed = sbi_run["observed"]
    if not observed:
        return []

    meta = sbi_run["metadata"]
    zc_df = sbi_run["z_score_contraction"]
    zc_map = (
        {row["parameter"]: row for _, row in zc_df.iterrows()} if zc_df is not None else {}
    )
    local_df = sbi_run.get("local_calibration")
    local_map = (
        {row["parameter"]: row for _, row in local_df.iterrows()}
        if local_df is not None else {}
    )

    lines = ["---\n", "## Stage 2: NPE posterior shifts\n"]
    run_path = sbi_run["run_path"]
    lines.append(
        f"Run: `{run_path.name}` — contraction + shift against pre-stage-1 prior "
        "(`pdac_priors.csv`) and post-stage-1 prior (`submodel_priors.yaml`).\n"
    )

    bits = []
    if meta.get("prcc_tier") is not None:
        bits.append(f"PRCC tier {meta['prcc_tier']} ({meta.get('n_params', '?')} of {meta.get('n_params_full', '?')} params)")
    if meta.get("scenarios"):
        bits.append(f"{len(meta['scenarios'])} scenarios")
    if meta.get("n_train") is not None:
        bits.append(f"n_train={meta['n_train']}")
    if stage1_priors:
        bits.append(f"{len(stage1_priors)} params with stage-1 posterior")
    if bits:
        lines.append(" · ".join(bits) + "\n")

    def _stats(post_sigma: float, post_mu: float, ref: dict | None):
        if not ref or ref.get("sigma", 0) <= 0:
            return None, None
        contr = 1.0 - (post_sigma / ref["sigma"]) ** 2
        z = abs(post_mu - ref["mu"]) / ref["sigma"]
        return contr, z

    rows = []
    for pname, stats in observed.items():
        prior = priors.get(pname)
        if not prior or prior.get("sigma", 0) <= 0:
            continue
        c_pre, z_pre = _stats(stats["post_sigma"], stats["post_mu"], prior)
        s1 = stage1_priors.get(pname)
        c_s1, z_s1 = _stats(stats["post_sigma"], stats["post_mu"], s1)
        zc = zc_map.get(pname, {})
        lc = local_map.get(pname, {})
        rows.append({
            "name": pname,
            "prior_median": prior["median"],
            "prior_sigma": prior["sigma"],
            "s1_median": s1["median"] if s1 else None,
            "s1_sigma": s1["sigma"] if s1 else None,
            "post_median": stats["median"],
            "post_sigma": stats["post_sigma"],
            "contraction_pre": c_pre,
            "z_pre": z_pre,
            "contraction_s1": c_s1,
            "z_s1": z_s1,
            "contraction_sbc": zc.get("contraction_median"),
            "contraction_global": lc.get("contraction_global") if lc is not None else None,
            "contraction_local_shell": lc.get("contraction_local") if lc is not None else None,
            "ks_p_global": lc.get("ks_pval_global") if lc is not None else None,
            "ks_p_local": lc.get("ks_pval_local") if lc is not None else None,
            "sbc_z_abs_mean": zc.get("z_score_abs_mean"),
            "sbc_z_mean": zc.get("z_score_mean"),
        })

    if not rows:
        lines.append("_No parameters matched the priors CSV — check parameter names._\n")
        return lines

    # Sort by the trustworthy metric: mean contraction over theta_test drawn
    # from a shell around x_obs. The single-point `contraction_pre` at x_obs
    # can look tight as a flow artifact (see artifact flag below), so ranking
    # by it puts non-identified params at the top. Fall back to the global
    # mean contraction, then the single-point value, when shell/global are
    # unavailable.
    def _sort_key(r):
        for k in ("contraction_local_shell", "contraction_global", "contraction_pre"):
            v = r.get(k)
            if v is not None:
                return -v
        return 0.0

    rows.sort(key=_sort_key)

    def _strong(r):
        v = r.get("contraction_local_shell")
        if v is None:
            v = r.get("contraction_global")
        if v is None:
            v = r.get("contraction_pre")
        return v is not None and v >= 0.5

    def _weak(r):
        v = r.get("contraction_local_shell")
        if v is None:
            v = r.get("contraction_global")
        if v is None:
            v = r.get("contraction_pre")
        return v is not None and v < 0.1

    n_strong_shell = sum(1 for r in rows if _strong(r))
    n_weak_shell = sum(1 for r in rows if _weak(r))
    n_shift = sum(1 for r in rows if r["z_pre"] > 2)
    s1_rows = [r for r in rows if r["contraction_s1"] is not None]
    lines.append(
        f"shell-local: **{n_strong_shell}/{len(rows)}** contraction ≥50%, "
        f"**{n_weak_shell}** below 10% (identifiability near x_obs — the "
        "trustworthy metric; table sorted by this column).\n"
    )
    lines.append(f"vs pre (single-point at x_obs): **{n_shift}** with |z|>2.\n")
    if s1_rows:
        n_s1_strong = sum(1 for r in s1_rows if r["contraction_s1"] >= 0.5)
        n_s1_weak = sum(1 for r in s1_rows if r["contraction_s1"] < 0.1)
        n_s1_shift = sum(1 for r in s1_rows if r["z_s1"] > 2)
        lines.append(
            f"vs stage-1 ({len(s1_rows)} params): **{n_s1_strong}** ≥50%, "
            f"**{n_s1_weak}** below 10%, **{n_s1_shift}** with |z|>2.\n"
        )

    lines.append(
        "| Parameter | σ pre | σ S1 | σ post | Median pre | Median S1 | Median post | "
        "z (pre) | Contr (pre) | z (S1) | Contr (S1) | Contr global | Contr local-shell | "
        "KS p local | z̄ SBC | Flags |"
    )
    lines.append(
        "|-----------|:----:|:----:|:-----:|-----------:|----------:|------------:|"
        ":------:|:-----------:|:------:|:----------:|:------------:|:-----------------:|"
        ":---------:|:-----:|:-----:|"
    )
    for r in rows:
        flags = []
        # Single-point artifact: tight posterior at x_obs that does not
        # persist when averaging over nearby test observations — classic NPE
        # single-point overfit rather than genuine identification.
        if (
            r["contraction_pre"] >= 0.5
            and r["contraction_local_shell"] is not None
            and r["contraction_local_shell"] < 0.2
        ):
            flags.append("artifact")
        if r["contraction_pre"] < 0.1:
            flags.append("weak_pre")
        if r["z_pre"] > 2:
            flags.append("shift_pre")
        if r["contraction_s1"] is not None and r["z_s1"] > 2:
            flags.append("shift_S1")
        if r["contraction_sbc"] is not None and r["contraction_sbc"] < 0.1:
            flags.append("non-id")
        if r["ks_p_local"] is not None and r["ks_p_local"] < 0.05:
            flags.append("miscal")
        if r["sbc_z_abs_mean"] is not None and r["sbc_z_abs_mean"] > 1.0:
            flags.append("biased")
        flag_str = ", ".join(flags) if flags else ""
        c_global = (
            f"{r['contraction_global']:.0%}" if r["contraction_global"] is not None
            else (f"{r['contraction_sbc']:.0%}" if r["contraction_sbc"] is not None else "---")
        )
        c_local = (
            _contraction_bar(r["contraction_local_shell"])
            if r["contraction_local_shell"] is not None else "---"
        )
        s1_med = f"{r['s1_median']:.4g}" if r["s1_median"] is not None else "---"
        s1_z = f"{r['z_s1']:.2f}" if r["z_s1"] is not None else "---"
        s1_contr = _contraction_bar(r["contraction_s1"]) if r["contraction_s1"] is not None else "---"
        ks_str = f"{r['ks_p_local']:.2g}" if r["ks_p_local"] is not None else "---"
        z_sbc_str = f"{r['sbc_z_abs_mean']:.2f}" if r["sbc_z_abs_mean"] is not None else "---"
        lines.append(
            f"| `{r['name']}` | {_fmt_sigma(r['prior_sigma'])} | {_fmt_sigma(r['s1_sigma'])} | "
            f"{_fmt_sigma(r['post_sigma'])} | {r['prior_median']:.4g} | {s1_med} | {r['post_median']:.4g} | "
            f"{r['z_pre']:.2f} | {_contraction_bar(r['contraction_pre'])} | "
            f"{s1_z} | {s1_contr} | {c_global} | {c_local} | "
            f"{ks_str} | {z_sbc_str} | {flag_str} |"
        )
    lines.append("")
    lines.append(
        "> **pre** = `pdac_priors.csv` (pre-stage-1 baseline). "
        "**S1** = `submodel_priors.yaml` (post-stage-1; empty when the parameter has no stage-1 target). "
        "**z (pre/S1)** and **Contr (pre/S1)** are _local_ metrics at the observed data: "
        "`|μ_post − μ_ref|/σ_ref` and `1 − (σ_post/σ_ref)²` on posterior samples at x_obs. "
        "**Contr global** = mean contraction across `theta_test` ~ prior (from `local_calibration.csv`, "
        "fallback `z_score_contraction.csv` median) — identifiability under the prior. "
        "**Contr local-shell** = mean contraction across `theta_test_local` (shell near x_obs) from `local_calibration.csv` — identifiability in the neighborhood of the observation. "
        "**KS p local** = p-value of KS test on posterior z-scores in the shell around x_obs (`ks_pval_local`) — calibration check; low p ⇒ posterior ranks deviate from uniform, flagged `miscal`. "
        "**z̄ SBC** = mean |z-score| over `theta_test ~ prior` (`z_score_abs_mean`) — global bias; values >1 flagged `biased` (posterior mean systematically offset from truth). "
        "**`artifact`** flag = single-point posterior at x_obs looks tight (`Contr (pre) ≥ 50%`) but fails to hold up under shell averaging (`Contr local-shell < 20%`) — the concentration is a flow artifact, not genuine identification.\n"
    )
    return lines


def generate_report(
    priors: dict,
    targets: dict,
    groups: dict,
    prcc: dict,
    compare_results: dict,
    sbi_runs: list,
    joint_samples: dict | None = None,
    comp_diags: list | None = None,
    output_dir: Path | None = None,
    cross_model: dict | None = None,
    submodel_dir: Path | None = None,
    sbi_run: dict | None = None,
    stage1_priors: dict | None = None,
) -> str:
    n_total = len(priors)
    n_covered = sum(1 for p in priors if p in targets)
    scored = _score_params(priors, targets, groups, prcc, compare_results, cross_model, sbi_run=sbi_run)

    lines = []
    lines.extend(_section_header(n_covered, n_total, bool(prcc)))
    lines.extend(_section_extraction_priority(scored, groups, targets, cross_model))
    lines.extend(_section_target_health(priors, targets, compare_results, output_dir, joint_samples, submodel_dir=submodel_dir))
    lines.extend(_section_component_diagnostics(comp_diags))
    lines.extend(_section_whats_left(priors, targets, groups, prcc, sbi_runs))
    lines.extend(_section_stage2_npe(priors, stage1_priors or {}, sbi_run))
    lines.extend(_section_cross_model(priors, targets, compare_results, cross_model))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Submodel priors export
# ---------------------------------------------------------------------------


def _write_submodel_priors(
    joint_samples: dict[str, list],
    targets: dict[str, list[str]],
    groups: dict[str, str],
    output_path: Path,
    copula_threshold: float = 0.05,
) -> None:
    """Write submodel_priors.yaml from cached joint posterior samples.

    Uses fit_marginals and fit_gaussian_copula from the parameterizer module
    but avoids loading SubmodelTarget Pydantic objects.

    Args:
        joint_samples: {param_name: list[float]} from load_joint_samples
        targets: {param_name: [target_ids]} from load_submodel_targets
        groups: {param_name: group_id} from load_parameter_groups
        output_path: Where to write submodel_priors.yaml
        copula_threshold: Minimum |correlation| to include in copula
    """
    import numpy as np

    from qsp_inference.submodel.parameterizer import (
        _build_marginal_cdf,
        fit_gaussian_copula,
        fit_marginals,
        threshold_copula,
        write_priors_yaml,
    )

    # Filter to non-nuisance, non-hyperparameter samples
    output_samples = {
        k: np.asarray(v)
        for k, v in joint_samples.items()
        if k in targets and "__" not in k
    }

    if not output_samples:
        print("No posterior samples to parameterize — skipping submodel_priors.yaml")
        return

    param_names = sorted(output_samples.keys())

    marginals = fit_marginals(output_samples)

    # Build copula block-diagonally.  Parameters from different inference
    # components are independent (inferred separately) so their correlation
    # is exactly 0.  Group by sample length as a proxy for component
    # membership — same-length samples are paired MCMC/NPE draws.
    if len(param_names) > 1:
        from collections import defaultdict

        size_groups: dict[int, list[str]] = defaultdict(list)
        for n in param_names:
            size_groups[len(output_samples[n])].append(n)

        name_to_idx = {n: i for i, n in enumerate(param_names)}
        R = np.eye(len(param_names))

        for group_names in size_groups.values():
            if len(group_names) < 2:
                continue
            block_matrix = np.column_stack(
                [output_samples[n] for n in group_names]
            )
            block_cdfs = [_build_marginal_cdf(marginals[n]) for n in group_names]
            R_block = fit_gaussian_copula(block_matrix, block_cdfs)
            for bi, ni in enumerate(group_names):
                for bj, nj in enumerate(group_names):
                    R[name_to_idx[ni], name_to_idx[nj]] = R_block[bi, bj]

        R_thresh, copula_params = threshold_copula(R, param_names, copula_threshold)
    else:
        R_thresh = np.array([[1.0]])
        copula_params = []

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
    result = {
        "metadata": {
            "generated_by": "qsp_inference.audit.report",
            "n_parameters": len(param_names),
            "n_samples": n_samples,
        },
        "parameters": parameters,
    }
    if copula_block:
        result["copula"] = copula_block

    write_priors_yaml(result, output_path)
    print(f"Submodel priors written to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run_inference(config: AuditConfig, invalidate_params=None):
    """Run component-wise inference to generate/update cached results."""
    from qsp_inference.submodel.comparison import run_comparison

    if invalidate_params:
        print(f"Invalidating components containing: {invalidate_params}")
    print("Running inference comparison...")
    run_comparison(
        priors_csv=str(config.priors_csv),
        submodel_dir=str(config.submodel_dir),
        num_samples=4000,
        invalidate_params=invalidate_params,
    )
    print("Done.")


def run_audit(config: AuditConfig, output: Path | None = None, invalidate_params=None) -> str:
    """Run the full audit pipeline and return the report as a string.

    Args:
        config: Paths configuration.
        output: Optional path to write the markdown report.
        invalidate_params: Optional list of parameter names to invalidate
            in the inference cache before re-running.

    Returns:
        The generated markdown report.
    """
    _run_inference(config, invalidate_params=invalidate_params)

    priors = load_priors(config.priors_csv)
    targets = load_submodel_targets(config.submodel_dir)
    groups = load_parameter_groups(config.param_groups)
    prcc = load_prcc(config.prcc_csv)
    cross_model = load_cross_model(config.cross_model_csv, model_names=config.model_names)
    compare_results = load_compare_results(config.compare_cache, priors=priors)
    sbi_runs = find_sbi_posteriors(config.sbi_dir)

    joint_samples = load_joint_samples(config.compare_cache)
    if joint_samples:
        print(f"Loaded joint samples for {len(joint_samples)} parameters")

    comp_diags = load_component_diagnostics(config.compare_cache)
    if comp_diags:
        print(f"Loaded diagnostics for {len(comp_diags)} components")

    sbi_run = load_sbi_run(config.sbi_run_path) if config.sbi_run_path else None
    if sbi_run:
        print(f"Loaded SBI run: {config.sbi_run_path}")

    stage1_priors = load_submodel_priors(config.submodel_priors_yaml)
    if stage1_priors:
        print(f"Loaded stage-1 priors for {len(stage1_priors)} parameters")

    output_dir = output.parent if output else None

    report = generate_report(
        priors, targets, groups, prcc, compare_results, sbi_runs,
        joint_samples=joint_samples,
        comp_diags=comp_diags,
        output_dir=output_dir,
        cross_model=cross_model,
        submodel_dir=config.submodel_dir,
        sbi_run=sbi_run,
        stage1_priors=stage1_priors,
    )

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report)
        print(f"Report written to {output}")

    # Generate submodel_priors.yaml from joint posterior samples
    if joint_samples:
        priors_yaml_path = (output.parent if output else Path.cwd()) / "submodel_priors.yaml"
        _write_submodel_priors(joint_samples, targets, groups, priors_yaml_path)

    return report


def main():
    parser = argparse.ArgumentParser(description="Parameter coverage audit")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown file (default: print to stdout)",
    )
    parser.add_argument(
        "--priors-csv",
        type=Path,
        default=None,
        help="Path to priors CSV (default: <root>/parameters/pdac_priors.csv)",
    )
    parser.add_argument(
        "--submodel-dir",
        type=Path,
        default=None,
        help="Path to submodel targets directory",
    )
    parser.add_argument(
        "--invalidate",
        nargs="+",
        metavar="PARAM",
        help="Invalidate cached components containing these parameters before re-running inference",
    )
    parser.add_argument(
        "--sbi-run-path",
        type=Path,
        default=None,
        help="Pinned SBI run directory with posterior_samples.csv and diagnostics/ (enables Stage 2 section)",
    )
    args = parser.parse_args()

    config = AuditConfig(
        project_root=args.project_root,
        priors_csv=args.priors_csv,
        submodel_dir=args.submodel_dir,
        sbi_run_path=args.sbi_run_path,
    )

    report = run_audit(config, output=args.output, invalidate_params=args.invalidate)

    if not args.output:
        print(report)


if __name__ == "__main__":
    main()
