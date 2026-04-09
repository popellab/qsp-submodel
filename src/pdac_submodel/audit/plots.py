"""Plotting functions for the parameter coverage audit report.

Extracted from parameter_audit.py to keep the report generator focused on
data loading and markdown generation.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, lognorm


# ---------------------------------------------------------------------------
# Shared grid scaffold
# ---------------------------------------------------------------------------


def _make_grid(
    n_panels: int,
    ncols: int = 6,
    *,
    legend_handles: list[Patch] | None = None,
    cell_size: tuple[float, float] = (3.2, 2.5),
):
    """Create a figure with a grid of axes plus an optional top-row legend.

    Returns (fig, panel_axes) where panel_axes is a flat list of length
    n_panels (legend row excluded).
    """
    has_legend = legend_handles is not None
    nrows = math.ceil(n_panels / ncols) + (1 if has_legend else 0)
    height_ratios = ([0.4] if has_legend else []) + [1] * (nrows - (1 if has_legend else 0))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(cell_size[0] * ncols, cell_size[1] * nrows),
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes = np.atleast_2d(axes)

    if has_legend:
        for c in range(ncols):
            axes[0, c].set_visible(False)
        legend_ax = fig.add_subplot(nrows, 1, 1)
        legend_ax.axis("off")
        legend_ax.legend(
            handles=legend_handles,
            fontsize=11, loc="center", ncol=len(legend_handles), frameon=True,
        )

    data_row_offset = 1 if has_legend else 0
    panel_axes = []
    for idx in range(n_panels):
        row, col = divmod(idx, ncols)
        panel_axes.append(axes[row + data_row_offset, col])

    # Hide unused cells
    total_cells = (nrows - data_row_offset) * ncols
    for idx in range(n_panels, total_cells):
        row, col = divmod(idx, ncols)
        axes[row + data_row_offset, col].set_visible(False)

    return fig, panel_axes


def _save(fig, output_dir: Path, filename: str, title: str) -> Path:
    """Save figure with standard settings."""
    fig.suptitle(title, fontsize=12, y=1.01)
    try:
        fig.tight_layout()
    except Exception:
        pass
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Log-space KDE helpers
# ---------------------------------------------------------------------------


def _log_hist_kde(ax, samples, color, alpha_hist, alpha_kde, label, lw=1.0):
    """Plot histogram + KDE from positive samples in log space."""
    samps = np.asarray(samples)
    samps = samps[np.isfinite(samps) & (samps > 0)]
    if len(samps) < 5:
        return
    lo, hi = samps.min() * 0.5, samps.max() * 2
    bins = np.geomspace(max(lo, 1e-30), hi, 30)
    ax.hist(samps, bins=bins, density=True, alpha=alpha_hist, color=color,
            edgecolor="none", label=label)
    if len(samps) >= 20:
        try:
            kde = gaussian_kde(np.log(samps))
            x = np.geomspace(max(lo, 1e-30), hi, 200)
            y = kde(np.log(x)) / x
            ax.plot(x, y, color=color, lw=lw, alpha=alpha_kde)
        except np.linalg.LinAlgError:
            pass


def _ci_to_lognorm(med, ci):
    """Fit lognormal from median + CI95 (fallback when no samples)."""
    if med <= 0 or ci[1] <= 0:
        return None
    ci_lo = max(ci[0], med * 0.01)
    sigma = (np.log(ci[1]) - np.log(ci_lo)) / (2 * 1.96)
    if sigma <= 0.001:
        sigma = 0.01
    return lognorm(s=sigma, scale=med)


# ---------------------------------------------------------------------------
# Inference DAG
# ---------------------------------------------------------------------------


def plot_inference_dag(
    submodel_dir: Path,
    output_dir: Path,
) -> Path | None:
    """Draw the full component inference DAG.

    Top-to-bottom stage flow. Every component is shown. Each component
    cluster contains parameter nodes (boxes) and target nodes (ellipses)
    with edges showing which targets constrain which parameters.
    Independent single-target components are arranged in a grid sub-cluster.
    """
    import graphviz
    import yaml as _yaml

    from qsp_inference.submodel.comparison import (
        _build_stage_dag,
        _find_components_lightweight,
        _lightweight_parse,
    )
    from qsp_inference.submodel.parameter_groups import load_parameter_groups

    param_groups = load_parameter_groups(submodel_dir / "submodel_config.yaml")
    if not param_groups.cascade_cuts:
        return None

    # Parse targets — _lightweight_parse accepts dicts or raw YAML strings
    lightweight_targets = []
    for yf in sorted(submodel_dir.glob("*_PDAC_deriv*.yaml")):
        try:
            with open(yf) as f:
                data = _yaml.safe_load(f)
            lt = _lightweight_parse(data)
            if lt is not None:
                lt["filename"] = yf.name
                lightweight_targets.append(lt)
        except Exception:
            continue

    cascade_cut_params = frozenset(param_groups.cascade_cut_params)
    components = _find_components_lightweight(
        lightweight_targets, param_groups, cascade_cut_params
    )
    active = [c for c in components if c["target_filenames"]]
    stages, cascade_edges = _build_stage_dag(
        active, param_groups.cascade_cuts, lightweight_targets
    )

    if len(stages) <= 1:
        return None

    # Build lookups
    fname_to_tid = {lt["filename"]: lt["target_id"] for lt in lightweight_targets}
    fname_to_params = {lt["filename"]: lt["qsp_params"] for lt in lightweight_targets}
    fname_to_type = {}
    for lt in lightweight_targets:
        try:
            with open(submodel_dir / lt["filename"]) as f:
                data = _yaml.safe_load(f)
            fname_to_type[lt["filename"]] = (
                data.get("calibration", {}).get("forward_model", {}).get("type", "?")
            )
        except Exception:
            fname_to_type[lt["filename"]] = "?"

    type_short = {"custom_ode": "ODE", "algebraic": "alg", "direct_fit": "fit",
                  "steady_state_ratio": "SS", "power_law": "pow"}

    def _short_tid(tid):
        s = tid.replace("_PDAC_deriv001", "").replace("_PDAC_deriv002", "")
        return s[:28] + ".." if len(s) > 28 else s

    # Param → group
    param_to_group = {}
    if param_groups.groups:
        for group in param_groups.groups:
            for m in group.members:
                param_to_group[m.name] = group.group_id

    def _param_node_id(ci, param):
        return f"p_{ci}_{param}"

    def _target_node_id(ci, tid):
        return f"t_{ci}_{tid}"

    # Only mark cascade params red if they actually have downstream components
    active_cascade_params = {
        p for p, info in cascade_edges.items() if info["downstream_comps"]
    }

    def _param_color(param):
        if param in active_cascade_params:
            return "#ffcdd2", "#c62828"
        if param in param_to_group:
            return "#e8eaf6", "#3949ab"
        return "#ffffff", "#888888"

    # Identify cascade-involved components (multi-target or in cascade/group)
    cascade_involved = set()
    for edge_info in cascade_edges.values():
        cascade_involved.add(edge_info["upstream_comp"])
        cascade_involved.update(edge_info["downstream_comps"])
    if param_groups.groups:
        for group in param_groups.groups:
            member_names = {m.name for m in group.members}
            for ci, comp in enumerate(active):
                if comp["params"] & member_names:
                    cascade_involved.add(ci)

    method_colors = {"NUTS": ("#e8f5e9", "#388e3c"), "NPE": ("#e3f2fd", "#1565c0")}

    def _comp_method(comp):
        has_ode = any(fname_to_type.get(fn, "?") == "custom_ode" for fn in comp["target_filenames"])
        return "NPE" if has_ode else "NUTS"

    # Single graphviz graph — invisible anchors enforce TB stage ordering,
    # cascade edges connect specific param nodes across stages naturally.
    dot = graphviz.Digraph(
        "inference_dag",
        graph_attr={
            "rankdir": "LR", "fontname": "Helvetica", "fontsize": "11",
            "bgcolor": "white", "nodesep": "0.15", "ranksep": "0.6",
            "compound": "true", "newrank": "true",
        },
        node_attr={"fontname": "Helvetica", "fontsize": "7"},
        edge_attr={"fontname": "Helvetica", "fontsize": "6"},
    )

    # ── Legend (as an HTML table node, compact) ──
    dot.node(
        "legend",
        shape="plaintext",
        label=(
            '<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4" '
            'BGCOLOR="white" COLOR="#cccccc">'
            '<TR><TD COLSPAN="2"><B>Legend</B></TD></TR>'
            '<TR><TD BGCOLOR="#ffffff" BORDER="1" STYLE="rounded">&nbsp;</TD>'
            '<TD ALIGN="LEFT"><FONT POINT-SIZE="7">parameter</FONT></TD></TR>'
            '<TR><TD BGCOLOR="#ffcdd2" BORDER="1">&nbsp;</TD>'
            '<TD ALIGN="LEFT"><FONT POINT-SIZE="7">cascade bridge</FONT></TD></TR>'
            '<TR><TD BGCOLOR="#e8eaf6" BORDER="1">&nbsp;</TD>'
            '<TD ALIGN="LEFT"><FONT POINT-SIZE="7">grouped param</FONT></TD></TR>'
            '<TR><TD BGCOLOR="#fff9c4" BORDER="1" STYLE="rounded">&nbsp;</TD>'
            '<TD ALIGN="LEFT"><FONT POINT-SIZE="7">target</FONT></TD></TR>'
            '<TR><TD BGCOLOR="#e8f5e9" BORDER="1">&nbsp;</TD>'
            '<TD ALIGN="LEFT"><FONT POINT-SIZE="7" COLOR="#388e3c">NUTS component</FONT></TD></TR>'
            '<TR><TD BGCOLOR="#e3f2fd" BORDER="1">&nbsp;</TD>'
            '<TD ALIGN="LEFT"><FONT POINT-SIZE="7" COLOR="#1565c0">NPE component</FONT></TD></TR>'
            '<TR><TD COLSPAN="2"><FONT POINT-SIZE="7" COLOR="#d32f2f">'
            '<B>&#x2501;&#x25B6;</B> cascade flow</FONT></TD></TR>'
            '<TR><TD COLSPAN="2"><FONT POINT-SIZE="7" COLOR="#7e57c2">'
            '&#x2504;&#x2504; parameter group</FONT></TD></TR>'
            '<TR><TD COLSPAN="2"><FONT POINT-SIZE="7" COLOR="#bdbdbd">'
            '&#x2500;&#x25B8; target constrains param</FONT></TD></TR>'
            "</TABLE>>"
        ),
    )

    def _render_comp(parent_sg, ci):
        """Render a single component cluster with param + target nodes."""
        comp = active[ci]
        method = _comp_method(comp)
        mbg, mborder = method_colors[method]
        with parent_sg.subgraph(name=f"cluster_comp{ci}") as csg:
            csg.attr(label="", style="rounded,filled", fillcolor=mbg, color=mborder)
            for param in sorted(comp["params"]):
                bg, border = _param_color(param)
                csg.node(
                    _param_node_id(ci, param), label=param, shape="box",
                    style="rounded,filled", fillcolor=bg, color=border,
                    fontsize="7", fontcolor="#222222",
                    width="0", height="0", margin="0.05,0.02",
                )
            for fn in comp["target_filenames"]:
                tid = fname_to_tid.get(fn, fn)
                fm = type_short.get(fname_to_type.get(fn, "?"), "?")
                csg.node(
                    _target_node_id(ci, tid),
                    label=f"{_short_tid(tid)}\n[{fm}]",
                    shape="ellipse", style="filled",
                    fillcolor="#fff9c4", color="#f9a825",
                    fontsize="5", fontcolor="#555555",
                    width="0", height="0", margin="0.03,0.02",
                )
            for fn in comp["target_filenames"]:
                tid = fname_to_tid.get(fn, fn)
                for param in fname_to_params.get(fn, set()):
                    if param in comp["params"]:
                        csg.edge(
                            _target_node_id(ci, tid), _param_node_id(ci, param),
                            color="#bdbdbd", style="solid",
                            penwidth="0.5", arrowsize="0.3",
                        )

    # ── Render all components by stage ──
    # Sort so cascade-involved components render last within each stage.
    # In LR layout, later-declared nodes go lower — placing cascade
    # components near the boundary with the next stage, shortening the
    # cross-stage cascade edges.
    for sx, stage_cis in enumerate(stages):
        independent = [ci for ci in stage_cis if ci not in cascade_involved]
        involved = [ci for ci in stage_cis if ci in cascade_involved]
        ordered_cis = independent + involved

        with dot.subgraph(name=f"cluster_stage{sx}") as stage_sg:
            stage_sg.attr(
                label=f"Stage {sx}",
                style="dashed,rounded",
                color="#999999",
                fontcolor="#555555",
                fontsize="9",
                fontname="Helvetica Bold",
                labeljust="l",
            )
            # Invisible anchor for stage ordering
            stage_sg.node(f"_anchor_{sx}", label="", shape="point",
                          width="0", height="0", style="invis")

            for ci in ordered_cis:
                _render_comp(stage_sg, ci)

            # Invisible chain from last independent → first cascade-involved
            # to keep them vertically adjacent (minimizes cascade edge length)
            if independent and involved:
                last_indep = independent[-1]
                first_involved = involved[0]
                # Pick representative nodes from each
                last_p = sorted(active[last_indep]["params"])[0]
                first_p = sorted(active[first_involved]["params"])[0]
                stage_sg.edge(
                    _param_node_id(last_indep, last_p),
                    _param_node_id(first_involved, first_p),
                    style="invis", weight="10",
                )

    # ── Invisible anchor chain to force stage ordering ──
    for sx in range(len(stages) - 1):
        dot.edge(f"_anchor_{sx}", f"_anchor_{sx + 1}",
                 style="invis", weight="100")

    # ── Invisible alignment edges to pull cascade nodes toward stage boundaries ──
    # For each cascade edge, add a high-weight invisible edge between the
    # upstream param node and the downstream param node. dot uses edge weight
    # to minimize vertical distance, pulling connected nodes together.
    for param_name, edge_info in cascade_edges.items():
        uci = edge_info["upstream_comp"]
        for dci in edge_info["downstream_comps"]:
            dot.edge(
                _param_node_id(uci, param_name),
                _param_node_id(dci, param_name),
                style="invis", weight="50",
            )

    # ── Load inference results for cascade edge annotations ──
    compare_results_path = submodel_dir / "compare_inference_results.yaml"
    cascade_fit_labels = {}
    if compare_results_path.exists():
        with open(compare_results_path) as f:
            cr = _yaml.safe_load(f)
        for param_name in cascade_edges:
            pdata = cr.get("parameters", {}).get(param_name, {})
            joint = pdata.get("joint", {})
            dist = joint.get("distribution", "?")
            median = joint.get("median")
            sigma = joint.get("sigma")
            if median is not None and sigma is not None:
                cascade_fit_labels[param_name] = (
                    f"{dist}(med={median:.3g}, σ={sigma:.2g})"
                )

    # ── Cascade edges (red, bold) — skip if no downstream ──
    for param_name, edge_info in cascade_edges.items():
        uci = edge_info["upstream_comp"]
        for dci in edge_info["downstream_comps"]:
            fit_str = cascade_fit_labels.get(param_name, "")
            label = f" {param_name} "
            if fit_str:
                label += f"\n{fit_str} "
            dot.edge(
                _param_node_id(uci, param_name),
                _param_node_id(dci, param_name),
                label=label,
                color="#d32f2f", fontcolor="#d32f2f",
                style="bold", penwidth="2.0",
            )

    # ── Parameter group edges (purple, dotted) ──
    # Build group_id → between_member_sd for labels
    group_tau = {}
    if param_groups.groups:
        for group in param_groups.groups:
            group_tau[group.group_id] = group.between_member_sd.sigma

    seen_group_edges = set()
    if param_groups.groups:
        for group in param_groups.groups:
            members = sorted(m.name for m in group.members)
            tau = group_tau.get(group.group_id)
            tau_str = f" (τ={tau:.2g})" if tau is not None else ""
            member_locs = []
            for ci, comp in enumerate(active):
                for m in members:
                    if m in comp["params"]:
                        member_locs.append((ci, m))
            for i in range(len(member_locs)):
                for j in range(i + 1, len(member_locs)):
                    ci_a, m_a = member_locs[i]
                    ci_b, m_b = member_locs[j]
                    key = tuple(sorted([(ci_a, m_a), (ci_b, m_b)]))
                    if key in seen_group_edges:
                        continue
                    seen_group_edges.add(key)
                    edge_label = f" {group.group_id}{tau_str} " if i == 0 and j == 1 else ""
                    dot.edge(
                        _param_node_id(ci_a, m_a), _param_node_id(ci_b, m_b),
                        label=edge_label,
                        color="#7e57c2", fontcolor="#7e57c2",
                        style="dotted", dir="none", penwidth="1.0", fontsize="6",
                    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "inference_dag"
    dot.render(out_path, format="svg", cleanup=True)
    dot.render(out_path, format="png", cleanup=True)
    return Path(f"{out_path}.svg")


# ---------------------------------------------------------------------------
# Prior vs posterior marginals
# ---------------------------------------------------------------------------


def plot_marginals(
    priors: dict,
    compare_results: dict,
    joint_samples: dict | None,
    output_dir: Path,
) -> Path | None:
    """Plot prior vs posterior marginal densities for all constrained parameters."""
    params_data = compare_results.get("parameters", {})
    plot_params = sorted(
        name for name, pdata in params_data.items()
        if pdata.get("joint") and name in priors
    )
    if not plot_params:
        return None

    fig, panel_axes = _make_grid(
        len(plot_params),
        legend_handles=[
            Patch(facecolor="#cc4444", alpha=0.25, label="Prior"),
            Patch(facecolor="#2266aa", alpha=0.25, label="Posterior (samples + KDE)"),
        ],
    )

    for ax, pname in zip(panel_axes, plot_params):
        try:
            _plot_one_marginal(ax, pname, priors, params_data, joint_samples)
        except Exception:
            ax.clear()
            ax.text(0.5, 0.5, f"{pname}\n(plot failed)", ha="center", va="center",
                    fontsize=8, transform=ax.transAxes)
            ax.set_yticks([])
            ax.set_xticks([])
        ax.tick_params(axis="x", labelsize=7)

    return _save(fig, output_dir, "prior_vs_posterior_marginals.png",
                 "Prior vs Posterior Marginals")


def _plot_one_marginal(ax, pname, priors, params_data, joint_samples):
    """Plot a single prior vs posterior marginal."""
    prior_info = priors[pname]
    joint_info = params_data[pname]["joint"]

    prior_mu = prior_info["mu"]
    prior_sigma = prior_info["sigma"]
    post_sigma = joint_info.get("sigma") or prior_sigma
    post_median = joint_info.get("median")
    post_mu = np.log(post_median) if (post_median and post_median > 0) else prior_mu

    # x-range: center on posterior, wide enough to show both
    span_sigma = max(prior_sigma, post_sigma, 0.3)
    x_lo = max(np.exp(post_mu - 4.0 * span_sigma), 1e-30)
    x_hi = np.exp(post_mu + 4.0 * span_sigma)
    if x_hi <= x_lo:
        x_lo = np.exp(prior_mu - 3 * prior_sigma)
        x_hi = np.exp(prior_mu + 3 * prior_sigma)
    x = np.geomspace(x_lo, x_hi, 500)

    # Prior density
    prior_dist = lognorm(s=prior_sigma, scale=np.exp(prior_mu))
    ax.plot(x, prior_dist.pdf(x), color="#cc4444", lw=1.5)
    ax.fill_between(x, prior_dist.pdf(x), alpha=0.15, color="#cc4444")

    # Posterior: histogram + KDE from raw samples, or parametric fallback
    plotted = False
    if joint_samples and pname in joint_samples:
        samps = np.asarray(joint_samples[pname])
        samps = samps[np.isfinite(samps) & (samps > 0)]
        if len(samps) > 20:
            bins = np.geomspace(max(samps.min(), x_lo), min(samps.max(), x_hi), 40)
            ax.hist(samps, bins=bins, density=True, alpha=0.25, color="#2266aa",
                    edgecolor="none")
            plotted = True
        if len(samps) > 50:
            try:
                kde = gaussian_kde(np.log(samps))
                ax.plot(x, kde(np.log(x)) / x, color="#2266aa", lw=1.5)
                plotted = True
            except np.linalg.LinAlgError:
                pass

    if not plotted and joint_info.get("sigma") and joint_info.get("median"):
        ps = joint_info["sigma"]
        pm = np.log(joint_info["median"])
        if ps > 0 and np.isfinite(pm):
            post_dist = lognorm(s=ps, scale=np.exp(pm))
            ax.plot(x, post_dist.pdf(x), color="#2266aa", lw=1.5, ls="--")
            ax.fill_between(x, post_dist.pdf(x), alpha=0.10, color="#2266aa")

    ax.set_title(pname, fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Posterior predictive check histograms
# ---------------------------------------------------------------------------


def plot_ppc_histograms(ppc_obs_list: list, output_dir: Path) -> Path | None:
    """Plot PPC: one panel per observable with prior, observed, posterior."""
    valid_obs = [o for o in ppc_obs_list if o.get("post_ci95")]
    if not valid_obs:
        return None

    fig, panel_axes = _make_grid(
        len(valid_obs),
        cell_size=(2.5, 2.5),
        legend_handles=[
            Patch(facecolor="#cc4444", alpha=0.15, label="Prior pred."),
            Patch(facecolor="#22aa44", alpha=0.25, label="Observed"),
            Patch(facecolor="#2266aa", alpha=0.2, label="Posterior pred."),
        ],
    )

    for ax, obs in zip(panel_axes, valid_obs):
        try:
            _plot_one_ppc(ax, obs)
        except Exception:
            ax.set_title("error", fontsize=10)

    total_cov = sum(1 for o in valid_obs if o.get("covered"))
    n = len(valid_obs)
    return _save(fig, output_dir, "ppc_coverage.png",
                 f"Posterior Predictive Check ({total_cov}/{n} covered)")


def _plot_one_ppc(ax, obs):
    """Plot a single PPC panel."""
    observed = obs["observed"]
    has_prior_samples = obs.get("prior_samples") and len(obs["prior_samples"]) >= 5
    has_post_samples = obs.get("post_samples") and len(obs["post_samples"]) >= 5
    has_obs_samples = obs.get("obs_samples") and len(obs["obs_samples"]) >= 5

    # Prior predictive (red)
    if has_prior_samples:
        _log_hist_kde(ax, obs["prior_samples"], "#cc4444", 0.12, 0.7, "Prior pred.")
    elif obs.get("prior_ci95"):
        prior_med = obs.get("prior_median", (obs["prior_ci95"][0] + obs["prior_ci95"][1]) / 2)
        _plot_lognorm_fill(ax, prior_med, obs["prior_ci95"], obs, "#cc4444", 0.12, 0.7)

    # Observed (green)
    if has_obs_samples:
        _log_hist_kde(ax, obs["obs_samples"], "#22aa44", 0.2, 0.9, "Observed")
    elif obs.get("obs_ci95"):
        _plot_lognorm_fill(ax, observed, obs["obs_ci95"], obs, "#22aa44", 0.2, 1.0)
    ax.axvline(observed, color="#22aa44", lw=0.8, ls="--", alpha=0.7)

    # Posterior predictive (blue)
    if has_post_samples:
        _log_hist_kde(ax, obs["post_samples"], "#2266aa", 0.18, 0.9, "Posterior pred.")
    else:
        post_ci = obs["post_ci95"]
        post_med = obs.get("post_median", (post_ci[0] + post_ci[1]) / 2)
        _plot_lognorm_fill(ax, post_med, post_ci, obs, "#2266aa", 0.18, 1.0)

    ax.set_xscale("log")
    ax.set_yticks([])

    name = obs.get("name", "?")
    short = name.split("__")[-1] if "__" in name else name[-20:]
    covered = obs.get("covered", False)
    color = "#222222" if covered else "#cc4444"
    ax.set_title(short, fontsize=10, color=color, fontweight="bold")
    ax.tick_params(axis="x", labelsize=5)


def _plot_lognorm_fill(ax, median, ci, obs, color, alpha_fill, alpha_line):
    """Plot analytical lognormal fill from median + CI95."""
    d = _ci_to_lognorm(median, ci)
    if not d:
        return
    all_bounds = [
        b for b in [*ci, *obs.get("post_ci95", []), obs.get("observed", 0)]
        if b and b > 0
    ]
    if not all_bounds:
        return
    x = np.geomspace(min(all_bounds) * 0.3, max(all_bounds) * 3, 200)
    ax.fill_between(x, d.pdf(x), alpha=alpha_fill, color=color)
    ax.plot(x, d.pdf(x), color=color, lw=0.8, alpha=alpha_line)
