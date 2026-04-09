"""Click CLI for the parameter coverage audit.

Entry point: ``qsp-audit``
"""

from __future__ import annotations

import sys
from pathlib import Path

import click


# ---------------------------------------------------------------------------
# Shared options via a Click group
# ---------------------------------------------------------------------------

class AuditContext:
    """Bag carried through the Click context."""

    def __init__(self, project_root: Path, **overrides):
        self.project_root = project_root
        self.overrides = overrides

    def build_config(self):
        from qsp_inference.audit.report import AuditConfig

        kwargs = {k: v for k, v in self.overrides.items() if v is not None}
        return AuditConfig(project_root=self.project_root, **kwargs)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--project-root", "-r",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default=".",
    show_default=True,
    help="Project root directory.",
)
@click.option(
    "--priors-csv",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    help="Path to priors CSV. [default: <root>/parameters/pdac_priors.csv]",
)
@click.option(
    "--submodel-dir",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default=None,
    help="Submodel targets directory. [default: <root>/calibration_targets/submodel_targets]",
)
@click.version_option(package_name="qsp-inference")
@click.pass_context
def cli(ctx, project_root, priors_csv, submodel_dir):
    """Parameter coverage audit for QSP models.

    Auto-detects available data and reports prior coverage, posterior
    contraction, PRCC-weighted priority, and inference diagnostics.
    """
    ctx.ensure_object(dict)
    ctx.obj = AuditContext(
        project_root=Path(project_root),
        priors_csv=Path(priors_csv) if priors_csv else None,
        submodel_dir=Path(submodel_dir) if submodel_dir else None,
    )


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=False, resolve_path=True),
    default=None,
    help="Output markdown file. Prints to stdout if omitted.",
)
@click.option(
    "--invalidate", "-i",
    multiple=True,
    metavar="PARAM",
    help="Invalidate cached components containing PARAM before re-running. Repeatable.",
)
@click.pass_obj
def report(ctx: AuditContext, output, invalidate):
    """Generate the full audit report.

    Runs component-wise inference (if cache is missing or invalidated),
    then produces a Markdown report with coverage tables, contraction
    summaries, PPC plots, and extraction priority rankings.

    \b
    Examples:
        qsp-audit report
        qsp-audit report -o audit.md
        qsp-audit report -i k_CD8_kill -i k_Treg_act
        qsp-audit -r ../pdac-build report -o notes/calibration/report.md
    """
    from qsp_inference.audit.report import run_audit

    config = ctx.build_config()
    output_path = Path(output) if output else None
    inv = list(invalidate) if invalidate else None

    report_text = run_audit(
        config,
        output=output_path,
        invalidate_params=inv,
    )

    if not output_path:
        click.echo(report_text)


# ---------------------------------------------------------------------------
# dag
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False, resolve_path=True),
    default=None,
    help="Directory for DAG output files. [default: <root>/figures]",
)
@click.pass_obj
def dag(ctx: AuditContext, output_dir):
    """Generate only the inference DAG visualization.

    Produces SVG and PNG files showing staged component inference with
    cascade edges and parameter groups.

    \b
    Examples:
        qsp-audit dag
        qsp-audit dag -o ./figures
    """
    from qsp_inference.audit.plots import plot_inference_dag

    config = ctx.build_config()
    out = Path(output_dir) if output_dir else config.project_root / "figures"

    result = plot_inference_dag(config.submodel_dir, out)
    if result:
        click.echo(f"DAG written to {result}")
    else:
        click.echo("No multi-stage DAG to render (no cascade cuts defined).", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# status  (quick, no inference)
# ---------------------------------------------------------------------------

@cli.command()
@click.pass_obj
def status(ctx: AuditContext):
    """Show a quick coverage summary without running inference.

    Prints parameter counts, target coverage, and PRCC stats from
    existing data files. Does not touch the inference cache.

    \b
    Examples:
        qsp-audit status
        qsp-audit -r ../pdac-build status
    """
    from qsp_inference.audit.report import (
        load_parameter_groups,
        load_prcc,
        load_priors,
        load_submodel_targets,
    )

    config = ctx.build_config()

    priors = load_priors(config.priors_csv)
    targets = load_submodel_targets(config.submodel_dir)
    groups = load_parameter_groups(config.param_groups)
    prcc = load_prcc(config.prcc_csv)

    n_total = len(priors)
    n_covered = sum(1 for p in priors if p in targets)
    n_grouped = sum(1 for p in priors if p in groups)
    pct = 100 * n_covered / n_total if n_total else 0

    click.echo(f"Parameters:  {n_total}")
    click.echo(f"Covered:     {n_covered}/{n_total} ({pct:.0f}%)")
    click.echo(f"Grouped:     {n_grouped}")
    click.echo(f"Targets:     {sum(len(v) for v in targets.values())} across {len(targets)} params")

    if prcc:
        sig = sum(1 for v in prcc.values() if v.get("significant"))
        sig_covered = sum(1 for k, v in prcc.items() if v.get("significant") and k in targets)
        click.echo(f"PRCC sig:    {sig_covered}/{sig} significant params covered")
    else:
        click.echo("PRCC:        not available")


# ---------------------------------------------------------------------------
# invalidate
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("params", nargs=-1, required=True)
@click.pass_obj
def invalidate(ctx: AuditContext, params):
    """Invalidate cached inference results for specific parameters.

    Removes cache entries for components that contain any of the listed
    parameters, so the next ``qsp-audit report`` will re-run inference
    for those components.

    \b
    Examples:
        qsp-audit invalidate k_CD8_kill k_Treg_act
    """
    import json

    config = ctx.build_config()
    cache_dir = config.compare_cache

    if not cache_dir.exists():
        click.echo(f"Cache directory not found: {cache_dir}", err=True)
        sys.exit(1)

    params_set = set(params)
    removed = 0
    for path in sorted(cache_dir.glob("comp_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            cached_params = set(data.get("samples", {}).keys())
            if cached_params & params_set:
                path.unlink()
                click.echo(f"  removed {path.name} ({', '.join(sorted(cached_params & params_set))})")
                removed += 1
        except (json.JSONDecodeError, OSError):
            continue

    if removed:
        click.echo(f"Invalidated {removed} component(s).")
    else:
        click.echo("No cached components matched the given parameters.")


def main():
    cli()


if __name__ == "__main__":
    main()
