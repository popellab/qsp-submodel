"""Refit a subset of submodel targets in isolation and score the result.

Tests whether an edit to a target improves its own fit. Both arms run over an
identical target set with identical code, so the edit is the only difference.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from qsp_inference.submodel.ppc_audit import ComponentPPC, load_components

DEFAULT_GLOB = "*_deriv*.yaml"


@dataclass
class FitScore:
    """Posterior predictive summary over a set of components."""

    coverage: float
    worst_log10: float
    n_observables: int


@dataclass
class EditComparison:
    """One target set fitted with and without an edit."""

    before: FitScore
    after: FitScore
    params: list[str]
    target_set: list[str]
    edited: list[str]

    @property
    def improved(self) -> bool:
        """Coverage rose, or held while the worst miss shrank."""
        if self.after.coverage > self.before.coverage:
            return True
        return (
            self.after.coverage == self.before.coverage
            and self.after.worst_log10 < self.before.worst_log10
        )


def target_index(
    target_dir: str | Path,
    glob_pattern: str = DEFAULT_GLOB,
) -> tuple[dict[str, str], dict[str, set[str]]]:
    """``target_id -> filename`` and ``filename -> declared parameter names``."""
    by_id: dict[str, str] = {}
    params: dict[str, set[str]] = {}
    for path in sorted(Path(target_dir).glob(glob_pattern)):
        data = yaml.safe_load(path.read_text()) or {}
        if data.get("target_id"):
            by_id[data["target_id"]] = path.name
        params[path.name] = {
            p.get("name")
            for p in ((data.get("calibration") or {}).get("parameters") or [])
        }
    return by_id, params


def resolve_target_set(
    target_dir: str | Path,
    filenames: list[str],
    config_path: str | Path | None,
    glob_pattern: str = DEFAULT_GLOB,
) -> list[str]:
    """Expand ``filenames`` to a set the stage DAG will accept.

    ``_build_stage_dag`` walks the whole cascade cut list and raises on any
    upstream target it cannot place, whether or not that cut's parameter is in
    the run, so every cut's upstream has to be present. The closure then repeats
    over the parameters those upstreams declare, since one can trigger another.
    """
    out = set(filenames)
    if config_path is None:
        return sorted(out)
    cuts = (yaml.safe_load(Path(config_path).read_text()) or {}).get("cascade_cuts") or []
    by_id, params = target_index(target_dir, glob_pattern)
    out |= {
        by_id[up]
        for cut in cuts
        for up in (cut.get("upstream") or [])
        if up in by_id
    }
    while True:
        have: set[str] = set()
        for fn in out:
            have |= params.get(fn, set())
        add = {
            by_id[up]
            for cut in cuts
            if cut.get("parameter") in have
            for up in (cut.get("upstream") or [])
            if up in by_id and by_id[up] not in out
        }
        if not add:
            return sorted(out)
        out |= add


def build_isolated_dir(
    dest: str | Path,
    target_dir: str | Path,
    filenames: list[str],
    config_path: str | Path | None = None,
    overrides: dict[str, str | Path] | None = None,
) -> Path:
    """Populate ``dest`` with ``filenames``, the config, and any overrides.

    ``overrides`` maps a filename to a replacement copied in its place.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        shutil.copy2(config_path, dest / Path(config_path).name)
    for fn in filenames:
        shutil.copy2(Path(target_dir) / fn, dest / fn)
    for fn, src in (overrides or {}).items():
        shutil.copy2(src, dest / fn)
    return dest


def fit_isolated(
    target_dir: str | Path,
    priors_csv: str | Path,
    config_name: str | None = None,
    glob_pattern: str = DEFAULT_GLOB,
    **kwargs,
) -> list[ComponentPPC]:
    """Fit every target in ``target_dir`` from a cold cache and read it back."""
    from qsp_inference.submodel.comparison import run_comparison

    target_dir = Path(target_dir)
    run_comparison(
        priors_csv=priors_csv,
        submodel_dir=target_dir,
        glob_pattern=glob_pattern,
        parameter_groups_path=(target_dir / config_name) if config_name else None,
        **kwargs,
    )
    return load_components(target_dir / ".compare_cache", priors_csv)


def score(comps: list[ComponentPPC], params: set[str] | None = None) -> FitScore:
    """Coverage and worst miss over the components carrying ``params``.

    Restricting to ``params`` keeps targets pulled in only to satisfy a cascade
    cut from diluting the score.
    """
    sel = [c for c in comps if params is None or set(c.params) & params]
    obs = [o for c in sel for o in c.observables]
    if not obs:
        return FitScore(coverage=0.0, worst_log10=float("nan"), n_observables=0)
    ratios = [abs(o.log10_ratio) for o in obs]
    return FitScore(
        coverage=sum(o.covered for o in obs) / len(obs),
        worst_log10=float(np.nanmax(ratios)),
        n_observables=len(obs),
    )


def check_targets(
    target_dir: str | Path,
    filenames: list[str],
    priors_csv: str | Path,
    config_path: str | Path | None = None,
    glob_pattern: str = DEFAULT_GLOB,
) -> list[ComponentPPC]:
    """Fit ``filenames`` in isolation and return their per-observable evidence.

    A smoke test for a target nothing has fitted yet: does its own forward model
    reproduce its own data? Components here are isolated, so a target that would
    merge with others in the full corpus is seen alone.
    """
    target_set = resolve_target_set(target_dir, filenames, config_path, glob_pattern)
    config_name = Path(config_path).name if config_path else None
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = build_isolated_dir(tmp, target_dir, target_set, config_path)
        comps = fit_isolated(run_dir, priors_csv, config_name, glob_pattern)
    fitted = {fn for fn in filenames}
    return [c for c in comps if set(c.targets) & fitted]


def compare_edit(
    target_dir: str | Path,
    filenames: list[str],
    edits: dict[str, str | Path],
    priors_csv: str | Path,
    config_path: str | Path | None = None,
    params: set[str] | None = None,
    glob_pattern: str = DEFAULT_GLOB,
) -> EditComparison:
    """Fit ``filenames`` with and without ``edits``, and score both.

    ``edits`` maps a filename in ``filenames`` to the replacement to test.
    """
    target_set = resolve_target_set(target_dir, filenames, config_path, glob_pattern)
    config_name = Path(config_path).name if config_path else None
    with tempfile.TemporaryDirectory() as tmp:
        before_dir = build_isolated_dir(
            Path(tmp) / "before", target_dir, target_set, config_path
        )
        after_dir = build_isolated_dir(
            Path(tmp) / "after", target_dir, target_set, config_path, overrides=edits
        )
        before = score(
            fit_isolated(before_dir, priors_csv, config_name, glob_pattern), params
        )
        after = score(
            fit_isolated(after_dir, priors_csv, config_name, glob_pattern), params
        )
    return EditComparison(
        before=before,
        after=after,
        params=sorted(params or []),
        target_set=target_set,
        edited=sorted(edits),
    )
