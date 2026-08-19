"""Per-observable fit evidence from a submodel compare cache.

Reports what the fit did, without judging it: the datum, what the CSV prior
produces through the forward model, and what the posterior produces.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ObservablePPC:
    """One observable's datum, prior predictive and posterior predictive."""

    name: str
    observed: float
    obs_ci95: tuple[float, float]
    prior_median: float
    prior_lo: float
    prior_hi: float
    prior_log_width: float
    post_median: float
    post_ci95: tuple[float, float]
    covered: bool

    @property
    def log10_ratio(self) -> float:
        """log10(posterior predictive median / observed)."""
        if self.post_median <= 0 or self.observed <= 0:
            return float("nan")
        return math.log10(self.post_median / self.observed)

    @property
    def prior_z(self) -> float:
        """Datum's distance from the prior predictive centre, in its log-sds."""
        sd = self.prior_log_width / 6.0  # min/max of ~200 draws spans about 6 sd
        if sd <= 0 or self.observed <= 0 or self.prior_median <= 0:
            return float("nan")
        return math.log10(self.observed / self.prior_median) / sd


@dataclass
class ComponentPPC:
    """A jointly-fitted set of parameters and the observables it was fit to."""

    component: str
    params: list[str]
    targets: list[str]
    observables: list[ObservablePPC] = field(default_factory=list)
    prior_log_width: dict[str, float] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return sum(o.covered for o in self.observables) / len(self.observables)

    @property
    def worst(self) -> ObservablePPC:
        return max(self.observables, key=lambda o: abs(o.log10_ratio))

    def sensitivity(self, obs: ObservablePPC) -> float:
        """Decades the observable moves per decade of the widest prior on it."""
        widths = [w for w in self.prior_log_width.values() if np.isfinite(w) and w > 0]
        if not widths:
            return float("nan")
        return obs.prior_log_width / max(widths)


def _observable(entry: dict) -> ObservablePPC | None:
    observed = entry.get("observed")
    prior = np.asarray(entry.get("prior_samples") or [], float)
    if observed is None or not np.isfinite(observed) or prior.size == 0:
        return None
    prior = prior[np.isfinite(prior) & (prior > 0)]
    post_median = entry.get("post_median")
    if prior.size < 10 or post_median is None:
        return None

    obs = np.asarray(entry.get("obs_samples") or [], float)
    obs = obs[np.isfinite(obs)]
    if obs.size:
        olo, ohi = (float(v) for v in np.percentile(obs, [2.5, 97.5]))
    else:
        olo = ohi = float(observed)

    lo, hi = float(prior.min()), float(prior.max())
    post_ci = entry.get("post_ci95") or [float("nan"), float("nan")]
    return ObservablePPC(
        name=entry.get("name") or "",
        observed=float(observed),
        obs_ci95=(olo, ohi),
        prior_median=float(np.median(prior)),
        prior_lo=lo,
        prior_hi=hi,
        prior_log_width=math.log10(hi / lo),
        post_median=float(post_median),
        post_ci95=(float(post_ci[0]), float(post_ci[1])),
        covered=bool(entry.get("covered")),
    )


def load_prior_log_widths(priors_csv: str | Path) -> dict[str, float]:
    """Per-parameter CSV prior width in decades."""
    widths = {}
    with open(priors_csv) as fh:
        for row in csv.DictReader(fh):
            try:
                widths[row["name"]] = float(row["dist_param2"]) / math.log(10)
            except (TypeError, ValueError, KeyError):
                continue
    return widths


def load_components(
    cache_dir: str | Path,
    priors_csv: str | Path | None = None,
) -> list[ComponentPPC]:
    """Read every ``comp_*.json`` in a compare cache."""
    widths = load_prior_log_widths(priors_csv) if priors_csv else {}
    comps = []
    for path in sorted(Path(cache_dir).glob("comp_*.json")):
        data = json.loads(path.read_text())
        entries = ((data.get("diag") or {}).get("ppc_observables")) or []
        if not entries:
            continue
        params = sorted((data.get("fits") or {}).keys())
        comp = ComponentPPC(
            component=path.stem.replace("comp_", ""),
            params=params,
            targets=sorted(((data.get("freshness") or {}).get("inputs") or {})
                           .get("target_yamls", {}).keys()),
            prior_log_width={p: widths[p] for p in params if p in widths},
        )
        comp.observables = [o for o in map(_observable, entries) if o is not None]
        if comp.observables:
            comps.append(comp)
    return comps


def format_component(comp: ComponentPPC) -> str:
    """Per-observable evidence table for one component."""
    lines = [
        f"component {comp.component}",
        f"  jointly fitted parameters: {', '.join(comp.params)}",
        f"  from targets:              {', '.join(comp.targets)}",
        f"  posterior predictive coverage: {comp.coverage:.2f} "
        f"({sum(o.covered for o in comp.observables)}/{len(comp.observables)})",
        "",
        "  CSV prior width, decades:",
    ]
    for p in comp.params:
        w = comp.prior_log_width.get(p)
        lines.append(f"    {p:<28} {w:.3f}" if w is not None else f"    {p:<28} n/a")
    lines += [
        "",
        f"  {'observable':<42} {'observed':>11} {'obs_lo':>10} {'obs_hi':>10} "
        f"{'prior_lo':>11} {'prior_med':>11} {'prior_hi':>11} "
        f"{'post_med':>11} {'post_lo':>11} {'post_hi':>11} "
        f"{'log10':>7} {'sens':>6} {'z':>7} cov",
    ]
    for o in sorted(comp.observables, key=lambda o: -abs(o.log10_ratio)):
        lines.append(
            f"  {o.name[:42]:<42} {o.observed:>11.4g} "
            f"{o.obs_ci95[0]:>10.4g} {o.obs_ci95[1]:>10.4g} "
            f"{o.prior_lo:>11.4g} {o.prior_median:>11.4g} {o.prior_hi:>11.4g} "
            f"{o.post_median:>11.4g} {o.post_ci95[0]:>11.4g} {o.post_ci95[1]:>11.4g} "
            f"{o.log10_ratio:>+7.2f} {comp.sensitivity(o):>6.3f} "
            f"{o.prior_z:>+7.1f} {'y' if o.covered else 'n'}"
        )
    lines += [
        "",
        "  observed / obs_lo / obs_hi   the datum and its own 95% interval.",
        "  prior_*                      the observable over 200 draws from the CSV",
        "                               prior, pushed through the forward model.",
        "  post_*                       the same over the posterior.",
        "  log10                        log10(post_med / observed).",
        "  sens                         prior predictive width divided by the widest",
        "                               CSV prior width, both in decades. Near zero",
        "                               means the fitted parameters barely move this",
        "                               observable.",
        "  z                            log10(observed / prior_med) in prior",
        "                               predictive log-sds.",
        "  cov                          datum inside the posterior predictive 95%.",
    ]
    return "\n".join(lines)


def rank_by_miss(comps: list[ComponentPPC]) -> list[ComponentPPC]:
    """Worst posterior predictive miss first."""
    return sorted(comps, key=lambda c: -abs(c.worst.log10_ratio))
