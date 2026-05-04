"""Per-component content fingerprinting for submodel inference caches.

The submodel inference cache (``calibration_targets/submodel_targets/.compare_cache/comp_*.json``)
is keyed by **component identity** alone — the SHA of the sorted parameter
names in the component. That key is stable across input edits: changing a
target YAML, a prior CSV row, or ``submodel_config.yaml`` does not invalidate
the cache. A user has to remember to pass ``--invalidate PARAM`` or
``rm -rf .compare_cache/`` after any such edit.

This module adds a **content fingerprint** alongside each cache entry, so
downstream consumers (and humans) can detect staleness without re-running
inference.

Each fingerprint covers the inputs that materially shape a component's
posterior:

1. Raw bytes of every target YAML in the component.
2. The priors-CSV rows for parameters that appear in the component.
3. The ``submodel_config.yaml`` slice covering any of those parameters
   (groups + cascade-cut entries that touch them).
4. ``reference_values.yaml`` (target ``code:`` blocks reference it).
5. The qsp-inference package version (algorithm changes shift posteriors).

The aggregated ``metadata.freshness`` block in ``submodel_priors.yaml``
records each component's fingerprint so a checker can later re-derive the
fingerprint from the current tree and report which components are stale.

The cache itself is **not** automatically invalidated by a fingerprint
mismatch — invalidation triggers minutes-to-hours of MCMC, so the
intentional design is "warn first, opt-in to delete."
"""

from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------


def _sha256(*chunks: str | bytes) -> str:
    """SHA-256 of concatenated chunks, returned as full hex digest."""
    h = hashlib.sha256()
    for c in chunks:
        if isinstance(c, str):
            c = c.encode()
        h.update(c)
    return h.hexdigest()


def _qsp_inference_version() -> str:
    try:
        return metadata.version("qsp-inference")
    except metadata.PackageNotFoundError:
        return "unknown"


# ---------------------------------------------------------------------------
# Per-input slicers
# ---------------------------------------------------------------------------


def hash_target_yamls(
    target_filenames: list[str],
    submodel_dir: Path,
) -> dict[str, str]:
    """Hash the raw bytes of each target YAML in the component.

    Returns ``{filename: sha256-hex}``. Missing files map to ``"missing"``
    so a deletion still trips the checker.
    """
    submodel_dir = Path(submodel_dir)
    out: dict[str, str] = {}
    for fname in sorted(set(target_filenames)):
        path = submodel_dir / fname
        if not path.exists():
            out[fname] = "missing"
            continue
        out[fname] = _sha256(path.read_bytes())
    return out


def hash_priors_rows(
    priors_csv: Path,
    params: set[str] | list[str],
) -> dict[str, str]:
    """Hash the priors-CSV rows for the parameters in *params*.

    Returns ``{param_name: sha256-hex}``. Parameters absent from the CSV
    map to ``"missing"`` (they're driven entirely by inline / nuisance
    priors and therefore aren't sensitive to CSV edits, but we still
    record their absence so a later CSV addition shows up as a change).
    """
    import pandas as pd

    df = pd.read_csv(priors_csv)
    df = df.set_index("name", drop=False)
    out: dict[str, str] = {}
    for p in sorted(set(params)):
        if p not in df.index:
            out[p] = "missing"
            continue
        row = df.loc[p]
        # Canonicalize: sort columns by name, render as "col=val" pairs
        # so float formatting is reproducible (CSV round-trip is text, so
        # this is robust to read/write cycles).
        canonical = "\n".join(
            f"{col}={row[col]}" for col in sorted(df.columns)
        )
        out[p] = _sha256(canonical)
    return out


def hash_submodel_config_slice(
    config_path: Path,
    params: set[str] | list[str],
) -> str:
    """Hash the ``submodel_config.yaml`` slice that touches *params*.

    Includes:
      * Groups whose ``members`` intersect *params*.
      * Cascade-cut entries naming any param in *params*.
      * Per-target overrides referencing any param in *params*.

    Returns ``"absent"`` if the config file does not exist, ``"empty"``
    if it exists but contains no relevant slice.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        return "absent"
    raw = config_path.read_text()
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        # Conservative: if we can't parse, fingerprint the raw bytes so any
        # edit busts the cache.
        return _sha256(raw)

    params_set = set(params)
    relevant: dict = {}

    # Groups -------------------------------------------------------------
    groups = data.get("groups") or []
    relevant_groups = []
    for g in groups:
        members = {m.get("name") for m in g.get("members", []) if isinstance(m, dict)}
        if members & params_set:
            relevant_groups.append(g)
    if relevant_groups:
        # Sort by group name for determinism.
        relevant_groups.sort(key=lambda g: g.get("name", ""))
        relevant["groups"] = relevant_groups

    # Cascade cuts -------------------------------------------------------
    cuts = data.get("cascade_cuts") or []
    relevant_cuts = [c for c in cuts if c.get("param") in params_set]
    if relevant_cuts:
        relevant_cuts.sort(key=lambda c: c.get("param", ""))
        relevant["cascade_cuts"] = relevant_cuts

    # Per-target overrides ----------------------------------------------
    overrides = data.get("target_overrides") or {}
    relevant_overrides = {}
    for target_id, override in overrides.items():
        if not isinstance(override, dict):
            continue
        ov_params = set(override.get("parameters", {}).keys())
        if ov_params & params_set:
            relevant_overrides[target_id] = override
    if relevant_overrides:
        relevant["target_overrides"] = dict(sorted(relevant_overrides.items()))

    if not relevant:
        return "empty"

    canonical = yaml.safe_dump(relevant, sort_keys=True)
    return _sha256(canonical)


def hash_reference_values(submodel_dir: Path) -> str:
    """Hash ``calibration_targets/reference_values.yaml`` if present.

    The reference-values file lives one directory up from the submodel
    targets in pdac-build's layout (``calibration_targets/reference_values.yaml``
    while submodel YAMLs live in ``calibration_targets/submodel_targets/``).
    Falls back to a sibling path inside *submodel_dir* if the conventional
    location is absent.
    """
    submodel_dir = Path(submodel_dir)
    candidates = [
        submodel_dir.parent / "reference_values.yaml",
        submodel_dir / "reference_values.yaml",
    ]
    for path in candidates:
        if path.exists():
            return _sha256(path.read_bytes())
    return "absent"


# ---------------------------------------------------------------------------
# Component-level fingerprint
# ---------------------------------------------------------------------------


def compute_component_freshness(
    *,
    params: set[str] | list[str],
    target_filenames: list[str],
    submodel_dir: Path,
    priors_csv: Path,
    submodel_config_path: Path | None = None,
) -> dict:
    """Build a freshness manifest for one inference component.

    The returned dict is JSON-serializable and is meant to be embedded
    inside ``comp_*.json`` alongside ``fits/diag/samples``.

    The top-level ``content_hash`` is a SHA-256 of the canonical
    representation of every input listed in ``inputs``; downstream
    checkers compare it against a freshly-computed one to detect drift.
    """
    submodel_dir = Path(submodel_dir)
    priors_csv = Path(priors_csv)
    if submodel_config_path is None:
        candidate = submodel_dir / "submodel_config.yaml"
        submodel_config_path = candidate if candidate.exists() else None
    elif submodel_config_path is not None:
        submodel_config_path = Path(submodel_config_path)

    params = set(params)
    target_yaml_hashes = hash_target_yamls(target_filenames, submodel_dir)
    prior_row_hashes = hash_priors_rows(priors_csv, params)
    config_slice_hash = (
        hash_submodel_config_slice(submodel_config_path, params)
        if submodel_config_path is not None
        else "absent"
    )
    reference_values_hash = hash_reference_values(submodel_dir)
    version = _qsp_inference_version()

    inputs = {
        "qsp_inference_version": version,
        "target_yamls": target_yaml_hashes,
        "prior_rows": prior_row_hashes,
        "submodel_config_slice": config_slice_hash,
        "reference_values": reference_values_hash,
    }

    # Aggregate hash: a canonical YAML dump of inputs (sort_keys=True)
    # gives us a deterministic byte-string regardless of insertion order.
    canonical = yaml.safe_dump(inputs, sort_keys=True)
    content_hash = _sha256(canonical)

    return {
        "content_hash": content_hash,
        "inputs": inputs,
    }


# ---------------------------------------------------------------------------
# Staleness checking
# ---------------------------------------------------------------------------


class StalenessReport:
    """Result of comparing stored freshness against a live recomputation.

    Attributes:
        stale: ``{component_id: list[diff_message]}`` — components whose
            current content_hash differs from the stored one. Empty
            mapping means "all components fresh."
        missing_in_yaml: component IDs present in the live tree but
            absent from the stored manifest.
        missing_on_disk: component IDs present in the stored manifest
            but no longer present on disk.
        version_drift: ``(stored_version, current_version)`` if the
            qsp-inference version differs; ``None`` otherwise.

    A non-empty ``stale``/``missing_*`` set or a non-None
    ``version_drift`` means the cache should be regenerated for the
    affected components.
    """

    def __init__(self) -> None:
        self.stale: dict[str, list[str]] = {}
        self.missing_in_yaml: list[str] = []
        self.missing_on_disk: list[str] = []
        self.version_drift: tuple[str, str] | None = None

    def is_fresh(self) -> bool:
        return (
            not self.stale
            and not self.missing_in_yaml
            and not self.missing_on_disk
            and self.version_drift is None
        )

    def format(self) -> str:
        if self.is_fresh():
            return "submodel_priors.yaml is fresh."
        lines = ["submodel_priors.yaml is STALE:"]
        if self.version_drift:
            lines.append(
                f"  qsp_inference version drift: stored={self.version_drift[0]} "
                f"current={self.version_drift[1]}"
            )
        for cid, diffs in sorted(self.stale.items()):
            lines.append(f"  {cid}:")
            for d in diffs:
                lines.append(f"    - {d}")
        for cid in self.missing_in_yaml:
            lines.append(f"  {cid}: present on disk but missing from yaml header")
        for cid in self.missing_on_disk:
            lines.append(f"  {cid}: present in yaml header but missing on disk")
        return "\n".join(lines)


def _diff_inputs(stored: dict, current: dict) -> list[str]:
    """Human-readable diff between two ``inputs`` dicts."""
    diffs: list[str] = []
    keys = set(stored) | set(current)
    for k in sorted(keys):
        sv = stored.get(k)
        cv = current.get(k)
        if sv == cv:
            continue
        if isinstance(sv, dict) or isinstance(cv, dict):
            sub_keys = set(sv or {}) | set(cv or {})
            for sk in sorted(sub_keys):
                ssv = (sv or {}).get(sk)
                ccv = (cv or {}).get(sk)
                if ssv != ccv:
                    diffs.append(f"{k}.{sk}: stored={ssv!r} current={ccv!r}")
        else:
            diffs.append(f"{k}: stored={sv!r} current={cv!r}")
    return diffs


def compute_live_components_freshness(
    submodel_dir: Path,
    priors_csv: Path,
    glob_pattern: str = "*_PDAC_deriv*.yaml",
) -> dict[str, dict]:
    """Recompute the freshness manifest for every active component.

    Mirrors the discovery logic in ``run_comparison``: lightweight-parse
    the target YAMLs, build connected components (with parameter groups +
    cascade cuts), and return ``{comp_id: freshness_dict}``.

    Returns an empty dict when no targets match.
    """
    # Local imports to avoid a circular import at module load time
    # (comparison.py imports this module).
    from qsp_inference.submodel.comparison import (
        _compute_hash,
        _find_components_lightweight,
        _lightweight_parse,
    )
    from qsp_inference.submodel.parameter_groups import load_parameter_groups

    submodel_dir = Path(submodel_dir)
    priors_csv = Path(priors_csv)
    config_path = submodel_dir / "submodel_config.yaml"
    submodel_config_path = config_path if config_path.exists() else None

    param_groups = (
        load_parameter_groups(config_path) if config_path.exists() else None
    )

    yaml_files = sorted(submodel_dir.glob(glob_pattern))
    yaml_files = [f for f in yaml_files if f.name != "submodel_config.yaml"]
    lightweight_targets: list[dict] = []
    for yf in yaml_files:
        try:
            lt = _lightweight_parse(yf.read_text())
        except Exception:
            continue
        if lt is None:
            continue
        lt["filename"] = yf.name
        lightweight_targets.append(lt)

    cascade_cut_params = frozenset(
        param_groups.cascade_cut_params if param_groups else ()
    )
    components = _find_components_lightweight(
        lightweight_targets, param_groups, cascade_cut_params
    )
    active_components = [c for c in components if c["target_filenames"]]

    out: dict[str, dict] = {}
    for comp in active_components:
        comp_id = "comp_" + _compute_hash("\n".join(sorted(comp["params"])))
        out[comp_id] = compute_component_freshness(
            params=comp["params"],
            target_filenames=comp["target_filenames"],
            submodel_dir=submodel_dir,
            priors_csv=priors_csv,
            submodel_config_path=submodel_config_path,
        )
    return out


def load_stored_freshness(submodel_priors_yaml: Path) -> dict[str, dict]:
    """Read ``metadata.freshness`` out of an existing ``submodel_priors.yaml``.

    Returns an empty dict if the file or the freshness block is absent.
    """
    submodel_priors_yaml = Path(submodel_priors_yaml)
    if not submodel_priors_yaml.exists():
        return {}
    with open(submodel_priors_yaml) as f:
        data = yaml.safe_load(f) or {}
    metadata = data.get("metadata", {}) or {}
    return metadata.get("freshness", {}) or {}


def check_submodel_priors_freshness(
    submodel_priors_yaml: Path,
    submodel_dir: Path,
    priors_csv: Path,
    glob_pattern: str = "*_PDAC_deriv*.yaml",
) -> StalenessReport:
    """Diff a written ``submodel_priors.yaml`` against the live tree.

    Top-level entry point for both CLI checkers (e.g. a pdac-build
    ``scripts/check_submodel_priors_freshness.py``) and in-process
    pre-flight gates (the SBI runner can call this and warn before
    consuming the prior).
    """
    stored = load_stored_freshness(Path(submodel_priors_yaml))
    live = compute_live_components_freshness(
        Path(submodel_dir), Path(priors_csv), glob_pattern=glob_pattern
    )
    return check_freshness_against_components(stored, live)


def check_freshness_against_components(
    stored_freshness: dict[str, dict],
    live_components: dict[str, dict],
) -> StalenessReport:
    """Compare a stored freshness header against live-recomputed manifests.

    Args:
        stored_freshness: ``{component_id: freshness_dict}`` as written
            into ``submodel_priors.yaml``'s ``metadata.freshness``.
        live_components: ``{component_id: freshness_dict}`` recomputed
            from the current tree (one entry per active component).
    """
    report = StalenessReport()

    stored_ids = set(stored_freshness)
    live_ids = set(live_components)

    report.missing_in_yaml = sorted(live_ids - stored_ids)
    report.missing_on_disk = sorted(stored_ids - live_ids)

    # Version drift: pull from any stored entry. They should all agree.
    if stored_freshness:
        any_stored = next(iter(stored_freshness.values()))
        stored_version = any_stored.get("inputs", {}).get(
            "qsp_inference_version", "unknown"
        )
        current_version = _qsp_inference_version()
        if stored_version != current_version:
            report.version_drift = (stored_version, current_version)

    for cid in stored_ids & live_ids:
        stored_hash = stored_freshness[cid].get("content_hash")
        live_hash = live_components[cid].get("content_hash")
        if stored_hash == live_hash:
            continue
        diffs = _diff_inputs(
            stored_freshness[cid].get("inputs", {}),
            live_components[cid].get("inputs", {}),
        )
        report.stale[cid] = diffs or [
            f"content_hash mismatch (stored={stored_hash}, current={live_hash})"
        ]
    return report
