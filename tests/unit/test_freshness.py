"""Tests for per-component content fingerprinting.

The submodel inference cache is keyed by component identity (sorted parameter
names). These tests verify that the new freshness manifest stamped into
each ``comp_*.json`` correctly fingerprints the inputs that materially
shape a component's posterior, so a downstream checker can detect when
``submodel_priors.yaml`` is stale relative to the current tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from qsp_inference.submodel.freshness import (
    check_freshness_against_components,
    compute_component_freshness,
    hash_priors_rows,
    hash_reference_values,
    hash_submodel_config_slice,
    hash_target_yamls,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def submodel_tree(tmp_path: Path) -> dict[str, Path]:
    """Create a minimal calibration_targets/ tree with two components.

    Component A: ``{k_A1, k_A2}`` constrained by ``target_a.yaml``.
    Component B: ``{k_B}`` constrained by ``target_b.yaml``.
    """
    submodel_dir = tmp_path / "submodel_targets"
    submodel_dir.mkdir()

    (submodel_dir / "target_a_PDAC_derived.yaml").write_text(
        """
target_id: target_a
calibration:
  parameters:
    - {name: k_A1}
    - {name: k_A2}
"""
    )
    (submodel_dir / "target_b_PDAC_derived.yaml").write_text(
        """
target_id: target_b
calibration:
  parameters:
    - {name: k_B}
"""
    )
    (submodel_dir / "submodel_config.yaml").write_text(
        """
groups:
  - name: A_group
    members:
      - {name: k_A1}
      - {name: k_A2}
"""
    )

    # reference_values.yaml lives one directory up in pdac-build's layout.
    (tmp_path / "reference_values.yaml").write_text("cell_diameter_um: 10.0\n")

    priors_csv = tmp_path / "priors.csv"
    priors_csv.write_text(
        "name,distribution,units,dist_param1,dist_param2\n"
        "k_A1,lognormal,1/day,0.0,0.5\n"
        "k_A2,lognormal,1/day,0.5,0.5\n"
        "k_B,lognormal,1/day,1.0,0.5\n"
        "k_unrelated,lognormal,1/day,2.0,0.3\n"
    )

    return {
        "submodel_dir": submodel_dir,
        "priors_csv": priors_csv,
        "reference_values": tmp_path / "reference_values.yaml",
        "config": submodel_dir / "submodel_config.yaml",
    }


# ---------------------------------------------------------------------------
# Per-input slicers
# ---------------------------------------------------------------------------


def test_hash_target_yamls_changes_when_bytes_change(submodel_tree):
    submodel_dir = submodel_tree["submodel_dir"]
    h1 = hash_target_yamls(["target_a_PDAC_derived.yaml"], submodel_dir)

    target = submodel_dir / "target_a_PDAC_derived.yaml"
    target.write_text(target.read_text() + "# unrelated comment\n")
    h2 = hash_target_yamls(["target_a_PDAC_derived.yaml"], submodel_dir)

    assert h1["target_a_PDAC_derived.yaml"] != h2["target_a_PDAC_derived.yaml"]


def test_hash_target_yamls_handles_missing(submodel_tree):
    h = hash_target_yamls(["does_not_exist.yaml"], submodel_tree["submodel_dir"])
    assert h["does_not_exist.yaml"] == "missing"


def test_hash_priors_rows_only_includes_requested_params(submodel_tree):
    h = hash_priors_rows(submodel_tree["priors_csv"], {"k_A1", "k_A2"})
    assert set(h.keys()) == {"k_A1", "k_A2"}


def test_hash_priors_rows_changes_when_relevant_row_changes(submodel_tree):
    csv = submodel_tree["priors_csv"]
    h1 = hash_priors_rows(csv, {"k_A1"})

    text = csv.read_text().replace("k_A1,lognormal,1/day,0.0,0.5", "k_A1,lognormal,1/day,0.0,0.7")
    csv.write_text(text)
    h2 = hash_priors_rows(csv, {"k_A1"})

    assert h1["k_A1"] != h2["k_A1"]


def test_hash_priors_rows_unchanged_when_unrelated_row_changes(submodel_tree):
    csv = submodel_tree["priors_csv"]
    h1 = hash_priors_rows(csv, {"k_A1"})

    text = csv.read_text().replace(
        "k_unrelated,lognormal,1/day,2.0,0.3", "k_unrelated,lognormal,1/day,9.0,0.9"
    )
    csv.write_text(text)
    h2 = hash_priors_rows(csv, {"k_A1"})

    assert h1["k_A1"] == h2["k_A1"]


def test_hash_priors_rows_missing_param(submodel_tree):
    h = hash_priors_rows(submodel_tree["priors_csv"], {"k_does_not_exist"})
    assert h["k_does_not_exist"] == "missing"


def test_hash_submodel_config_slice_only_includes_relevant_groups(submodel_tree):
    config = submodel_tree["config"]
    # Add an unrelated group and verify it doesn't influence the slice
    h1 = hash_submodel_config_slice(config, {"k_A1", "k_A2"})

    config.write_text(
        config.read_text()
        + "  - name: B_group\n"
        + "    members:\n"
        + "      - {name: k_B}\n"
    )
    h2 = hash_submodel_config_slice(config, {"k_A1", "k_A2"})

    assert h1 == h2  # the new B_group entry shouldn't bust A's fingerprint


def test_hash_submodel_config_slice_changes_when_relevant_group_changes(submodel_tree):
    config = submodel_tree["config"]
    h1 = hash_submodel_config_slice(config, {"k_A1", "k_A2"})

    config.write_text(
        config.read_text().replace(
            "      - {name: k_A2}", "      - {name: k_A2}\n      - {name: k_A3}"
        )
    )
    h2 = hash_submodel_config_slice(config, {"k_A1", "k_A2"})

    assert h1 != h2


def test_hash_reference_values_finds_parent_dir(submodel_tree):
    h = hash_reference_values(submodel_tree["submodel_dir"])
    assert h != "absent"

    submodel_tree["reference_values"].write_text("cell_diameter_um: 12.0\n")
    h2 = hash_reference_values(submodel_tree["submodel_dir"])
    assert h != h2


# ---------------------------------------------------------------------------
# Component-level fingerprint
# ---------------------------------------------------------------------------


def test_compute_component_freshness_is_deterministic(submodel_tree):
    f1 = compute_component_freshness(
        params={"k_A1", "k_A2"},
        target_filenames=["target_a_PDAC_derived.yaml"],
        submodel_dir=submodel_tree["submodel_dir"],
        priors_csv=submodel_tree["priors_csv"],
    )
    f2 = compute_component_freshness(
        params={"k_A1", "k_A2"},
        target_filenames=["target_a_PDAC_derived.yaml"],
        submodel_dir=submodel_tree["submodel_dir"],
        priors_csv=submodel_tree["priors_csv"],
    )
    assert f1["content_hash"] == f2["content_hash"]


def test_component_freshness_unchanged_when_unrelated_component_edited(submodel_tree):
    """Component A's fingerprint must not move when target_b changes."""
    fa1 = compute_component_freshness(
        params={"k_A1", "k_A2"},
        target_filenames=["target_a_PDAC_derived.yaml"],
        submodel_dir=submodel_tree["submodel_dir"],
        priors_csv=submodel_tree["priors_csv"],
    )

    # Edit the *other* component's target YAML.
    other = submodel_tree["submodel_dir"] / "target_b_PDAC_derived.yaml"
    other.write_text(other.read_text() + "# touched\n")

    # And edit an unrelated CSV row.
    csv = submodel_tree["priors_csv"]
    csv.write_text(
        csv.read_text().replace("k_B,lognormal,1/day,1.0,0.5", "k_B,lognormal,1/day,1.5,0.9")
    )

    fa2 = compute_component_freshness(
        params={"k_A1", "k_A2"},
        target_filenames=["target_a_PDAC_derived.yaml"],
        submodel_dir=submodel_tree["submodel_dir"],
        priors_csv=submodel_tree["priors_csv"],
    )
    assert fa1["content_hash"] == fa2["content_hash"]


def test_component_freshness_changes_when_target_yaml_edited(submodel_tree):
    fa1 = compute_component_freshness(
        params={"k_A1", "k_A2"},
        target_filenames=["target_a_PDAC_derived.yaml"],
        submodel_dir=submodel_tree["submodel_dir"],
        priors_csv=submodel_tree["priors_csv"],
    )

    target = submodel_tree["submodel_dir"] / "target_a_PDAC_derived.yaml"
    target.write_text(target.read_text() + "# data revision\n")

    fa2 = compute_component_freshness(
        params={"k_A1", "k_A2"},
        target_filenames=["target_a_PDAC_derived.yaml"],
        submodel_dir=submodel_tree["submodel_dir"],
        priors_csv=submodel_tree["priors_csv"],
    )
    assert fa1["content_hash"] != fa2["content_hash"]


# ---------------------------------------------------------------------------
# Diff / staleness checker
# ---------------------------------------------------------------------------


def test_staleness_report_clean_when_inputs_unchanged(submodel_tree):
    f = compute_component_freshness(
        params={"k_A1", "k_A2"},
        target_filenames=["target_a_PDAC_derived.yaml"],
        submodel_dir=submodel_tree["submodel_dir"],
        priors_csv=submodel_tree["priors_csv"],
    )
    stored = {"comp_abc": f}
    live = {"comp_abc": dict(f)}  # same content
    report = check_freshness_against_components(stored, live)
    assert report.is_fresh()


def test_staleness_report_flags_input_diffs(submodel_tree):
    fa = compute_component_freshness(
        params={"k_A1", "k_A2"},
        target_filenames=["target_a_PDAC_derived.yaml"],
        submodel_dir=submodel_tree["submodel_dir"],
        priors_csv=submodel_tree["priors_csv"],
    )

    target = submodel_tree["submodel_dir"] / "target_a_PDAC_derived.yaml"
    target.write_text(target.read_text() + "# revised\n")

    fa_live = compute_component_freshness(
        params={"k_A1", "k_A2"},
        target_filenames=["target_a_PDAC_derived.yaml"],
        submodel_dir=submodel_tree["submodel_dir"],
        priors_csv=submodel_tree["priors_csv"],
    )

    report = check_freshness_against_components({"comp_abc": fa}, {"comp_abc": fa_live})
    assert not report.is_fresh()
    assert "comp_abc" in report.stale
    # The diff should mention which target file moved.
    assert any("target_a_PDAC_derived.yaml" in d for d in report.stale["comp_abc"])


def test_staleness_report_detects_missing_components(submodel_tree):
    f = compute_component_freshness(
        params={"k_A1"},
        target_filenames=["target_a_PDAC_derived.yaml"],
        submodel_dir=submodel_tree["submodel_dir"],
        priors_csv=submodel_tree["priors_csv"],
    )
    report = check_freshness_against_components({"comp_old": f}, {"comp_new": f})
    assert "comp_old" in report.missing_on_disk
    assert "comp_new" in report.missing_in_yaml
    assert not report.is_fresh()
