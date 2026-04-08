"""
Hierarchical parameter groups for partial pooling in Bayesian inference.

Declares groups of related parameters (e.g., CAF subtype death rates) that
share a common biological origin. Instead of independent priors from the CSV,
grouped parameters are sampled hierarchically:

    log(k_base) ~ Normal(mu_prior, sigma_prior)   # group mean
    tau ~ HalfNormal(tau_prior)                     # between-member SD
    delta_i ~ Normal(mu_i, sigma_i)                 # per-member deviation
    log(k_i) = log(k_base) + delta_i               # final parameter value

Members with submodel target data get pulled toward their observations.
Members without data shrink toward the group mean (partial pooling).
Optional delta priors encode biological ordering without hard constraints.

Usage::

    from qsp_inference.submodel.parameter_groups import load_parameter_groups

    groups = load_parameter_groups(Path("parameter_groups.yaml"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# =============================================================================
# Schema
# =============================================================================


class GroupPrior(BaseModel):
    """Prior specification for the group base rate or between-member SD."""

    model_config = ConfigDict(extra="forbid")

    distribution: str = Field(
        description="Distribution family: 'lognormal', 'normal', 'half_normal'"
    )
    mu: Optional[float] = Field(
        default=None,
        description="Location parameter (log-space mean for lognormal, mean for normal)",
    )
    sigma: float = Field(description="Scale parameter (log-space SD or SD)")


class DeltaPrior(BaseModel):
    """Prior on a member's deviation from the group mean (log-scale).

    Allows encoding biological ordering: e.g., delta_prior with mu=-0.5
    biases a member toward lower values than the group mean.
    """

    model_config = ConfigDict(extra="forbid")

    distribution: str = Field(
        default="normal",
        description="Distribution for delta: 'normal' (default)",
    )
    mu: float = Field(
        default=0.0,
        description="Mean of delta prior (0 = centered on group mean)",
    )
    sigma: float = Field(
        default=0.3,
        description="SD of delta prior (controls deviation strength)",
    )


class GroupMember(BaseModel):
    """A parameter that belongs to a hierarchical group."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Parameter name (must match pdac_priors.csv)")
    units: str = Field(description="Parameter units (for validation)")
    delta_prior: Optional[DeltaPrior] = Field(
        default=None,
        description="Optional informative prior on this member's deviation from group mean. "
        "If None, uses the shared tau: delta ~ Normal(0, tau).",
    )


class ParameterGroup(BaseModel):
    """A hierarchical group of related parameters.

    Parameters in a group share a latent base rate. Each member's value is:
        log(k_i) = log(k_base) + delta_i

    where k_base is sampled from base_prior and delta_i captures per-member
    deviations. Members with submodel target data get pulled by their
    likelihoods; members without data shrink toward k_base.
    """

    model_config = ConfigDict(extra="forbid")

    group_id: str = Field(description="Unique identifier for this group")
    description: str = Field(
        default="", description="Human-readable description of why these parameters are grouped"
    )

    base_prior: Optional[GroupPrior] = Field(
        default=None,
        description="Prior on the shared group base rate (log-space). "
        "If omitted, derived at runtime from the CSV priors of the group members "
        "(mean of log-space mu, max of log-space sigma). This is the recommended "
        "approach to avoid duplicating values from the CSV.",
    )
    between_member_sd: GroupPrior = Field(
        description="Prior on tau, the between-member standard deviation (log-scale). "
        "Controls how far members can deviate from the group mean. "
        "Typically half_normal with sigma ~0.3-0.5."
    )

    members: list[GroupMember] = Field(
        min_length=2,
        description="Parameters in this group (minimum 2).",
    )

    @model_validator(mode="after")
    def _validate_base_prior_distribution(self):
        if self.base_prior is not None:
            allowed = {"lognormal", "normal"}
            if self.base_prior.distribution not in allowed:
                raise ValueError(
                    f"base_prior.distribution must be one of {allowed}, "
                    f"got '{self.base_prior.distribution}'"
                )
        return self

    @model_validator(mode="after")
    def _validate_tau_distribution(self):
        allowed = {"half_normal"}
        if self.between_member_sd.distribution not in allowed:
            raise ValueError(
                f"between_member_sd.distribution must be one of {allowed}, "
                f"got '{self.between_member_sd.distribution}'"
            )
        return self

    @model_validator(mode="after")
    def _validate_unique_members(self):
        names = [m.name for m in self.members]
        dupes = [n for n in names if names.count(n) > 1]
        if dupes:
            raise ValueError(f"Duplicate member names in group '{self.group_id}': {set(dupes)}")
        return self

    @property
    def member_names(self) -> set[str]:
        """Set of parameter names in this group."""
        return {m.name for m in self.members}

    def resolve_base_prior(self, prior_specs: dict) -> GroupPrior:
        """Return the base_prior, deriving from CSV priors if not explicitly set.

        When base_prior is None, computes the group base prior from the CSV
        priors of the member parameters:
          - mu = mean of members' log-space mu values
          - sigma = max of members' log-space sigma values
          - distribution = "lognormal" (all members must be lognormal)

        Args:
            prior_specs: dict of {param_name: PriorSpec} from CSV

        Returns:
            GroupPrior (either the explicit one or the derived one)

        Raises:
            ValueError: if base_prior is None and members have incompatible priors
        """
        if self.base_prior is not None:
            return self.base_prior

        mus = []
        sigmas = []
        for member in self.members:
            spec = prior_specs.get(member.name)
            if spec is None:
                raise ValueError(
                    f"Cannot derive base_prior for group '{self.group_id}': "
                    f"member '{member.name}' not found in priors CSV"
                )
            if spec.distribution != "lognormal":
                raise ValueError(
                    f"Cannot derive base_prior for group '{self.group_id}': "
                    f"member '{member.name}' has distribution '{spec.distribution}', "
                    f"expected 'lognormal'"
                )
            mus.append(spec.mu)
            sigmas.append(spec.sigma)

        return GroupPrior(
            distribution="lognormal",
            mu=sum(mus) / len(mus),
            sigma=max(sigmas),
        )


class CascadeCut(BaseModel):
    """A parameter bridge to sever between components.

    Instead of merging components that share this parameter during BFS,
    the upstream component's posterior becomes the downstream component's
    prior for this parameter. Enables staged inference where NUTS-compatible
    targets run first and feed into ODE components via NPE.

    The cascade parameter still belongs to both components' parameter sets —
    it just doesn't cause BFS to merge them. After the upstream component
    runs, its posterior for this parameter is fitted as a lognormal and
    injected as the downstream component's prior (replacing the CSV prior).
    """

    model_config = ConfigDict(extra="forbid")

    parameter: str = Field(description="QSP parameter name that bridges two components")
    upstream: list[str] = Field(
        min_length=1,
        description="Target IDs whose component is authoritative for this parameter. "
        "Their posterior becomes the downstream prior.",
    )
    reason: str = Field(
        default="",
        description="Why this cut was made (documentation only)",
    )


class ParameterGroupsConfig(BaseModel):
    """Top-level config for parameter_groups.yaml."""

    model_config = ConfigDict(extra="forbid")

    groups: list[ParameterGroup] = Field(
        default_factory=list,
        description="List of hierarchical parameter groups",
    )
    cascade_cuts: list[CascadeCut] = Field(
        default_factory=list,
        description="Parameter bridges to sever between components. "
        "Each cut splits a mega-component and propagates the upstream "
        "posterior as the downstream prior for that parameter.",
    )

    @model_validator(mode="after")
    def _validate_no_overlapping_members(self):
        """Ensure no parameter appears in multiple groups."""
        seen: dict[str, str] = {}
        for group in self.groups:
            for member in group.members:
                if member.name in seen:
                    raise ValueError(
                        f"Parameter '{member.name}' appears in both "
                        f"'{seen[member.name]}' and '{group.group_id}'"
                    )
                seen[member.name] = group.group_id
        return self

    @model_validator(mode="after")
    def _validate_no_duplicate_cascade_params(self):
        """Ensure no parameter appears in multiple cascade cuts."""
        seen: set[str] = set()
        for cut in self.cascade_cuts:
            if cut.parameter in seen:
                raise ValueError(f"Parameter '{cut.parameter}' appears in multiple cascade_cuts")
            seen.add(cut.parameter)
        return self

    @model_validator(mode="after")
    def _validate_no_cascade_group_overlap(self):
        """Cascade cut parameters must not also be group members.

        Hierarchical sampling and cascade prior injection conflict — a
        parameter can't simultaneously be pooled with group siblings and
        receive an injected posterior from an upstream stage.
        """
        grouped = self.all_grouped_params
        for cut in self.cascade_cuts:
            if cut.parameter in grouped:
                group = self.get_group_for_param(cut.parameter)
                raise ValueError(
                    f"Parameter '{cut.parameter}' is both a cascade_cut and a member "
                    f"of group '{group.group_id}'. Remove it from one or the other."
                )
        return self

    @property
    def all_grouped_params(self) -> set[str]:
        """Set of all parameter names across all groups."""
        return {m.name for g in self.groups for m in g.members}

    @property
    def cascade_cut_params(self) -> set[str]:
        """Set of parameter names that are cascade cuts."""
        return {c.parameter for c in self.cascade_cuts}

    def get_group_for_param(self, param_name: str) -> Optional[ParameterGroup]:
        """Return the group containing this parameter, or None."""
        for group in self.groups:
            if param_name in group.member_names:
                return group
        return None

    def get_upstream_targets(self, param_name: str) -> list[str]:
        """Return upstream target IDs for a cascade cut parameter, or []."""
        for cut in self.cascade_cuts:
            if cut.parameter == param_name:
                return cut.upstream
        return []


# =============================================================================
# Loader
# =============================================================================


def load_parameter_groups(yaml_path: Path) -> ParameterGroupsConfig:
    """Load and validate parameter_groups.yaml.

    Args:
        yaml_path: Path to parameter_groups.yaml

    Returns:
        Validated ParameterGroupsConfig

    Raises:
        FileNotFoundError: if yaml_path does not exist
        ValidationError: if schema validation fails
    """
    import yaml

    if not yaml_path.exists():
        return ParameterGroupsConfig(groups=[])

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if data is None:
        return ParameterGroupsConfig(groups=[])

    return ParameterGroupsConfig(**data)
