from __future__ import annotations

from .basic_policies import HeuristicPolicy, Policy, RandomLegalPolicy
from .search_policy import (
    DeterminizedSearchPolicy,
    SearchConfig,
    TDDeterminizedSearchPolicy,
    TDSearchPolicyConfig,
)
from .value_policy import TDValuePolicy, TDValuePolicyConfig


def policy_from_name(
    name: str,
    *,
    search_config: SearchConfig | None = None,
    td_value_config: TDValuePolicyConfig | None = None,
    td_search_config: TDSearchPolicyConfig | None = None,
) -> Policy:
    normalized = name.strip().lower()
    if normalized == "random":
        return RandomLegalPolicy()
    if normalized == "heuristic":
        return HeuristicPolicy()
    if normalized == "search":
        return DeterminizedSearchPolicy(config=search_config or SearchConfig())
    if normalized == "td-value":
        if td_value_config is None:
            raise ValueError("td-value policy requires td_value_config with checkpoint path.")
        return TDValuePolicy(config=td_value_config)
    if normalized == "td-search":
        if td_search_config is None:
            raise ValueError("td-search policy requires td_search_config with value checkpoint.")
        return TDDeterminizedSearchPolicy(config=td_search_config)
    raise ValueError(
        f"Unknown policy name: {name!r}. Expected one of: random, heuristic, search, td-value, td-search."
    )


__all__ = ["policy_from_name"]
