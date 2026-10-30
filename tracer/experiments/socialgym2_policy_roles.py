"""Execution-role metadata shared by SocialGym2 runners and analyses."""

from __future__ import annotations


POLICY_ROLE_SCHEMA = "socialgym2-policy-execution-roles-v1"

BELIEF_POLICIES = {
    "tracer-persistent": "persistent posterior",
    "tracer-no-pair": "persistent posterior with no pair interactions",
    "tracer-hazard": "persistent posterior with hazard update",
    "tracer-forgetting": "persistent posterior with forgetting",
    "tracer-change-point": "persistent posterior with change-point reset",
    "tracer-reset": "one-window reset posterior",
    "tracer-uniform": "fixed uniform mixture",
}

EXTERNAL_POLICIES = {
    "social-ref": "official SocialGym2/PIPS reference policy",
    "social-greedy": "official SocialGym2/PIPS greedy policy",
}


def socialgym2_policy_role(policy: str, counterfactual_enabled: bool) -> dict:
    if policy in EXTERNAL_POLICIES:
        return {
            "role": "official SocialGym2 external control",
            "response_planner_evaluated": False,
            "posterior_updated": False,
            "posterior_used_for_selection": False,
            "belief_policy": "not applicable",
            "pair_interactions": None,
            "same_state_uniform_counterfactual_evaluated": False,
            "external_source": EXTERNAL_POLICIES[policy],
        }
    if policy.startswith("official-"):
        return {
            "role": "shared-environment external control",
            "response_planner_evaluated": False,
            "posterior_updated": False,
            "posterior_used_for_selection": False,
            "belief_policy": "not applicable",
            "pair_interactions": None,
            "same_state_uniform_counterfactual_evaluated": False,
            "external_source": "benchmark control, no literature controller",
        }
    if policy not in BELIEF_POLICIES:
        raise ValueError(f"unknown SocialGym2 policy role: {policy}")
    inferred_belief = policy != "tracer-uniform"
    return {
        "role": "TRACER component policy",
        "response_planner_evaluated": True,
        "posterior_updated": inferred_belief,
        "posterior_used_for_selection": inferred_belief,
        "belief_policy": BELIEF_POLICIES[policy],
        "pair_interactions": policy != "tracer-no-pair",
        "same_state_uniform_counterfactual_evaluated": bool(counterfactual_enabled),
    }


def socialgym2_policy_roles(policies: list[str] | tuple[str, ...], counterfactual_enabled: bool) -> dict:
    return {
        policy: socialgym2_policy_role(policy, counterfactual_enabled)
        for policy in policies
    }
