from __future__ import annotations

from typing import Any, Mapping

from tracer.data.heterogeneous_scene import AgentRole, ROLE_TO_ID


DEFAULT_ROLE_RADIUS = {
    int(AgentRole.PHA): 3.6,
    int(AgentRole.HNAA): 5.0,
    int(AgentRole.SSAA): 6.0,
    int(AgentRole.UNKNOWN_ENTITY): 4.0,
}


def _role_key_to_id(key: object) -> int | None:
    if isinstance(key, AgentRole):
        return int(key)
    try:
        return int(key)
    except (TypeError, ValueError):
        return ROLE_TO_ID.get(str(key).lower())


def normalize_role_radius(raw: Mapping[object, object] | None) -> dict[int, float]:
    radius = dict(DEFAULT_ROLE_RADIUS)
    if raw:
        for key, value in raw.items():
            role_id = _role_key_to_id(key)
            if role_id is None:
                continue
            radius[int(role_id)] = float(value)
    return radius


def perception_kwargs_from_config(dataset_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    dataset_cfg = dataset_cfg or {}
    raw = dataset_cfg.get("perception", {}) or {}
    role_radius = normalize_role_radius(raw.get("role_radius", None))
    return {
        "perception_mode": str(raw.get("mode", "nearest")),
        "perception_base_radius": float(raw.get("base_radius", 3.6)),
        "perception_max_radius": float(raw.get("max_radius", 8.0)),
        "perception_speed_scale": float(raw.get("speed_scale", 0.8)),
        "perception_geometry_scale": float(raw.get("geometry_scale", 0.5)),
        "perception_role_radius": role_radius,
        "perception_fallback_to_nearest": bool(raw.get("fallback_to_nearest", True)),
    }


def effective_radius(
    role_id: int,
    speed: float,
    geometry: tuple[float, float] | None,
    *,
    base_radius: float,
    max_radius: float,
    speed_scale: float,
    geometry_scale: float,
    role_radius: Mapping[int, float],
) -> float:
    role_base = float(role_radius.get(int(role_id), role_radius.get(int(AgentRole.UNKNOWN_ENTITY), base_radius)))
    geom_bonus = 0.0
    if geometry is not None:
        length, width = geometry
        geom_bonus = geometry_scale * max(float(length), float(width), 0.0)
    radius = max(float(base_radius), role_base + speed_scale * max(float(speed), 0.0) + geom_bonus)
    return min(float(max_radius), radius)
