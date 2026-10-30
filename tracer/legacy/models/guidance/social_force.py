import torch
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class SocialForceParams:
    """
    Parameters for diffusion-time social guidance.
    """

    sigma_long: float = 2.0
    sigma_lat: float = 1.0
    # If true, ignore neighbors behind the agent (front-only social constraint).
    front_only: bool = True

    # Collision shaping.
    collision_sigma: float = 0.8
    collision_margin: float = 0.5

    # Kinematic penalties.
    max_yaw_rate: float = 1.2
    max_lateral_speed: float = 0.3

    # Weights.
    w_social: float = 1.0
    w_collision: float = 1.0
    w_kinematic: float = 0.0

    # Small constant to avoid division-by-zero.
    eps: float = 1e-6


def integrate_velocities_to_positions(
    v: torch.Tensor,
    p0: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """
    v:  (B, T, 2) velocities
    p0: (B, 2) initial position at t=0
    return: (B, T, 2) positions for future steps 1..T in same frame as p0
    """
    disp = torch.cumsum(v * dt, dim=1)
    return p0[:, None, :] + disp


def _heading_from_velocity(v: torch.Tensor, eps: float) -> torch.Tensor:
    """
    v: (B, T, 2)
    return: heading angle theta (B, T) in radians
    """
    vx, vy = v[..., 0], v[..., 1]
    return torch.atan2(vy, vx + eps)


def anisotropic_social_energy(
    v: torch.Tensor,
    p0: torch.Tensor,
    neighbors_pos: torch.Tensor,
    dt: float,
    params: SocialForceParams,
    neighbor_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute anisotropic social potential energy.

    v:            (B, T, 2) predicted velocities for the controlled agent.
    p0:           (B, 2) initial position at t=0.
    neighbors_pos:(B, M, T, 2) neighbor positions over the same future horizon.
    neighbor_mask:(B, M) bool/0-1 mask; 1 means neighbor exists/active.

    Returns:
      energy: (B,) energy per batch element.
    """
    _, T, _ = v.shape
    _, _, Tn, _ = neighbors_pos.shape
    if Tn != T:
        raise ValueError(f"neighbors_pos horizon mismatch: got T={Tn}, expected {T}")

    pos = integrate_velocities_to_positions(v, p0, dt=dt)  # (B, T, 2)
    theta = _heading_from_velocity(v, eps=params.eps)  # (B, T)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    rel = neighbors_pos - pos[:, None, :, :]  # (B, M, T, 2)
    dx = rel[..., 0]
    dy = rel[..., 1]

    long = cos_t[:, None, :] * dx + sin_t[:, None, :] * dy
    lat = -sin_t[:, None, :] * dx + cos_t[:, None, :] * dy

    if params.front_only:
        front_mask = (long >= 0).to(v.dtype)
    else:
        front_mask = 1.0

    if neighbor_mask is not None:
        nmask = neighbor_mask.to(v.dtype)[:, :, None]  # (B, M, 1)
    else:
        nmask = 1.0

    denom_long = 2.0 * (params.sigma_long ** 2 + params.eps)
    denom_lat = 2.0 * (params.sigma_lat ** 2 + params.eps)
    e = torch.exp(-(long ** 2) / denom_long - (lat ** 2) / denom_lat)

    e = e * front_mask * nmask
    return e.sum(dim=(1, 2))


def collision_energy(
    v: torch.Tensor,
    p0: torch.Tensor,
    neighbors_pos: torch.Tensor,
    dt: float,
    params: SocialForceParams,
    neighbor_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Smooth collision/barrier energy from pairwise distance.
    """
    pos = integrate_velocities_to_positions(v, p0, dt=dt)  # (B, T, 2)
    rel = neighbors_pos - pos[:, None, :, :]  # (B, M, T, 2)
    dist = torch.sqrt((rel ** 2).sum(dim=-1) + params.eps)  # (B, M, T)

    gaussian = torch.exp(-(dist ** 2) / (2.0 * (params.collision_sigma ** 2 + params.eps)))
    barrier = torch.relu(params.collision_margin - dist) ** 2

    if neighbor_mask is not None:
        mask = neighbor_mask.to(v.dtype)[:, :, None]
        gaussian = gaussian * mask
        barrier = barrier * mask

    return (gaussian + barrier).sum(dim=(1, 2))


def kinematic_energy(
    v: torch.Tensor,
    dt: float,
    params: SocialForceParams,
    current_heading: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Soft penalty for yaw-rate/lateral-motion constraints.
    """
    theta = _heading_from_velocity(v, eps=params.eps)  # (B, T)

    if current_heading is None:
        prev_theta = theta[:, :1]
    else:
        prev_theta = current_heading[:, None]

    theta_all = torch.cat([prev_theta, theta], dim=1)  # (B, T+1)
    yaw_rate = (theta_all[:, 1:] - theta_all[:, :-1]) / max(float(dt), params.eps)
    yaw_pen = torch.relu(torch.abs(yaw_rate) - params.max_yaw_rate) ** 2

    if current_heading is None:
        lat_pen = torch.zeros_like(yaw_pen)
    else:
        vx, vy = v[..., 0], v[..., 1]
        sin0 = torch.sin(current_heading)[:, None]
        cos0 = torch.cos(current_heading)[:, None]
        lat_speed = -sin0 * vx + cos0 * vy
        lat_pen = torch.relu(torch.abs(lat_speed) - params.max_lateral_speed) ** 2

    return (yaw_pen + lat_pen).sum(dim=1)


def full_guidance_energy_and_grad(
    x0_v: torch.Tensor,
    p0: torch.Tensor,
    neighbors_pos: torch.Tensor,
    dt: float,
    params: SocialForceParams,
    neighbor_mask: Optional[torch.Tensor] = None,
    current_heading: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combined guidance energy and gradient for diffusion sampling.
    """
    x0_v = x0_v.requires_grad_(True)

    e_social = anisotropic_social_energy(
        v=x0_v,
        p0=p0,
        neighbors_pos=neighbors_pos,
        dt=dt,
        params=params,
        neighbor_mask=neighbor_mask,
    )
    e_collision = collision_energy(
        v=x0_v,
        p0=p0,
        neighbors_pos=neighbors_pos,
        dt=dt,
        params=params,
        neighbor_mask=neighbor_mask,
    )
    e_kin = kinematic_energy(
        v=x0_v,
        dt=dt,
        params=params,
        current_heading=current_heading,
    )

    total = params.w_social * e_social + params.w_collision * e_collision + params.w_kinematic * e_kin
    grad = torch.autograd.grad(total.sum(), x0_v, create_graph=False, retain_graph=False)[0]
    return total.detach(), grad


def energy_guidance_grad(
    x0_v: torch.Tensor,
    p0: torch.Tensor,
    neighbors_pos: torch.Tensor,
    dt: float,
    params: SocialForceParams,
    neighbor_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward-compatible social-only guidance gradient.
    """
    x0_v = x0_v.requires_grad_(True)
    E = anisotropic_social_energy(
        v=x0_v,
        p0=p0,
        neighbors_pos=neighbors_pos,
        dt=dt,
        params=params,
        neighbor_mask=neighbor_mask,
    )
    grad = torch.autograd.grad(E.sum(), x0_v, create_graph=False, retain_graph=False)[0]
    return E.detach(), grad
