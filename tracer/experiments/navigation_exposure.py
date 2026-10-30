from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NavigationExposure:
    stagnation_time: float = 0.0
    human_velocity_variation: float = 0.0

    def observe(self, previous_distance, current_distance, previous_velocities, current_velocities, dt):
        previous = np.asarray(previous_velocities, dtype=float)
        current = np.asarray(current_velocities, dtype=float)
        if (previous.shape != current.shape or previous.ndim != 2 or previous.shape[-1] != 2
                or not np.isfinite(previous).all() or not np.isfinite(current).all()
                or not np.isfinite([previous_distance, current_distance, dt]).all() or dt <= 0):
            raise ValueError("navigation exposure requires aligned finite observations and a positive timestep")
        self.stagnation_time += float(dt) * int(previous_distance - current_distance <= 1e-4 * dt)
        if len(current):
            self.human_velocity_variation += float(np.linalg.norm(current - previous, axis=-1).mean())

    def snapshot(self):
        return {"stagnation_time": self.stagnation_time,
                "human_velocity_variation": self.human_velocity_variation}
