from __future__ import annotations

import heapq
from functools import lru_cache

import numpy as np

from tracer.response.unicycle_candidates import project_unicycle_candidates


class StaticRoute:
    def __init__(self, obstacles, goal, radius, resolution=0.08):
        self.goal = np.asarray(goal, dtype=np.float64)
        self.resolution = float(resolution)
        segments = np.asarray(obstacles, dtype=np.float64).reshape(-1, 2, 2)
        points = np.concatenate([segments.reshape(-1, 2), self.goal[None]])
        lower, upper = points.min(axis=0) - 1.0, points.max(axis=0) + 1.0
        self.origin = lower
        self.shape = np.ceil((upper - lower) / resolution).astype(int) + 1
        indices = np.indices(tuple(self.shape)).transpose(1, 2, 0)
        self.positions = lower + indices * resolution
        free = np.ones(tuple(self.shape), dtype=bool)
        for start, end in segments:
            direction = end - start
            fraction = ((self.positions - start) * direction).sum(-1) / max(float(direction @ direction), 1e-12)
            closest = start + fraction.clip(0, 1)[..., None] * direction
            free &= np.linalg.norm(self.positions - closest, axis=-1) >= radius + .03 + resolution / np.sqrt(2)
        self.free = free
        self.distance = np.full(tuple(self.shape), np.inf)
        self.next_index = np.full((*self.shape, 2), -1, dtype=int)
        goal_index = self._nearest_index(self.goal, free)
        self.distance[goal_index] = float(np.linalg.norm(self.positions[goal_index] - self.goal))
        queue = [(self.distance[goal_index], *goal_index)]
        while queue:
            cost, first, second = heapq.heappop(queue)
            if cost != self.distance[first, second]:
                continue
            for delta_first, delta_second in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                target = (first + delta_first, second + delta_second)
                if not (0 <= target[0] < self.shape[0] and 0 <= target[1] < self.shape[1]) or not free[target]:
                    continue
                if delta_first and delta_second and not (free[first, target[1]] and free[target[0], second]):
                    continue
                proposed = cost + resolution * np.hypot(delta_first, delta_second)
                if proposed < self.distance[target]:
                    self.distance[target] = proposed
                    self.next_index[target] = (first, second)
                    heapq.heappush(queue, (proposed, *target))

    def _nearest_index(self, position, mask):
        errors = np.linalg.norm(self.positions - np.asarray(position), axis=-1)
        errors[~mask] = np.inf
        if not np.isfinite(errors).any():
            raise ValueError("static navigation map contains no reachable grid cells")
        return tuple(np.unravel_index(errors.argmin(), errors.shape))

    def distance_at(self, positions):
        positions = np.asarray(positions, dtype=np.float64)
        flat = positions.reshape(-1, 2)
        result = []
        for position in flat:
            index = self._nearest_index(position, np.isfinite(self.distance))
            result.append(self.distance[index] + np.linalg.norm(position - self.positions[index]))
        return np.asarray(result).reshape(positions.shape[:-1])

    def waypoints(self, position):
        if np.linalg.norm(np.asarray(position) - self.goal) < 1e-7:
            return self.goal[None].copy()
        index = self._nearest_index(position, np.isfinite(self.distance))
        points = [np.asarray(position, dtype=np.float64), self.positions[index]]
        while self.next_index[index][0] >= 0:
            index = tuple(self.next_index[index])
            points.append(self.positions[index])
        points.append(self.goal)
        return np.asarray(points)

    def candidates(self, position, horizon, dt, speeds):
        path = self.waypoints(position)
        lengths = np.linalg.norm(np.diff(path, axis=0), axis=-1)
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        candidates = []
        for speed in speeds:
            progress = np.minimum(np.arange(1, horizon + 1) * dt * speed, cumulative[-1])
            coordinates = np.stack([np.interp(progress, cumulative, path[:, axis]) for axis in (0, 1)], axis=-1)
            velocity = np.diff(coordinates, axis=0, prepend=np.asarray(position)[None]) / dt
            candidates.append(np.concatenate([coordinates, velocity], axis=-1))
        return np.asarray(candidates, dtype=np.float32)

    def free_positions(self, positions):
        coordinates = np.asarray(positions)
        indices = np.rint((coordinates - self.origin) / self.resolution).astype(int)
        inside = ((indices >= 0) & (indices < self.shape)).all(axis=-1)
        clipped = np.clip(indices, 0, self.shape - 1)
        return inside & self.free[clipped[..., 0], clipped[..., 1]]

    def unicycle_candidates(self, position, velocity, heading, horizon, dt, speeds, limits, align_before_motion=False):
        path = self.waypoints(position)
        result, all_headings = [], []
        for speed in speeds:
            current = np.asarray(position, dtype=np.float64).copy()
            current_velocity = np.asarray(velocity, dtype=np.float64).copy()
            current_heading = float(heading)
            states, headings = [], []
            waypoint = 0
            for step in range(horizon):
                while waypoint < len(path) - 1 and np.linalg.norm(path[waypoint] - current) < .3:
                    waypoint += 1
                offset = path[waypoint] - current
                distance = float(np.linalg.norm(self.goal - current))
                speed_limit = min(speed, np.sqrt(2 * limits.deceleration * distance), distance / dt)
                target = np.zeros((1, 1, 4), dtype=np.float32)
                target[0, 0, 2:] = offset / max(float(np.linalg.norm(offset)), 1e-8) * speed_limit
                desired_heading = float(np.arctan2(offset[1], offset[0]))
                heading_error = (desired_heading - current_heading + np.pi) % (2 * np.pi) - np.pi
                if np.cos(heading_error) < 0:
                    heading_error = heading_error - np.sign(heading_error) * np.pi
                if align_before_motion and abs(heading_error) > limits.max_yaw_rate * dt:
                    target[...] = 0
                    projected, projected_heading = project_unicycle_candidates(
                        target, current, current_velocity, current_heading, dt, limits, return_headings=True,
                    )
                    if np.linalg.norm(current_velocity) < 1e-8:
                        projected_heading[0, 0] = current_heading + np.clip(heading_error, -limits.max_yaw_rate * dt, limits.max_yaw_rate * dt)
                    current = projected[0, 0, :2].astype(np.float64)
                    current_velocity = projected[0, 0, 2:].astype(np.float64)
                    current_heading = float(projected_heading[0, 0])
                    states.append(projected[0, 0])
                    headings.append(current_heading)
                    continue
                projected, projected_heading = project_unicycle_candidates(
                    target, current, current_velocity, current_heading, dt, limits, return_headings=True,
                )
                current = projected[0, 0, :2].astype(np.float64)
                current_velocity = projected[0, 0, 2:].astype(np.float64)
                current_heading = float(projected_heading[0, 0])
                states.append(projected[0, 0])
                headings.append(current_heading)
            result.append(states)
            all_headings.append(headings)
        return np.asarray(result, dtype=np.float32), np.asarray(all_headings)


@lru_cache(maxsize=32)
def _cached_route(obstacles, goal, radius):
    return StaticRoute(obstacles, goal, radius)


def static_route(obstacles, goal, radius):
    segments = tuple(tuple(tuple(float(coordinate) for coordinate in point) for point in segment) for segment in obstacles)
    return _cached_route(segments, tuple(float(coordinate) for coordinate in goal), float(radius))
