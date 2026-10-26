"""Run the TRACER loop in the official SocialGym2 ROS environment.

SocialGym2 exposes a discrete ``GoAlone``/``Halt`` action interface.  This
adapter therefore projects SAGE's continuous joint proposals onto the
official binary action lattice, evaluates the projected candidates with the
trained persistent response model, maintains one categorical belief per
tracked human, and executes the selected action through ``RosSocialEnv``.

The output is an external-environment integration result.  It is not a claim
about real human causality, and the action projection is reported explicitly
so that this experiment cannot be confused with the continuous local H3
protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sage.belief.filter import (
    apply_belief_forgetting,
    categorical_bayes_update,
    detect_change_point,
    hazard_belief_update,
    uniform_log_belief,
)
from sage.experiments.eval_persistent_closed_loop import _load_model
from sage.experiments.socialgym2_policy_roles import (
    EXTERNAL_POLICIES,
    POLICY_ROLE_SCHEMA,
    socialgym2_policy_roles,
)
from sage.experiments.socialgym2_execution_diagnostics import ExecutionDiagnostics, load_vector_map
from sage.experiments.socialgym2_joint_geometry import (
    add_observed_human_clearances,
    executable_candidate_prefix,
    joint_prefix_clearances,
    observable_route_directions,
    select_with_joint_geometry,
    stagnation_adjusted_costs,
    stagnation_recovery_mask,
)
from sage.response.closed_loop import (
    PlanningCostConfig,
    aggregate_robot_progress,
    expected_candidate_costs,
)
from sage.response.planning_support import belief_cost_adjustment
from sage.response.persistent_models import (
    stationary_reference_candidates,
    trajectory_mode_log_likelihood,
)
from sage.response.sage_candidate_adapter import (
    SAGECandidateAdapter,
    SAGECandidateAdapterConfig,
)


POLICIES = (
    "tracer-persistent",
    "tracer-no-pair",
    "tracer-hazard",
    "tracer-forgetting",
    "tracer-change-point",
    "tracer-reset",
    "tracer-uniform",
    "official-goalone",
    "official-halt",
    "official-random",
    "social-ref",
    "social-greedy",
)

EXTERNAL_ACTION_NAMES = {0: "GoAlone", 1: "Halt", 2: "Follow", 3: "Pass"}
EXTERNAL_ACTION_TO_BINARY = {0: 0, 1: 1, 2: 0, 3: 0}


@dataclass
class HumanTrack:
    track_id: int
    position: np.ndarray
    velocity: np.ndarray
    history: list[np.ndarray] = field(default_factory=list)
    belief: np.ndarray | None = None
    missed: int = 0
    observed: bool = True
    observed_velocity: np.ndarray | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except subprocess.CalledProcessError:
        # Evaluation source snapshots are intentionally copied without the
        # research repository's .git directory.  Keep the run reproducible
        # through algorithm/runtime/checkpoint hashes instead of aborting
        # before the simulator is reached.
        return {"commit": "source-snapshot-no-git", "branch": "", "dirty": True}


def _finite(value: float, fallback: float = 0.0) -> float:
    value = float(value)
    return value if math.isfinite(value) else fallback


def _seed_episode(seed: int) -> int:
    episode_seed = int(seed)
    random.seed(episode_seed)
    np.random.seed(episode_seed)
    torch.manual_seed(episode_seed)
    return episode_seed


def _episode_seed(base_seed: int, episode: int) -> int:
    if episode < 0 or episode >= 1000:
        raise ValueError("episode must be in [0, 999] for the disjoint seed stream")
    return int(base_seed) * 1000 + int(episode)


def _generated_scenario_hash(source: Path) -> str:
    digest = hashlib.sha256()
    config_dir = source / "config/gym_gen"
    for name in ("launch.launch", "pedsim_launch.launch", "humans.lua", "scene.xml", "sim_config.lua"):
        path = config_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_calibration_artifact(
    path: Path,
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("protocol") != "tracer-socialgym2-joint-belief-calibration-v1":
        raise ValueError(f"unsupported SocialGym2 calibration artifact: {path}")
    if artifact.get("source_protocol") != "tracer-socialgym2-joint-belief-synchronous-v5":
        raise ValueError("calibration uses an obsolete external runtime protocol")
    if artifact.get("phase") != "validation" or artifact.get("split") != "validation":
        raise ValueError("SocialGym2 calibration must use validation data only")
    if args.seed in artifact.get("validation_seeds", []):
        raise ValueError("evaluation seed overlaps SocialGym2 calibration seeds")
    expected = {
        "sim_dt": float(args.sim_dt),
        "sim_steps_per_planning_window": int(args.sim_steps_per_window),
        "planning_window_dt": float(args.env_dt),
        "num_agents": int(args.num_agents),
        "num_humans": int(args.num_humans),
    }
    for field, value in expected.items():
        observed = artifact.get(field)
        if isinstance(value, float):
            valid = math.isclose(float(observed), value, rel_tol=0.0, abs_tol=1e-7)
        else:
            valid = observed == value
        if not valid:
            raise ValueError(
                f"calibration artifact {path} has incompatible {field}: "
                f"{observed!r}, expected {value!r}"
            )
    expected_hash = _sha256(checkpoint_path)
    if artifact.get("source_checkpoint_sha256") != expected_hash:
        raise ValueError(f"calibration artifact {path} does not match the response checkpoint")
    selected = artifact.get("selected", {})
    selected_scale = float(selected.get("likelihood_scale", float("nan")))
    if not math.isfinite(selected_scale) or selected_scale <= 0.0:
        raise ValueError(f"calibration artifact {path} has no valid selected scale")
    if not math.isclose(float(args.likelihood_scale), selected_scale, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "likelihood_scale does not match the SocialGym2 validation calibration: "
            f"runtime={args.likelihood_scale}, calibrated={selected_scale}"
        )
    return artifact


def _initial_state_hash(
    robot_positions: np.ndarray,
    robot_velocities: np.ndarray,
    goals: np.ndarray,
    observations: list[tuple[np.ndarray, np.ndarray]],
) -> str:
    digest = hashlib.sha256()
    for name, values in (
        ("robots", robot_positions),
        ("robot_velocities", robot_velocities),
        ("goals", goals),
    ):
        digest.update(name.encode("utf-8"))
        digest.update(np.round(np.asarray(values, dtype=np.float32), 4).tobytes())
    human_values = sorted(
        (
            tuple(np.round(np.asarray(position, dtype=np.float32), 4)),
            tuple(np.round(np.asarray(velocity, dtype=np.float32), 4)),
        )
        for position, velocity in observations
    )
    digest.update(repr(human_values).encode("utf-8"))
    return digest.hexdigest()


def _algorithm_sources(root: Path) -> dict[str, str]:
    sources = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith("sage.") or not getattr(module, "__file__", None):
            continue
        path = Path(module.__file__).resolve()
        if path.suffix == ".py" and path.is_relative_to(root):
            sources[str(path.relative_to(root))] = _sha256(path)
    sources[str(Path(__file__).resolve().relative_to(root))] = _sha256(Path(__file__))
    return sources


def _runtime_manifest(path: Path | None, source: Path) -> dict[str, Any] | None:
    if path is None:
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "socialgym2-joint-synchronous-runtime-v5":
        raise ValueError("incompatible external runtime manifest")
    for relative, hashes in manifest["sources"].items():
        if _sha256(source / relative) != hashes["patched_sha256"]:
            raise ValueError(f"external runtime source changed: {relative}")
    for binary, expected in manifest["binaries"].items():
        if _sha256(Path(binary)) != expected:
            raise ValueError(f"external runtime binary changed: {binary}")
    return {"sha256": _sha256(path), "protocol": manifest["protocol"], "boundary": manifest["boundary"]}


def _pose_xy(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size < 2:
        return np.zeros(2, dtype=np.float32)
    return array[:2].copy()


def _pose_theta(value: Any) -> float:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return float(array[2]) if array.size >= 3 and math.isfinite(float(array[2])) else 0.0


def _navigation_state(responses: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return heading, angular rate, and graph target in robot-local xy."""
    headings = np.asarray([_pose_theta(response.robot_poses) for response in responses], dtype=np.float32)
    angular_rates = np.asarray([
        float(np.asarray(response.robot_vels, dtype=np.float32).reshape(-1)[2])
        if np.asarray(response.robot_vels).reshape(-1).size >= 3 else 0.0
        for response in responses
    ], dtype=np.float32)
    local_targets = np.asarray([
        _pose_xy(getattr(response, "local_target", np.zeros(2, dtype=np.float32)))
        for response in responses
    ], dtype=np.float32)
    return headings, angular_rates, local_targets


def _rotate(vector: np.ndarray, angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        [cosine * float(vector[0]) - sine * float(vector[1]),
         sine * float(vector[0]) + cosine * float(vector[1])],
        dtype=np.float32,
    )


def _build_env(
    source: Path,
    num_agents: int,
    num_humans: int,
    partial: bool,
    launch_config: Path,
    scenario_name: str,
    external_policy: str | None = None,
):
    """Construct the official environment without changing its source."""

    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "src"))
    from src.environment import RosSocialEnv
    from src.environment.observations import (
        AgentsGoalDistance,
        AgentsPose,
        CollisionObservation,
        OtherAgentObservables,
        SuccessObservation,
    )
    from src.environment.observations.observer import Observer
    from src.environment.rewards import ExistencePenalty, GoalDistanceChange, Rewarder
    from src.environment.scenarios.common_scenarios import (
        envs_intersection,
        envs_open,
    )
    import rospy
    from std_srvs.srv import Trigger
    from ut_multirobot_sim.srv import utmrsStepper

    for module_name, module in list(sys.modules.items()):
        if module_name.startswith("src.environment") and hasattr(module, "ROOT_FOLDER"):
            module.ROOT_FOLDER = source
            if hasattr(module, "DATA_FOLDER"):
                module.DATA_FOLDER = source / "data"

    factories = {
        "envs/scenario/intersection": envs_intersection,
        "envs/scenario/open": envs_open,
    }
    if scenario_name not in factories:
        raise ValueError(f"unsupported SocialGym2 scenario: {scenario_name}")
    scenario, _ = factories[scenario_name](
        partially_observable=partial,
        config_runner=True,
        all_config=False,
    )

    original_make_scenario = scenario.make_scenario

    def make_runtime_scenario(config):
        scenario.h3_expected_robot_positions = np.asarray(config["robot_start"], dtype=np.float32)[:, :2]
        scenario.h3_expected_goals = np.asarray(config["robot_end"], dtype=np.float32)[:, :2]
        result = original_make_scenario(config)
        sim_config = source / "config/gym_gen/sim_config.lua"
        text = sim_config.read_text(encoding="utf-8")
        map_path = (
            source
            / "submodules/ut_multirobot_sim/maps"
            / scenario_name
            / f"{Path(scenario_name).name}.vectormap.txt"
        )
        text = re.sub(
            r"(?m)^\s*human_config\s*=.*$",
            f'  human_config = "{source / "config/gym_gen/humans.lua"}"',
            text,
        )
        text = re.sub(
            r"(?m)^\s*robot_config\s*=.*$",
            f'  robot_config = "{source / "submodules/ut_multirobot_sim/config/ut_jackal_config.lua"}"',
            text,
        )
        text = re.sub(r"(?m)^\s*map_name\s*=.*$", f'  map_name = "{map_path}"', text)
        text = re.sub(
            r"(?m)^\s*nav_map_name\s*=.*$",
            f'  nav_map_name = "{Path(scenario_name).name}"',
            text,
        )
        sim_config.write_text(text, encoding="utf-8")
        scene_path = source / "config/gym_gen/scene.xml"
        scene_tree = ET.parse(scene_path)
        scene_root = scene_tree.getroot()
        for agent in list(scene_root.findall("agent")):
            if agent.get("type") == "2":
                scene_root.remove(agent)
        scene_tree.write(scene_path, encoding="utf-8", xml_declaration=True)
        maps_dir = source / "submodules/ut_multirobot_sim/maps" / Path(scenario_name).parent
        launch_text = launch_config.read_text(encoding="utf-8")
        launch_text = re.sub(
            r'args="-service_mode=true (?:-maps_dir=\S+ )?-map=[^"]+"',
            f'args="-service_mode=true -maps_dir={maps_dir} -map={Path(scenario_name).name}"',
            launch_text,
        )
        launch_text = launch_text.replace("--localize --use_pedsim", "--localize --use_pedsim --h3_synchronous=true")
        launch_text = launch_text.replace("$(find social_gym)", str(source))
        launch_text = re.sub(
            r"\n?\s*<!-- TRACER_EXTERNAL_CONTROLLER_BEGIN -->.*?"
            r"<!-- TRACER_EXTERNAL_CONTROLLER_END -->\s*\n?",
            "\n",
            launch_text,
            flags=re.DOTALL,
        )
        if external_policy in EXTERNAL_POLICIES:
            executable = {
                "social-ref": "social_ref",
                "social-greedy": "social_greed",
            }[external_policy]
            nodes = "\n".join(
                f'    <group ns="tracer_external_{index}">\n'
                f'        <node pkg="pips" type="{executable}" '
                f'name="{executable}" output="screen" />\n'
                "    </group>"
                for index in range(num_agents)
            )
            nodes = (
                "    <!-- TRACER_EXTERNAL_CONTROLLER_BEGIN -->\n"
                f"{nodes}\n"
                "    <!-- TRACER_EXTERNAL_CONTROLLER_END -->"
            )
            launch_marker = "    <param name=\"enable_statistics\" value=\"true\" />"
            if launch_marker not in launch_text:
                raise RuntimeError("SocialGym2 launch template has no external-controller insertion point")
            launch_text = launch_text.replace(
                launch_marker,
                f"{nodes}\n\n{launch_marker}",
                1,
            )
        launch_config.write_text(launch_text, encoding="utf-8")
        pedsim_launch = source / "config/gym_gen/pedsim_launch.launch"
        pedsim_text = pedsim_launch.read_text(encoding="utf-8")
        pedsim_text = pedsim_text.replace("$(find social_gym)config", f"{source}/config")
        pedsim_text = pedsim_text.replace("$(find social_gym)", str(source))
        pedsim_launch.write_text(pedsim_text, encoding="utf-8")
        return result

    scenario.make_scenario = make_runtime_scenario
    scenario.generate_scenario(num_humans=num_humans, num_agents=num_agents)
    observer = Observer(
        [
            AgentsGoalDistance(history_length=2),
            AgentsPose(ignore_theta=True),
            CollisionObservation(),
            SuccessObservation(),
            OtherAgentObservables(pos_x=True, pos_y=True, pos_theta=False),
        ]
    )
    rewarder = Rewarder([ExistencePenalty(1.0), GoalDistanceChange()])
    # Generate a correctly mapped configuration before construction because
    # the official constructor launches ROS before its first reset callback.
    # Suppress that duplicate callback and regenerate the measured episode
    # after the ROS graph is ready.
    original_new_scenario = RosSocialEnv.new_scenario
    RosSocialEnv.new_scenario = lambda self, *args, **kwargs: None
    try:
        env = RosSocialEnv(
            observer=observer,
            rewarder=rewarder,
            scenarios=[scenario],
            num_humans=num_humans,
            num_agents=num_agents,
            debug=False,
            launch_config=str(launch_config),
        )
    finally:
        RosSocialEnv.new_scenario = original_new_scenario
    for service_name in ("/graphNavSrv", "/socialNavReset", "/pedsim_simulator/ResetPedsim", "/utmrsObserve", "/utmrsClock"):
        rospy.wait_for_service(service_name, timeout=30)
    if external_policy in EXTERNAL_POLICIES:
        from amrl_msgs.srv import SocialPipsSrv

        external_services = []
        for index in range(num_agents):
            service_name = f"/tracer_external_{index}/SocialPipsSrv"
            rospy.wait_for_service(service_name, timeout=60)
            external_services.append(rospy.ServiceProxy(service_name, SocialPipsSrv))
        env.socialgym2_external_services = external_services
    observe_service = rospy.ServiceProxy("/utmrsObserve", utmrsStepper)
    clock_service = rospy.ServiceProxy("/utmrsClock", Trigger)
    def synchronous_reset(seed=None, return_info=True, options=None):
        _assert_ros_runtime_health()
        rospy.set_param("/h3_runtime/episode_seed", int(seed or 0))
        env.agents = env.real_possible_agents[:]
        env.possible_agents = env.agents
        env.visuals = []
        env.terminations = {agent: False for agent in env.agents}
        env.terminations_ = [False] * len(env.agents)
        env.observer.reset()
        env.rewarder.reset()
        time.sleep(0.25)
        env.utmrs_service.reset()
        if rospy.get_param("/h3_runtime/pedsim_robot_count", 0) != env.curr_num_agents:
            raise RuntimeError("pedestrian dynamics did not receive every controllable robot")
        responses = env.env_response_type.process(observe_service(*env.default_action()))
        if len(responses) != env.curr_num_agents:
            raise RuntimeError("synchronous reset returned the wrong robot count")
        reset_positions, reset_velocities, reset_goals, _ = _scene_from_responses(responses, partial)
        if not np.allclose(reset_positions, scenario.h3_expected_robot_positions, atol=1e-6):
            raise RuntimeError("reset robot positions do not match the freshly generated scene")
        if not np.allclose(reset_goals, scenario.h3_expected_goals, atol=1e-6):
            raise RuntimeError("reset goals do not match the freshly generated scene")
        if not np.allclose(reset_velocities, 0.0, atol=1e-7):
            raise RuntimeError("zero-step reset returned moving robots")
        observations, observation_maps = env.make_observation(responses)
        env.last_obs_maps = []
        env.last_reward_maps = []
        return dict(zip(env.agents, observations)), {agent: {} for agent in env.agents}

    def simulation_time():
        result = clock_service()
        if not result.success:
            raise RuntimeError("synchronous SocialGym2 runtime did not certify its clock")
        return float(result.message)

    env.reset = synchronous_reset
    env.h3_simulation_time = simulation_time
    original_new_scenario(env, num_humans=num_humans, num_agents=num_agents)
    return env


def _pose2df(value: Any, message_type: Any) -> Any:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    message = message_type()
    message.x = float(array[0]) if array.size > 0 else 0.0
    message.y = float(array[1]) if array.size > 1 else 0.0
    message.theta = float(array[2]) if array.size > 2 else 0.0
    return message


def _query_external_socialgym_policy(env: Any, responses: list[Any], robot_index: int) -> int:
    from amrl_msgs.msg import Pose2Df

    response = responses[robot_index]
    human_poses = [_pose2df(value, Pose2Df) for value in (response.human_poses or [])]
    human_velocities = [_pose2df(value, Pose2Df) for value in (response.human_vels or [])]
    while len(human_velocities) < len(human_poses):
        human_velocities.append(_pose2df(np.zeros(3, dtype=np.float32), Pose2Df))
    raw_action = int(env.socialgym2_external_services[robot_index](
        [_pose2df(response.robot_poses, Pose2Df)],
        [_pose2df(response.robot_vels, Pose2Df)],
        human_poses,
        human_velocities[:len(human_poses)],
        _pose2df(response.goal_pose, Pose2Df),
        _pose2df(response.local_target, Pose2Df),
        _pose2df(response.door_pose, Pose2Df),
        int(response.door_state),
        int(response.robot_state),
        int(response.follow_target),
    ).action)
    if raw_action not in EXTERNAL_ACTION_NAMES:
        raise RuntimeError(f"official SocialGym2 external policy returned invalid action {raw_action}")
    return raw_action


def _executed_socialgym_action(
    policy: str,
    external_action: np.ndarray | None,
    ordered_actions: np.ndarray,
    selected: int,
) -> np.ndarray:
    if policy in EXTERNAL_POLICIES:
        if external_action is None:
            raise ValueError("external SocialGym2 policy did not supply an action")
        action = np.asarray(external_action, dtype=np.int64)
        if action.ndim != 1 or not np.isin(action, tuple(EXTERNAL_ACTION_NAMES)).all():
            raise ValueError("external SocialGym2 action is outside the native four-action space")
        return action
    return np.asarray(ordered_actions[selected], dtype=np.int64)


def _external_source_hashes(source: Path, policy: str | None) -> dict[str, str]:
    if policy not in EXTERNAL_POLICIES:
        return {}
    files = [
        "submodules/pips/manifest.xml",
        "submodules/pips/CMakeLists.txt",
        "submodules/pips/src/policy/social_lib.cpp",
    ]
    files.append(
        "submodules/pips/src/policy/social_ref.cpp"
        if policy == "social-ref"
        else "submodules/pips/src/policy/greedy_ref.cpp"
    )
    return {relative: _sha256(source / relative) for relative in files}


def _external_runtime_controller(source: Path, policy: str | None) -> dict[str, Any]:
    if policy not in EXTERNAL_POLICIES:
        return {}
    import rospkg

    package = Path(rospkg.RosPack().get_path("pips")).resolve()
    executable = "social_ref" if policy == "social-ref" else "social_greed"
    binary = package / "bin" / executable
    if not binary.is_file():
        raise FileNotFoundError(f"official SocialGym2 external binary is missing: {binary}")
    runtime_hashes = {
        relative: _sha256(package / relative.removeprefix("submodules/pips/"))
        for relative in _external_source_hashes(source, policy)
    }
    if runtime_hashes != _external_source_hashes(source, policy):
        raise ValueError("PIPS runtime binary was built from a different official source revision")
    return {
        "ros_package": "pips",
        "package_path": str(package),
        "executable": executable,
        "binary_path": str(binary),
        "binary_sha256": _sha256(binary),
        "runtime_source_hashes": runtime_hashes,
    }


def _capture_observations(env: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    original = env.make_observation

    def wrapped(responses):
        captured["responses"] = responses
        return original(responses)

    env.make_observation = wrapped
    return captured


def _assert_ros_runtime_health() -> None:
    """Reject an episode when an official simulator node has died."""

    try:
        import rosnode
        import rosservice
    except ImportError:
        return
    names = set(rosnode.get_node_names())
    required_nodes = {"/simulator", "/pedsim_simulator", "/graph_navigation"}
    missing_nodes = sorted(required_nodes - names)
    if missing_nodes:
        raise RuntimeError(f"SocialGym2 runtime lost required nodes: {missing_nodes}")
    required_services = {"/utmrsStepper", "/utmrsReset", "/graphNavSrv", "/utmrsObserve", "/utmrsClock"}
    available_services = set(rosservice.get_service_list())
    missing_services = sorted(required_services - available_services)
    if missing_services:
        raise RuntimeError(
            f"SocialGym2 runtime lost required services: {missing_services}"
        )


def _scene_from_responses(
    responses: list[Any],
    partial_observation: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    robots = []
    robot_velocities = []
    goals = []
    observations: list[tuple[np.ndarray, np.ndarray, int]] = []
    for source_robot, response in enumerate(responses):
        robot_pose = _pose_xy(response.robot_poses)
        robot_angle = _pose_theta(response.robot_poses)
        robot_velocity = _pose_xy(response.robot_vels)
        robots.append(robot_pose)
        robot_velocities.append(_rotate(robot_velocity, robot_angle))
        goals.append(_pose_xy(response.goal_pose))
        poses = list(response.human_poses or [])
        velocities = list(response.human_vels or [])
        for index, pose in enumerate(poses):
            local_position = _pose_xy(pose)
            local_velocity = _pose_xy(velocities[index]) if index < len(velocities) else np.zeros(2, dtype=np.float32)
            # SocialGym2's UTMRS response is robot-centric.  A zero pose is
            # also the official partially-observable sentinel, not a human at
            # the robot origin.
            if partial_observation and np.linalg.norm(local_position) <= 1e-8:
                continue
            position = robot_pose + _rotate(local_position, robot_angle)
            velocity = _rotate(local_velocity + robot_velocity, robot_angle)
            observations.append((position, velocity, source_robot))

    if not robots:
        raise RuntimeError("SocialGym2 returned no robot responses")
    robot_positions = np.asarray(robots, dtype=np.float32)
    robot_velocities_array = np.asarray(robot_velocities, dtype=np.float32)
    goal_positions = np.asarray(goals, dtype=np.float32)

    # The official response has no persistent human IDs.  Merge the same
    # visible human reported by several robots before the track association.
    merged: list[tuple[np.ndarray, np.ndarray, set[int]]] = []
    for position, velocity, source_robot in observations:
        match = next(
            (
                index
                for index, item in enumerate(merged)
                if source_robot not in item[2]
                and np.linalg.norm(item[0] - position) <= 0.70
                and np.linalg.norm(item[1] - velocity) <= 1.25
            ),
            None,
        )
        if match is None:
            merged.append((position, velocity, {source_robot}))
        else:
            old_position, old_velocity, sources = merged[match]
            count = len(sources)
            weight = float(count) / float(count + 1)
            merged[match] = (
                old_position * weight + position / float(count + 1),
                old_velocity * weight + velocity / float(count + 1),
                sources | {source_robot},
            )
    return robot_positions, robot_velocities_array, goal_positions, [
        (position, velocity) for position, velocity, _ in merged
    ]


def _update_tracks(
    tracks: list[HumanTrack],
    observations: list[tuple[np.ndarray, np.ndarray]],
    history: int,
    dt: float,
    next_id: int,
    max_missed: int,
) -> int:
    for track in tracks:
        track.observed = False
        track.observed_velocity = None

    predictions = [track.position + track.velocity * float(dt) for track in tracks]
    candidates = sorted(
        (
            float(np.linalg.norm(predictions[track_index] - position)),
            track_index,
            observation_index,
        )
        for track_index in range(len(tracks))
        for observation_index, (position, _) in enumerate(observations)
    )
    used_tracks: set[int] = set()
    used_observations: set[int] = set()
    for distance, track_index, observation_index in candidates:
        if distance > 1.35 or track_index in used_tracks or observation_index in used_observations:
            continue
        track = tracks[track_index]
        position, velocity = observations[observation_index]
        track.position = position.astype(np.float32)
        track.velocity = velocity.astype(np.float32)
        track.history.append(track.position.copy())
        track.history = track.history[-history:]
        track.missed = 0
        track.observed = True
        track.observed_velocity = track.velocity.copy()
        used_tracks.add(track_index)
        used_observations.add(observation_index)

    for track_index, track in enumerate(tracks):
        if track_index not in used_tracks:
            track.position = (track.position + track.velocity * float(dt)).astype(np.float32)
            track.history.append(track.position.copy())
            track.history = track.history[-history:]
            track.missed += 1

    for observation_index, (position, velocity) in enumerate(observations):
        if observation_index in used_observations:
            continue
        tracks.append(
            HumanTrack(
                track_id=next_id,
                position=position.astype(np.float32),
                velocity=velocity.astype(np.float32),
                history=[position.astype(np.float32).copy()],
            )
        )
        next_id += 1

    tracks[:] = [track for track in tracks if track.missed <= max_missed]
    return next_id


def _track_history(track: HumanTrack, history: int, dt: float) -> np.ndarray:
    values = [item.copy() for item in track.history]
    if not values:
        values = [track.position.copy()]
    while len(values) < history:
        values.insert(0, values[0] - track.velocity * float(dt))
    return np.asarray(values[-history:], dtype=np.float32)


def _action_lattice(num_agents: int) -> np.ndarray:
    return np.asarray(list(itertools.product((0, 1), repeat=num_agents)), dtype=np.int64)


def _completed_robot_action_lattice(actions: np.ndarray, completed: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions)
    completed = np.asarray(completed, dtype=bool)
    if actions.ndim != 2 or completed.shape != (actions.shape[1],):
        raise ValueError("completed robot state must align with joint actions")
    if not np.isin(actions, [0, 1]).all():
        raise ValueError("joint actions must contain only GoAlone and Halt")
    retained = actions[np.all(actions[:, completed] == 1, axis=1)]
    if not len(retained):
        raise ValueError("joint support has no Halt action for completed robots")
    return retained


def _project_sage_order(
    sage_activities: np.ndarray | None,
    action_lattice: np.ndarray,
) -> np.ndarray:
    """Order official binary branches by proximity to SAGE proposal activity."""

    if sage_activities is None or len(sage_activities) == 0:
        return np.arange(len(action_lattice), dtype=np.int64)
    activity = np.asarray(sage_activities, dtype=np.float32)
    activity = activity[:, : action_lattice.shape[1]]
    proposal_binary = (activity >= 0.45).astype(np.float32)
    distances = np.linalg.norm(
        proposal_binary[:, None, :] - action_lattice[None, :, :], axis=-1
    )
    nearest = distances.min(axis=0)
    # The planner still evaluates every feasible official branch.  The order
    # only makes deterministic ties follow the SAGE proposal geometry.
    return np.lexsort((np.arange(len(action_lattice)), nearest)).astype(np.int64)


@torch.no_grad()
def _response_predictions(
    model: torch.nn.Module,
    tracks: list[HumanTrack],
    robot_positions: np.ndarray,
    robot_velocities: np.ndarray,
    candidate_world_positions: np.ndarray,
    candidate_world_velocities: np.ndarray,
    history: int,
    dt: float,
    device: str,
    include_pair_interactions: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate all action branches for every active human track."""

    if not tracks:
        empty = np.empty((0, len(candidate_world_positions), model.config.num_modes, model.config.horizon, 2), dtype=np.float32)
        return empty, empty.copy(), empty.copy(), np.empty((0, len(candidate_world_positions), robot_positions.shape[0], 4), dtype=np.float32)
    track_count = len(tracks)
    candidate_count = len(candidate_world_positions)
    horizon = candidate_world_positions.shape[2]
    robots = robot_positions.shape[0]
    histories = np.asarray([_track_history(track, history, dt) for track in tracks], dtype=np.float32)
    current = np.zeros((track_count, candidate_count, robots, 4), dtype=np.float32)
    actions = np.zeros((track_count, candidate_count, robots, horizon, 4), dtype=np.float32)
    for track_index, track in enumerate(tracks):
        relative = robot_positions - track.position[None]
        current[track_index, :, :, :2] = relative[None]
        current[track_index, :, :, 2:] = robot_velocities[None]
        actions[track_index, :, :, :, :2] = candidate_world_positions - track.position[None, None, None, :]
        actions[track_index, :, :, :, 2:] = candidate_world_velocities

    batch_size = track_count * candidate_count
    history_tensor = torch.from_numpy(histories[:, None]).to(device=device).expand(
        track_count, candidate_count, history, 2
    ).reshape(batch_size, 1, history, 2)
    current_tensor = torch.from_numpy(current).to(device=device).reshape(batch_size, 1, robots, 4)
    action_tensor = torch.from_numpy(actions).to(device=device).reshape(batch_size, 1, robots, horizon, 4)
    mask = torch.ones(batch_size, 1, robots, device=device, dtype=torch.bool)
    output = model(
        history_tensor,
        current_tensor,
        action_tensor,
        include_pair_interactions=include_pair_interactions,
        robot_mask=mask,
    )
    means = output["mean"][:, 0].reshape(track_count, candidate_count, model.config.num_modes, horizon, 2)
    log_stds = output["log_std"][:, 0].reshape_as(means)
    reference = stationary_reference_candidates(current_tensor, horizon)
    reference_output = model(
        history_tensor,
        current_tensor,
        reference,
        include_pair_interactions=include_pair_interactions,
        robot_mask=mask,
    )
    references = reference_output["mean"][:, 0].reshape_as(means)
    return (
        means.detach().cpu().numpy(),
        log_stds.detach().cpu().numpy(),
        references.detach().cpu().numpy(),
        current,
    )


def _select_action(
    policy: str,
    model: torch.nn.Module,
    tracks: list[HumanTrack],
    robot_positions: np.ndarray,
    robot_velocities: np.ndarray,
    goals: np.ndarray,
    actions: np.ndarray,
    candidate_positions: np.ndarray,
    candidate_velocities: np.ndarray,
    history: int,
    model_dt: float,
    args: argparse.Namespace,
    geometric_clearances: dict[str, np.ndarray] | None = None,
    task_directions: np.ndarray | None = None,
    stagnant_robots: np.ndarray | None = None,
    task_active_robots: np.ndarray | None = None,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    if task_directions is None:
        task_directions = (goals - robot_positions) / np.maximum(
            np.linalg.norm(goals - robot_positions, axis=-1, keepdims=True), 1e-6,
        )
    if stagnant_robots is None:
        stagnant_robots = np.zeros(len(robot_positions), dtype=bool)
    if task_active_robots is None:
        task_active_robots = np.ones(len(robot_positions), dtype=bool)
    task_active_robots = np.asarray(task_active_robots, dtype=bool)
    if task_active_robots.shape != (len(robot_positions),):
        raise ValueError("task_active_robots must align with robot positions")
    task_aggregation = str(getattr(args, "task_aggregation", "mean")).lower()
    task_min_blend = float(getattr(args, "task_min_blend", 0.5))
    task_active_mask = (
        torch.from_numpy(task_active_robots).to(args.device)
        if task_aggregation != "mean"
        else None
    )
    means, log_stds, references, current = _response_predictions(
        model,
        tracks,
        robot_positions,
        robot_velocities,
        candidate_positions,
        candidate_velocities,
        history,
        args.env_dt,
        args.device,
        include_pair_interactions=policy != "tracer-no-pair",
    )
    if not tracks:
        selected = 0
        if geometric_clearances is not None and policy not in EXTERNAL_POLICIES:
            progress_by_robot = (
                (candidate_positions[:, :, getattr(args, "task_steps", 1) - 1] - robot_positions)
                * task_directions
            ).sum(axis=-1)
            progress = aggregate_robot_progress(
                torch.from_numpy(progress_by_robot).to(args.device),
                task_aggregation,
                task_active_mask,
                task_min_blend,
            ).cpu().numpy()
            costs = stagnation_adjusted_costs(
                -progress, actions, stagnant_robots, getattr(args, "stagnation_halt_penalty", 0.0),
            )
            selected, _ = select_with_joint_geometry(
                costs,
                np.ones(len(actions), dtype=bool),
                np.zeros(len(actions)),
                geometric_clearances,
                actions,
                stagnant_robots,
                allow_hard_safe_approach=getattr(args, "stagnation_release_hard_safe", False),
                require_robust_robot_clearance=getattr(
                    args, "stagnation_release_require_robust_robot", False
                ),
                require_nonworsening_robot_clearance=getattr(
                    args, "stagnation_release_require_nonworsening_robot", False
                ),
                robot_clearance_floor=getattr(
                    args, "stagnation_release_robot_clearance_floor", None
                ),
                recovery_safety_first=getattr(args, "stagnation_recovery_safety_first", False),
            )
        return selected, means, log_stds, references, {
            "fallback": 1.0,
            "safe_fraction": 1.0,
            "min_violation_probability": 0.0,
            "belief_selection_changed": 0.0,
            "belief_action_changed": 0.0,
            "belief_score_l1": 0.0,
            "belief_selected_regret": 0.0,
            "stagnant_robot_count": int(np.sum(stagnant_robots)),
        }

    planning = PlanningCostConfig(
        task_weight=1.0,
        safety_weight=args.safety_weight,
        human_weight=args.human_cost_weight,
        comfort_scale=1.0,
        collision_margin=args.collision_margin,
        task_steps=getattr(args, "task_steps", 1),
        response_steps=getattr(args, "response_cost_steps", None),
        violation_probability_alpha=(args.risk_alpha if args.hard_risk_filter else None),
        risk_samples=args.risk_samples,
        risk_weight=args.risk_weight,
        smoothness_weight=args.smoothness_weight,
        task_aggregation=task_aggregation,
        task_min_blend=task_min_blend,
    )
    task_direction = torch.from_numpy(np.asarray(task_directions, dtype=np.float32)).to(args.device)
    executable_task = None
    if getattr(args, "candidate_prefix_model", "goal-line-legacy") == "executable":
        candidate_tensor = torch.from_numpy(candidate_positions).to(args.device)
        current_positions = torch.from_numpy(robot_positions).to(args.device)
        progress_by_robot = (
            (candidate_tensor[:, :, args.task_steps - 1] - current_positions[None])
            * task_direction[None]
        ).sum(dim=-1)
        moving = torch.from_numpy(
            (np.linalg.norm(robot_velocities, axis=-1) > 0.05).astype(np.float32)
        ).to(args.device)
        effective_steps = (args.task_steps - 1) + moving
        executable_task = -aggregate_robot_progress(
            progress_by_robot
            / (effective_steps[None] * args.model_dt).clamp_min(args.model_dt),
            task_aggregation,
            task_active_mask,
            task_min_blend,
        )

    def score_with_beliefs(belief_override: list[torch.Tensor] | None = None):
        all_costs = []
        all_safe = []
        all_violation = []
        for track_index, track in enumerate(tracks):
            if belief_override is not None:
                belief_tensor = belief_override[track_index]
            else:
                belief = track.belief
                if belief is None:
                    belief_tensor = uniform_log_belief(model.config.num_modes, device=args.device).exp()
                else:
                    belief_tensor = torch.from_numpy(np.exp(belief)).to(args.device)
            mode_means = torch.from_numpy(means[track_index]).to(args.device)
            mode_stds = torch.from_numpy(log_stds[track_index]).to(args.device)
            mode_reference = torch.from_numpy(references[track_index]).to(args.device)
            track_current = torch.from_numpy(current[track_index]).to(args.device)
            track_candidates = torch.from_numpy(
                np.concatenate(
                    [
                        candidate_positions - track.position[None, None, None, :],
                        candidate_velocities,
                    ],
                    axis=-1,
                ).astype(np.float32)
            ).to(args.device)
            cost = expected_candidate_costs(
                mode_means,
                belief_tensor,
                track_current,
                track_candidates,
                mode_reference,
                model_dt,
                planning,
                mode_stds,
                task_direction=task_direction,
                task_active_mask=task_active_mask,
            )
            unconstrained = cost["unconstrained_total"]
            if executable_task is not None:
                unconstrained = unconstrained + planning.task_weight * (
                    executable_task.to(unconstrained) - cost["task"]
                )
            all_costs.append(unconstrained)
            all_safe.append(cost["safe_mask"])
            all_violation.append(cost["violation_probability"])
        total = torch.stack(all_costs).mean(dim=0)
        uniform_costs = []
        uniform = uniform_log_belief(model.config.num_modes, device=args.device).exp()
        for track_index, track in enumerate(tracks):
            mode_means = torch.from_numpy(means[track_index]).to(args.device)
            mode_stds = torch.from_numpy(log_stds[track_index]).to(args.device)
            mode_reference = torch.from_numpy(references[track_index]).to(args.device)
            track_current = torch.from_numpy(current[track_index]).to(args.device)
            track_candidates = torch.from_numpy(
                np.concatenate(
                    [
                        candidate_positions - track.position[None, None, None, :],
                        candidate_velocities,
                    ],
                    axis=-1,
                ).astype(np.float32)
            ).to(args.device)
            uniform_cost = expected_candidate_costs(
                    mode_means,
                    uniform,
                    track_current,
                    track_candidates,
                    mode_reference,
                    model_dt,
                    planning,
                    mode_stds,
                    task_direction=task_direction,
                    task_active_mask=task_active_mask,
                )
            uniform_unconstrained = uniform_cost["unconstrained_total"]
            if executable_task is not None:
                uniform_unconstrained = uniform_unconstrained + planning.task_weight * (
                    executable_task.to(uniform_unconstrained) - uniform_cost["task"]
                )
            uniform_costs.append(uniform_unconstrained)
        uniform_total = torch.stack(uniform_costs).mean(dim=0)
        leverage = float(getattr(args, "belief_cost_leverage", 1.0))
        total = total + belief_cost_adjustment(total, uniform_total, leverage)
        total = torch.from_numpy(stagnation_adjusted_costs(
            total.detach().cpu().numpy(),
            actions,
            stagnant_robots,
            getattr(args, "stagnation_halt_penalty", 0.0),
        )).to(total)
        safe = torch.stack(all_safe).all(dim=0)
        violation = torch.stack(all_violation).mean(dim=0)
        finite = torch.isfinite(total) & safe
        fallback = not bool(finite.any())
        if fallback:
            selected = int(torch.argmin(violation).item())
        else:
            selected = int(torch.argmin(torch.where(finite, total, torch.full_like(total, float("inf")))).item())
        if geometric_clearances is not None:
            selected, fallback = select_with_joint_geometry(
                total.detach().cpu().numpy(), safe.detach().cpu().numpy(),
                violation.detach().cpu().numpy(), geometric_clearances, actions, stagnant_robots,
                allow_hard_safe_approach=getattr(args, "stagnation_release_hard_safe", False),
                require_robust_robot_clearance=getattr(
                    args, "stagnation_release_require_robust_robot", False
                ),
                require_nonworsening_robot_clearance=getattr(
                    args, "stagnation_release_require_nonworsening_robot", False
                ),
                robot_clearance_floor=getattr(
                    args, "stagnation_release_robot_clearance_floor", None
                ),
                recovery_safety_first=getattr(args, "stagnation_recovery_safety_first", False),
            )
        return selected, total, safe, violation, fallback

    selected, total, safe, violation, fallback = score_with_beliefs()
    diagnostics = {
        "fallback": float(fallback),
        "safe_fraction": float(safe.float().mean().item()),
        "min_total_cost": _finite(total[selected].item()),
        "min_violation_probability": _finite(violation[selected].item()),
        "belief_selection_changed": 0.0,
        "belief_action_changed": 0.0,
        "belief_score_l1": 0.0,
        "belief_selected_regret": 0.0,
        "belief_cost_leverage": float(getattr(args, "belief_cost_leverage", 1.0)),
        "stagnant_robot_count": int(np.sum(stagnant_robots)),
    }
    if getattr(args, "belief_selection_audit", False):
        uniform_beliefs = [
            uniform_log_belief(model.config.num_modes, device=args.device).exp()
            for _ in tracks
        ]
        uniform_selected, uniform_total, uniform_safe, uniform_violation, uniform_fallback = score_with_beliefs(
            uniform_beliefs
        )
        diagnostics["belief_selection_changed"] = float(selected != uniform_selected)
        diagnostics["belief_action_changed"] = float(
            not np.array_equal(actions[selected], actions[uniform_selected])
        )
        diagnostics["belief_score_l1"] = float(torch.mean(torch.abs(total - uniform_total)).item())
        diagnostics["belief_selected_regret"] = float(
            (uniform_total[selected] - uniform_total[uniform_selected]).item()
        )
    return selected, means, log_stds, references, diagnostics


def _update_beliefs(
    tracks: list[HumanTrack],
    means: np.ndarray,
    log_stds: np.ndarray,
    selected: int,
    policy: str,
    args: argparse.Namespace,
    track_ids: list[int] | None = None,
) -> int:
    if not policy.startswith("tracer-") or policy == "tracer-uniform":
        return 0
    change_points = 0
    row_ids = track_ids if track_ids is not None else [track.track_id for track in tracks]
    tracks_by_id = {track.track_id: track for track in tracks}
    if means.ndim != 5 or means.shape != log_stds.shape:
        raise ValueError("belief emissions must align as (tracks, candidates, modes, steps, xy)")
    if len(row_ids) != means.shape[0] or len(set(row_ids)) != len(row_ids):
        raise ValueError("prediction track IDs must uniquely align with emission rows")
    if len(tracks_by_id) != len(tracks) or not 0 <= selected < means.shape[1]:
        raise ValueError("belief update needs unique current identities and a valid candidate")
    for track_index, track_id in enumerate(row_ids):
        track = tracks_by_id.get(track_id)
        if track is None:
            continue
        if not track.observed or track.observed_velocity is None:
            continue
        prior = (
            track.belief
            if track.belief is not None
            else uniform_log_belief(means.shape[2], device=args.device).cpu().numpy()
        )
        prediction = torch.from_numpy(means[track_index, selected, :, :1])[None]
        log_std = torch.from_numpy(log_stds[track_index, selected, :, :1])[None]
        target = torch.from_numpy(track.observed_velocity[None, None].astype(np.float32))
        likelihood = trajectory_mode_log_likelihood(prediction, log_std, target)[0]
        scaled = args.likelihood_scale * likelihood
        current = torch.from_numpy(prior)
        if policy == "tracer-reset":
            current = uniform_log_belief(current.numel())
        if policy == "tracer-hazard":
            posterior = hazard_belief_update(current, scaled, args.hazard_rate)
        elif policy == "tracer-forgetting":
            posterior = apply_belief_forgetting(
                categorical_bayes_update(current, scaled), args.forgetting_rate
            )
        else:
            changed = policy == "tracer-change-point" and bool(
                detect_change_point(current, scaled, args.change_point_threshold)
            )
            if changed:
                current = uniform_log_belief(current.numel())
                change_points += 1
            posterior = categorical_bayes_update(current, scaled)
        track.belief = posterior.detach().cpu().numpy().astype(np.float32)
    return change_points


def _entropy(tracks: list[HumanTrack], mode_count: int) -> float:
    values = []
    for track in tracks:
        belief = track.belief
        if belief is None:
            values.append(math.log(mode_count))
        else:
            probability = np.exp(belief)
            values.append(float(-(probability * belief).sum()))
    return _finite(np.mean(values)) if values else 0.0


def _candidate_geometry(
    robot_positions: np.ndarray,
    robot_velocities: np.ndarray,
    *candidate_args: Any,
    candidate_model: str | None = None,
    candidate_max_speed: float = 1.5,
    candidate_nominal_start_speed: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    # Keep the old private helper call usable for frozen adapter tests and
    # historical analysis scripts.  New runtime calls pass the observable
    # heading/rate/waypoint state explicitly below.
    if len(candidate_args) == 4:
        goals, action_lattice, horizon, dt = candidate_args
        robot_headings = np.zeros(len(robot_positions), dtype=np.float32)
        robot_angular_velocities = np.zeros(len(robot_positions), dtype=np.float32)
        local_targets = np.asarray(goals) - np.asarray(robot_positions)
        candidate_model = "goal-line-legacy"
    elif len(candidate_args) in (7, 8):
        (
            robot_headings,
            robot_angular_velocities,
            local_targets,
            goals,
            action_lattice,
            horizon,
            dt,
            *model_arg,
        ) = candidate_args
        if model_arg:
            candidate_model = model_arg[0]
        if candidate_model is None:
            candidate_model = "executable"
    else:
        raise TypeError("candidate geometry expects legacy or observable execution state")
    if candidate_model == "executable":
        return executable_candidate_prefix(
            robot_positions,
            robot_headings,
            robot_velocities,
            robot_angular_velocities,
            local_targets,
            goals,
            action_lattice,
            horizon,
            dt,
            max_speed=candidate_max_speed,
            nominal_start_speed=candidate_nominal_start_speed,
        )
    if candidate_model != "goal-line-legacy":
        raise ValueError(f"unsupported SocialGym2 candidate model: {candidate_model}")
    robots = len(robot_positions)
    direction = goals - robot_positions
    direction_norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    fallback = robot_velocities.copy()
    fallback_norm = np.linalg.norm(fallback, axis=-1, keepdims=True)
    fallback = fallback / np.maximum(fallback_norm, 1e-6)
    fallback = np.where(fallback_norm > 1e-4, fallback, np.asarray([[1.0, 0.0]], dtype=np.float32))
    direction = direction / np.maximum(direction_norm, 1e-6)
    direction = np.where(direction_norm > 1e-4, direction, fallback)
    measured_speed = np.linalg.norm(robot_velocities, axis=-1, keepdims=True)
    speed = np.clip(np.where(measured_speed > 0.05, measured_speed, 0.75), 0.25, 1.5)
    go_velocity = direction * speed
    candidate_velocities = np.zeros((len(action_lattice), robots, horizon, 2), dtype=np.float32)
    candidate_positions = np.zeros_like(candidate_velocities)
    for candidate_index, action in enumerate(action_lattice):
        velocity = go_velocity * (1 - action[:, None])
        candidate_velocities[candidate_index] = velocity[:, None, :]
        candidate_positions[candidate_index] = robot_positions[:, None, :] + np.cumsum(
            candidate_velocities[candidate_index] * float(dt), axis=1
        )
    return candidate_positions, candidate_velocities


def _executed_candidate_geometry(
    robot_positions: np.ndarray,
    next_robot_positions: np.ndarray,
    horizon: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    displacement = np.asarray(next_robot_positions, dtype=np.float32) - np.asarray(
        robot_positions, dtype=np.float32
    )
    velocity = displacement / max(float(dt), 1e-6)
    candidate_velocities = np.repeat(velocity[:, None, :], horizon, axis=1)
    candidate_positions = np.asarray(robot_positions, dtype=np.float32)[:, None, :] + np.cumsum(
        candidate_velocities * float(dt), axis=1
    )
    return candidate_positions, candidate_velocities


def _clone_tracks(tracks: list[HumanTrack]) -> list[HumanTrack]:
    return [
        HumanTrack(
            track_id=track.track_id,
            position=track.position.copy(),
            velocity=track.velocity.copy(),
            history=[item.copy() for item in track.history],
            belief=None if track.belief is None else track.belief.copy(),
            missed=track.missed,
            observed=track.observed,
            observed_velocity=(
                None if track.observed_velocity is None else track.observed_velocity.copy()
            ),
        )
        for track in tracks
    ]


def _run_episode(
    env: Any,
    capture: dict[str, Any],
    model: torch.nn.Module,
    adapter: SAGECandidateAdapter | None,
    policy: str,
    args: argparse.Namespace,
    episode: int,
    rng: np.random.Generator,
    trace_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    execution_diagnostics: ExecutionDiagnostics | None = None,
    geometry_segments: np.ndarray | None = None,
) -> dict[str, Any]:
    _assert_ros_runtime_health()
    episode_seed = _seed_episode(_episode_seed(args.seed, episode))
    env.new_scenario(num_humans=args.num_humans, num_agents=args.num_agents)
    scenario_hash = _generated_scenario_hash(Path(args.source).resolve())
    env.reset(seed=episode_seed, return_info=True)
    if not math.isclose(env.h3_simulation_time(), 0.0, abs_tol=1e-7):
        raise RuntimeError("SocialGym2 reset advanced simulator time")
    responses = capture.pop("responses")
    robot_positions, robot_velocities, goals, observations = _scene_from_responses(responses, args.partial_observation)
    initial_state_hash = _initial_state_hash(
        robot_positions, robot_velocities, goals, observations
    )
    initial_scene_snapshot = json.dumps({
        "robot_positions": robot_positions.tolist(),
        "robot_velocities": robot_velocities.tolist(),
        "goals": goals.tolist(),
        "human_observations": [
            {"position": position.tolist(), "velocity": velocity.tolist()}
            for position, velocity in observations
        ],
        "reported_collisions": [bool(response.collision) for response in responses],
    }, separators=(",", ":"))
    tracks: list[HumanTrack] = []
    next_id = 0
    next_id = _update_tracks(tracks, observations, model.config.history, args.env_dt, next_id, args.max_missed)
    if execution_diagnostics is not None:
        execution_diagnostics.begin_episode({
            "policy": policy, "seed": args.seed, "episode": episode,
            "episode_seed": episode_seed, "scenario": args.scenario,
            "scenario_hash": scenario_hash, "initial_state_hash": initial_state_hash,
        }, responses, tracks)
    action_lattice = _action_lattice(len(robot_positions))
    initial_distances = np.linalg.norm(goals - robot_positions, axis=-1)
    min_distance = float("inf")
    collisions = 0
    successes = 0
    successful_robots = np.zeros(len(robot_positions), dtype=bool)
    stationary_unfinished_seconds = np.zeros(len(robot_positions), dtype=np.float64)
    initial_collision = any(bool(response.collision) for response in responses)
    fallback_count = 0
    change_points = 0
    action_history: list[np.ndarray] = []
    total_rewards = 0.0
    robot_progress = 0.0
    planned_windows = 0
    active_entity_counts: list[int] = []
    belief_audit_windows = 0
    belief_selection_changes = 0
    belief_action_changes = 0
    belief_score_l1: list[float] = []
    belief_selected_regret: list[float] = []
    start_time = time.perf_counter()
    means = np.empty((len(tracks), len(action_lattice), model.config.num_modes, model.config.horizon, 2), dtype=np.float32)
    log_stds = np.zeros_like(means)
    final_done = False
    clock_max_abs_error = 0.0
    executed_sim_ticks = 0
    stagnation_windows = 0
    stagnation_go_actions = 0
    external_raw_action_counts = np.zeros(4, dtype=np.int64)

    for step in range(args.max_steps):
        active_entity_counts.append(sum(track.missed == 0 for track in tracks))
        stagnant_robots = stationary_unfinished_seconds >= args.stagnation_trigger_seconds
        before_robot = robot_positions.copy()
        before_robot_velocities = robot_velocities.copy()
        before_tracks = {track.track_id: track.position.copy() for track in tracks}
        decision_tracks = _clone_tracks(tracks)
        sage_activities = None
        if tracks and policy not in EXTERNAL_POLICIES:
            if adapter is None:
                raise RuntimeError("TRACER policy requires a loaded SAGE candidate adapter")
            proposal_track = max(tracks, key=lambda item: (item.observed, -item.missed))
            sage_pool = adapter.candidate_pool(
                _track_history(proposal_track, adapter.model.config.diffusion.horizon, args.env_dt),
                robot_positions,
                before_robot_velocities,
                args.env_dt,
                args.candidate_seed + args.seed * 1009 + episode * 9176 + step,
                robot_goals=goals,
            )
            sage_activities = sage_pool.activities.numpy()
        decision_lattice = action_lattice
        if getattr(args, "completed_robot_halt", False) and policy.startswith("tracer-"):
            decision_lattice = _completed_robot_action_lattice(action_lattice, successful_robots)
        ordering = _project_sage_order(sage_activities, decision_lattice)
        ordered_actions = decision_lattice[ordering]
        robot_headings, robot_angular_velocities, local_targets = _navigation_state(responses)
        task_directions = observable_route_directions(
            robot_positions, robot_headings, local_targets, goals,
        ).astype(np.float32)
        candidate_positions, candidate_velocities = _candidate_geometry(
            robot_positions,
            robot_velocities,
            robot_headings,
            robot_angular_velocities,
            local_targets,
            goals,
            ordered_actions,
            model.config.horizon,
            args.model_dt,
            args.candidate_prefix_model,
            candidate_max_speed=args.candidate_max_speed,
            candidate_nominal_start_speed=args.candidate_nominal_start_speed,
        )
        geometric_clearances = None
        if geometry_segments is not None:
            observed_poses = np.asarray([np.asarray(response.robot_poses)[:3] for response in responses])
            observed_twists = np.asarray([np.asarray(response.robot_vels)[:3] for response in responses])
            short_prefix, _ = executable_candidate_prefix(
                robot_positions,
                robot_headings,
                robot_velocities,
                robot_angular_velocities,
                local_targets,
                goals,
                ordered_actions,
                args.sim_steps_per_window,
                args.sim_dt,
                max_speed=args.candidate_max_speed,
                nominal_start_speed=args.candidate_nominal_start_speed,
                measured_prefix_duration=args.env_dt,
            )
            geometric_clearances = joint_prefix_clearances(
                robot_positions,
                short_prefix,
                geometry_segments,
                robot_separation=0.4 + args.execution_position_margin,
                static_radius=0.22 + args.execution_position_margin,
                hard_robot_separation=0.4,
                hard_static_radius=0.22,
            )
            if getattr(args, "observed_human_geometry_filter", True):
                observed_tracks = [track for track in decision_tracks if track.observed]
                geometric_clearances = add_observed_human_clearances(
                    geometric_clearances,
                    robot_positions,
                    short_prefix,
                    [track.position for track in observed_tracks],
                    [track.velocity for track in observed_tracks],
                    args.sim_dt,
                    robot_human_separation=0.4 + args.execution_position_margin,
                    hard_robot_human_separation=0.4,
                )

        external_raw_actions = []
        if policy in EXTERNAL_POLICIES:
            # PIPS is a native SocialGym2 controller. Preserve its four-action
            # output so Follow/Pass are evaluated as actual external behavior.
            selected_action = np.empty(len(robot_positions), dtype=np.int64)
            for robot_index in range(len(robot_positions)):
                raw_action = _query_external_socialgym_policy(
                    env, responses, robot_index
                )
                external_raw_action_counts[raw_action] += 1
                external_raw_actions.append(raw_action)
                selected_action[robot_index] = raw_action
            selected = -1
            means = np.empty((len(tracks), len(ordered_actions), model.config.num_modes, model.config.horizon, 2), dtype=np.float32)
            log_stds = np.zeros_like(means)
            diagnostics = {
                "fallback": 0.0,
                "safe_fraction": float("nan"),
                "min_total_cost": float("nan"),
                "min_violation_probability": float("nan"),
                "belief_selection_changed": 0.0,
                "belief_action_changed": 0.0,
                "belief_score_l1": 0.0,
                "belief_selected_regret": 0.0,
                "stagnant_robot_count": int(np.sum(stagnant_robots)),
            }
        elif policy.startswith("official-"):
            if policy == "official-goalone":
                selected = int(np.flatnonzero(np.all(ordered_actions == 0, axis=1))[0])
            elif policy == "official-halt":
                selected = int(np.flatnonzero(np.all(ordered_actions == 1, axis=1))[0])
            else:
                selected = int(rng.integers(0, len(ordered_actions)))
            means = np.empty((len(tracks), len(ordered_actions), model.config.num_modes, model.config.horizon, 2), dtype=np.float32)
            log_stds = np.zeros_like(means)
            diagnostics = {
                "fallback": 0.0,
                "safe_fraction": float("nan"),
                "min_total_cost": float("nan"),
                "min_violation_probability": float("nan"),
                "belief_selection_changed": 0.0,
                "belief_action_changed": 0.0,
                "belief_score_l1": 0.0,
                "belief_selected_regret": 0.0,
                "stagnant_robot_count": int(np.sum(stagnant_robots)),
            }
        else:
            task_active_robots = (
                ~successful_robots
                & (np.linalg.norm(goals - robot_positions, axis=-1) > 0.5)
            )
            selected, means, log_stds, _, diagnostics = _select_action(
                policy,
                model,
                tracks,
                robot_positions,
                robot_velocities,
                goals,
                ordered_actions,
                candidate_positions,
                candidate_velocities,
                model.config.history,
                args.model_dt,
                args,
                geometric_clearances,
                task_directions=task_directions,
                stagnant_robots=stagnant_robots,
                task_active_robots=task_active_robots,
            )
        if geometric_clearances is not None:
            recovery_mask = stagnation_recovery_mask(
                geometric_clearances,
                ordered_actions,
                stagnant_robots,
                allow_hard_safe_approach=getattr(args, "stagnation_release_hard_safe", False),
                require_robust_robot_clearance=getattr(
                    args, "stagnation_release_require_robust_robot", False
                ),
                require_nonworsening_robot_clearance=getattr(
                    args, "stagnation_release_require_nonworsening_robot", False
                ),
                robot_clearance_floor=getattr(
                    args, "stagnation_release_robot_clearance_floor", None
                ),
            )
            selected_robot_clearance = float(geometric_clearances["robot_clearance"][selected])
            selected_static_clearance = float(geometric_clearances["static_clearance"][selected])
            diagnostics["joint_geometry"] = {
                "feasible_action_count": int(np.sum(geometric_clearances["feasible"])),
                "selected_feasible": bool(geometric_clearances["feasible"][selected]),
                "selected_robot_clearance": selected_robot_clearance if math.isfinite(selected_robot_clearance) else None,
                "selected_static_clearance": selected_static_clearance if math.isfinite(selected_static_clearance) else None,
                "selected_overlap_recovery": bool(geometric_clearances["recovering_initial_overlap"][selected]),
                "all_halt_feasible": bool(geometric_clearances["feasible"][np.flatnonzero(np.all(ordered_actions == 1, axis=1))[0]]),
                "all_halt_hard_feasible": bool(geometric_clearances["hard_feasible"][np.flatnonzero(np.all(ordered_actions == 1, axis=1))[0]]),
                "stagnation_recovery_action_count": int(np.sum(recovery_mask)),
                "selected_stagnation_recovery": bool(recovery_mask[selected]),
                "stagnation_release_hard_safe": bool(
                    getattr(args, "stagnation_release_hard_safe", False)
                ),
                "stagnation_release_require_robust_robot": bool(
                    getattr(args, "stagnation_release_require_robust_robot", False)
                ),
                "stagnation_release_require_nonworsening_robot": bool(
                    getattr(args, "stagnation_release_require_nonworsening_robot", False)
                ),
                "stagnation_release_robot_clearance_floor": getattr(
                    args, "stagnation_release_robot_clearance_floor", None
                ),
                "stagnation_recovery_safety_first": bool(
                    getattr(args, "stagnation_recovery_safety_first", False)
                ),
            }
            if "human_clearance" in geometric_clearances:
                selected_human_clearance = float(geometric_clearances["human_clearance"][selected])
                diagnostics["joint_geometry"].update({
                    "selected_human_clearance": selected_human_clearance if math.isfinite(selected_human_clearance) else None,
                    "selected_human_feasible": bool(geometric_clearances["human_feasible"][selected]),
                })
        selected_action = _executed_socialgym_action(
            policy,
            selected_action if policy in EXTERNAL_POLICIES else None,
            ordered_actions,
            selected,
        )
        stagnation_windows += int(stagnant_robots.any())
        stagnation_go_actions += int(np.sum(stagnant_robots & (selected_action == 0)))
        action_history.append(selected_action.copy())
        fallback_count += int(diagnostics["fallback"])
        if getattr(args, "belief_selection_audit", False) and policy not in EXTERNAL_POLICIES and not policy.startswith("official-"):
            belief_audit_windows += 1
            belief_selection_changes += int(diagnostics["belief_selection_changed"])
            belief_action_changes += int(diagnostics["belief_action_changed"])
            belief_score_l1.append(float(diagnostics["belief_score_l1"]))
            belief_selected_regret.append(float(diagnostics["belief_selected_regret"]))
        planned_windows += 1

        action_dict = {
            agent: int(selected_action[index])
            for index, agent in enumerate(env.agents)
        }
        window_rewards = []
        window_terminations = {}
        window_truncations = {}
        window_responses = []
        before_sim_time = env.h3_simulation_time()
        evidence_start = len(evidence_rows)
        if execution_diagnostics is not None:
            execution_diagnostics.begin_window(
                step, before_sim_time, candidate_positions[selected], candidate_velocities[selected],
                selected_action, decision_tracks, diagnostics,
            )
        for _ in range(args.sim_steps_per_window):
            _, rewards, terminations, truncations, infos = env.step(action_dict)
            _assert_ros_runtime_health()
            responses = capture.pop("responses")
            window_rewards.append(rewards)
            window_terminations.update(terminations)
            window_truncations.update(truncations)
            window_responses.append(responses)
            if execution_diagnostics is not None:
                execution_diagnostics.record_tick(env.h3_simulation_time(), responses)
            if responses and all(bool(response.done or response.success) for response in responses):
                break
        if not window_responses:
            raise RuntimeError("SocialGym2 returned no response during an execution window")
        elapsed_sim_time = env.h3_simulation_time() - before_sim_time
        expected_sim_time = len(window_responses) * args.sim_dt
        clock_error = abs(elapsed_sim_time - expected_sim_time)
        clock_max_abs_error = max(clock_max_abs_error, clock_error)
        executed_sim_ticks += len(window_responses)
        if clock_error > 2e-5:
            raise RuntimeError(f"SocialGym2 clock drift: observed={elapsed_sim_time}, expected={expected_sim_time}")
        responses = window_responses[-1]
        robot_positions, robot_velocities, goals, observations = _scene_from_responses(responses, args.partial_observation)
        for rewards in window_rewards:
            total_rewards += float(np.mean([float(value) for value in rewards.values()])) if rewards else 0.0
        robot_progress += float(np.mean(np.linalg.norm(goals - before_robot, axis=-1) - np.linalg.norm(goals - robot_positions, axis=-1)))
        collisions += sum(
            int(bool(response.collision))
            for response_batch in window_responses
            for response in response_batch
        )
        successes += sum(
            int(bool(response.success))
            for response_batch in window_responses
            for response in response_batch
        )
        for response_batch in window_responses:
            successful_robots |= np.asarray([bool(response.success) for response in response_batch])
        unfinished = (~successful_robots) & (np.linalg.norm(goals - robot_positions, axis=-1) > 0.5)
        stationary = np.linalg.norm(robot_positions - before_robot, axis=-1) < args.stagnation_motion_threshold
        stationary_unfinished_seconds = np.where(
            unfinished & stationary,
            stationary_unfinished_seconds + elapsed_sim_time,
            0.0,
        )
        next_id = _update_tracks(tracks, observations, model.config.history, elapsed_sim_time, next_id, args.max_missed)
        for track in tracks:
            if track.observed and track.track_id in before_tracks:
                measured = (track.position - before_tracks[track.track_id]) / max(args.env_dt, 1e-6)
                if track.observed_velocity is None or not np.isfinite(track.observed_velocity).all():
                    track.observed_velocity = measured.astype(np.float32)
        if policy not in EXTERNAL_POLICIES and not policy.startswith("official-") and len(tracks) and len(window_responses) == args.sim_steps_per_window:
            executed_positions, executed_velocities = _executed_candidate_geometry(
                before_robot,
                robot_positions,
                model.config.horizon,
                args.env_dt,
            )
            executed_means, executed_log_stds, _, _ = _response_predictions(
                model,
                decision_tracks,
                before_robot,
                before_robot_velocities,
                executed_positions[None],
                executed_velocities[None],
                model.config.history,
                args.env_dt,
                args.device,
                include_pair_interactions=policy != "tracer-no-pair",
            )
            decision_track_ids = [track.track_id for track in decision_tracks]
            tracks_by_id = {track.track_id: track for track in tracks}
            change_points += _update_beliefs(
                tracks,
                executed_means,
                executed_log_stds,
                0,
                policy,
                args,
                track_ids=decision_track_ids,
            )
            for track_index, track_id in enumerate(decision_track_ids):
                track = tracks_by_id.get(track_id)
                if track is None:
                    continue
                if not track.observed or track.observed_velocity is None:
                    continue
                target = torch.from_numpy(track.observed_velocity[None, None].astype(np.float32))
                likelihood = trajectory_mode_log_likelihood(
                    torch.from_numpy(executed_means[track_index, 0, :, :1]).to(args.device)[None],
                    torch.from_numpy(executed_log_stds[track_index, 0, :, :1]).to(args.device)[None],
                    target.to(args.device),
                )[0].detach().cpu().numpy()
                prior = (
                    uniform_log_belief(model.config.num_modes, device=args.device)
                    .cpu()
                    .numpy()
                    if decision_tracks[track_index].belief is None
                    else decision_tracks[track_index].belief.copy()
                )
                evidence_rows.append(
                    {
                        "policy": policy,
                        "seed": args.seed,
                        "episode_seed": episode_seed,
                        "scenario": args.scenario,
                        "scenario_hash": scenario_hash,
                        "initial_state_hash": initial_state_hash,
                        "episode": episode,
                        "step": step,
                        "track_id": track.track_id,
                        "sim_steps_executed": len(window_responses),
                        "decision_log_belief": json.dumps(prior.tolist(), separators=(",", ":")),
                        "factual_mode_log_likelihood": json.dumps(likelihood.tolist(), separators=(",", ":")),
                        "posterior_log_belief": json.dumps(
                            (
                                track.belief
                                if track.belief is not None
                                else uniform_log_belief(model.config.num_modes, device=args.device)
                                .cpu()
                                .numpy()
                            ).tolist(),
                            separators=(",", ":"),
                        ),
                    }
                )
        if execution_diagnostics is not None:
            execution_diagnostics.end_window(
                before_sim_time + elapsed_sim_time, tracks, evidence_rows[evidence_start:],
            )
        for track in tracks:
            min_distance = min(min_distance, float(np.min(np.linalg.norm(robot_positions - track.position[None], axis=-1))))
        observed_tracks = sum(int(track.observed) for track in tracks)
        for track in tracks:
            belief = track.belief
            if belief is None:
                belief = np.full(model.config.num_modes, -math.log(model.config.num_modes), dtype=np.float32)
            trace_rows.append(
                {
                    "policy": policy,
                    "seed": args.seed,
                    "episode": episode,
                    "step": step,
                    "track_id": track.track_id,
                    "observed": int(track.observed),
                    "missed": track.missed,
                    "active_tracks": len(tracks),
                    "observed_tracks": observed_tracks,
                    "selected_candidate": selected,
                    "action_vector": json.dumps(selected_action.tolist(), separators=(",", ":")),
                    "belief_entropy": _entropy(tracks, model.config.num_modes),
                    "belief_mode_0": float(np.exp(belief[0])) if len(belief) > 0 else 0.0,
                    "belief_mode_1": float(np.exp(belief[1])) if len(belief) > 1 else 0.0,
                    "belief_mode_2": float(np.exp(belief[2])) if len(belief) > 2 else 0.0,
                    "min_distance": _finite(min_distance, 0.0),
                    "fallback": float(diagnostics["fallback"]),
                    "belief_selection_changed": float(diagnostics["belief_selection_changed"]),
                    "belief_action_changed": float(diagnostics["belief_action_changed"]),
                    "belief_score_l1": float(diagnostics["belief_score_l1"]),
                    "belief_selected_regret": float(diagnostics["belief_selected_regret"]),
                    "stagnant_robot_count": int(np.sum(stagnant_robots)),
                    "external_raw_action": (
                        "" if not external_raw_actions else json.dumps(external_raw_actions)
                    ),
                }
            )
        final_done = bool(responses) and all(bool(response.done or response.success) for response in responses)
        if final_done or all(
            bool(window_terminations.get(agent, False) or window_truncations.get(agent, False))
            for agent in env.agents
        ):
            break

    posterior_entropy = _entropy(tracks, model.config.num_modes)
    action_changes = float(np.mean(np.any(np.diff(np.asarray(action_history), axis=0) != 0, axis=1))) if len(action_history) > 1 else 0.0
    return {
        "policy": policy,
        "seed": args.seed,
        "episode_seed": episode_seed,
        "scenario": args.scenario,
        "scenario_hash": scenario_hash,
        "initial_state_hash": initial_state_hash,
        "initial_scene_snapshot": initial_scene_snapshot,
        "episode": episode,
        "steps": len(action_history),
        "executed_sim_ticks": executed_sim_ticks,
        "last_window_sim_ticks": len(window_responses),
        "terminal_reached": bool(final_done),
        "clock_max_abs_error": clock_max_abs_error,
        "num_agents": len(robot_positions),
        "num_tracks_final": len(tracks),
        "success": float(successes > 0),
        "team_success": float(successful_robots.all()),
        "robot_success_fraction": float(successful_robots.mean()),
        "collision_free_team_success": float(successful_robots.all() and collisions == 0 and not initial_collision),
        "initial_collision": float(initial_collision),
        "collision": float(collisions > 0),
        "collision_events": int(collisions),
        "task_progress": _finite(robot_progress / max(float(np.mean(np.maximum(initial_distances, 1e-6))), 1e-6)),
        "mean_reward": _finite(total_rewards / max(len(action_history), 1)),
        "min_distance": _finite(min_distance, 0.0),
        "fallback_rate": _finite(fallback_count / max(len(action_history), 1)),
        "stagnation_window_rate": _finite(stagnation_windows / max(len(action_history), 1)),
        "stagnation_go_actions": int(stagnation_go_actions),
        "external_raw_action_counts": json.dumps(external_raw_action_counts.tolist()),
        "final_belief_entropy": posterior_entropy,
        "action_change_rate": action_changes,
        "change_point_events": int(change_points),
        "mean_active_entities": _finite(float(np.mean(active_entity_counts))) if active_entity_counts else 0.0,
        "belief_selection_audit_windows": int(belief_audit_windows),
        "belief_selection_change_rate": _finite(
            belief_selection_changes / max(belief_audit_windows, 1)
        ),
        "belief_action_change_rate": _finite(
            belief_action_changes / max(belief_audit_windows, 1)
        ),
        "belief_score_l1": _finite(float(np.mean(belief_score_l1))) if belief_score_l1 else 0.0,
        "belief_selected_regret": _finite(float(np.mean(belief_selected_regret))) if belief_selected_regret else 0.0,
        "sage_proposal_projection": float(policy not in EXTERNAL_POLICIES),
        "official_discrete_action_space": 1.0,
        "wall_time_seconds": time.perf_counter() - start_time,
    }


def _bootstrap(values: list[float], seed: int, samples: int = 4000) -> dict[str, Any]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "count": 0}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(samples, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "count": int(len(array)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.episodes < 1 or args.max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    root = Path.cwd().resolve()
    source = (root / args.source).resolve()
    launch_config = (source / "config/gym_gen/launch.launch").resolve()
    output = (root / args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite SocialGym2 experiment: {output}")
    output.mkdir(parents=True)
    external_policies = [policy for policy in args.policies if policy in EXTERNAL_POLICIES]
    if len(external_policies) > 1:
        raise ValueError("one SocialGym2 run can launch only one official external policy service")
    external_policy = external_policies[0] if external_policies else None
    if external_policy and (args.execution_diagnostics or args.joint_geometry_filter):
        raise ValueError(
            "native SocialGym2 external runs cannot use binary-only geometry or "
            "execution-prefix diagnostics"
        )
    external_source_hashes = _external_source_hashes(source, external_policy)
    external_runtime_controller = _external_runtime_controller(source, external_policy)
    git = _git_state(root)
    runtime_path = Path(args.runtime_manifest).resolve() if args.runtime_manifest else None
    runtime_contract = _runtime_manifest(runtime_path, source)
    algorithm_sources = _algorithm_sources(root)
    checkpoint_hashes = {
        "response": _sha256((root / args.response_checkpoint).resolve()),
        "sage": _sha256((root / args.sage_checkpoint).resolve()),
    }
    if args.sim_dt <= 0.0:
        raise ValueError("sim_dt must be positive")
    if args.execution_position_margin < 0 or not math.isfinite(args.execution_position_margin):
        raise ValueError("execution_position_margin must be finite and nonnegative")
    if (
        args.candidate_max_speed <= 0.0
        or args.candidate_nominal_start_speed <= 0.0
        or args.candidate_nominal_start_speed > args.candidate_max_speed
        or not math.isfinite(args.candidate_max_speed)
        or not math.isfinite(args.candidate_nominal_start_speed)
    ):
        raise ValueError("candidate speeds must be finite and satisfy 0 < nominal <= max")
    if (
        args.stagnation_trigger_seconds <= 0
        or args.stagnation_motion_threshold < 0
        or args.stagnation_halt_penalty < 0
        or not all(math.isfinite(value) for value in (
            args.stagnation_trigger_seconds,
            args.stagnation_motion_threshold,
            args.stagnation_halt_penalty,
        ))
    ):
        raise ValueError("stagnation recovery parameters must be finite and nonnegative with a positive trigger")
    leverage = float(getattr(args, "belief_cost_leverage", 1.0))
    if not math.isfinite(leverage) or leverage <= 0.0:
        raise ValueError("belief-cost-leverage must be finite and positive")
    if args.sim_steps_per_window < 1:
        raise ValueError("sim_steps_per_window must be positive")
    if not math.isclose(
        float(args.sim_dt) * int(args.sim_steps_per_window),
        float(args.env_dt),
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise ValueError(
            "sim_dt * sim_steps_per_window must equal env_dt so each planning "
            "window matches the response-model clock"
        )
    if not math.isclose(float(args.model_dt), float(args.env_dt), rel_tol=0.0, abs_tol=1e-7):
        raise ValueError(
            "model_dt and env_dt must match: response scoring and ROS execution must use one clock"
        )
    model = _load_model(
        (root / args.response_checkpoint).resolve(),
        args.device,
        expected_dt=args.env_dt,
    )
    if not 1 <= args.task_steps <= int(model.config.horizon):
        raise ValueError(
            f"task_steps must lie within the response horizon: {args.task_steps} vs {model.config.horizon}"
        )
    response_cost_steps = getattr(args, "response_cost_steps", None)
    if response_cost_steps is not None and not 1 <= response_cost_steps <= int(model.config.horizon):
        raise ValueError("response_cost_steps must lie within the response horizon")
    checkpoint_path = (root / args.response_checkpoint).resolve()
    calibration_artifact = None
    if args.calibration_artifact:
        calibration_path = (root / args.calibration_artifact).resolve()
        calibration_artifact = _validate_calibration_artifact(
            calibration_path,
            checkpoint_path,
            args,
        )
        if runtime_contract is None or calibration_artifact.get("runtime_contract") != runtime_contract:
            raise ValueError("calibration runtime differs from the evaluated frozen runtime")
        if calibration_artifact.get("planning_horizon_windows") != args.max_steps:
            raise ValueError("calibration and evaluation planning horizons differ")
    if args.num_agents < 2:
        raise ValueError("joint external H3 requires at least two controllable robots")
    if int(model.config.num_robots) != int(args.num_agents):
        raise ValueError(
            "joint external H3 requires the response checkpoint robot count to match "
            f"the SocialGym2 agent count: checkpoint={model.config.num_robots}, agents={args.num_agents}"
        )
    adapter = None
    if external_policy is None:
        adapter = SAGECandidateAdapter.from_checkpoint(
            (root / args.sage_checkpoint).resolve(),
            args.device,
            SAGECandidateAdapterConfig(
                num_candidates=args.sage_candidates,
                guided=True,
                step_stride=args.sage_step_stride,
                goal_speed=args.sage_goal_speed,
                batch_size=args.sage_candidates,
                guidance_step_clip_norm=0.1,
                include_task_anchor=True,
                task_anchor_speed_scales=(0.45,),
                goal_direction="task_goal",
            ),
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    env = _build_env(
        source,
        args.num_agents,
        args.num_humans,
        args.partial_observation,
        launch_config,
        args.scenario,
        external_policy,
    )
    capture = _capture_observations(env)
    records: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    execution_diagnostics = None
    geometry_segments = None
    geometry_map_path = None
    geometry_map_hash = None
    if getattr(args, "joint_geometry_filter", False):
        geometry_map_path = source / "submodules/ut_multirobot_sim/maps" / args.scenario / f"{Path(args.scenario).name}.vectormap.txt"
        geometry_map_hash = _sha256(geometry_map_path)
        geometry_segments = load_vector_map(geometry_map_path)
    if getattr(args, "execution_diagnostics", False):
        map_path = source / "submodules/ut_multirobot_sim/maps" / args.scenario / f"{Path(args.scenario).name}.vectormap.txt"
        execution_diagnostics = ExecutionDiagnostics(output, map_path, args.model_dt, args.sim_dt, args.partial_observation)
    execution_completed = False
    started = time.perf_counter()
    try:
        rng = np.random.default_rng(args.seed + 701)
        for episode in range(args.episodes):
            for policy in args.policies:
                print(f"socialgym2_tracer seed={args.seed} episode={episode} policy={policy}", flush=True)
                records.append(
                    _run_episode(
                        env,
                        capture,
                        model,
                        adapter,
                        policy,
                        args,
                        episode,
                        rng,
                        trace_rows,
                        evidence_rows,
                        execution_diagnostics,
                        geometry_segments,
                    )
                )
                if execution_diagnostics is not None:
                    execution_diagnostics.end_episode(records[-1])
        execution_completed = True
    finally:
        try:
            env.close()
        finally:
            env.launch.shutdown()
            if execution_diagnostics is not None:
                execution_diagnostics.close(execution_completed)

    for relative, expected in algorithm_sources.items():
        if _sha256(root / relative) != expected:
            raise RuntimeError(f"algorithm source changed during experiment: {relative}")
    if geometry_map_path is not None and _sha256(geometry_map_path) != geometry_map_hash:
        raise RuntimeError("deployed geometry map changed during experiment")
    if _runtime_manifest(runtime_path, source) != runtime_contract:
        raise RuntimeError("external runtime changed during experiment")
    for name, filename in (("response", args.response_checkpoint), ("sage", args.sage_checkpoint)):
        if _sha256((root / filename).resolve()) != checkpoint_hashes[name]:
            raise RuntimeError(f"{name} checkpoint changed during experiment")
    if external_source_hashes and _external_source_hashes(source, external_policy) != external_source_hashes:
        raise RuntimeError("official SocialGym2 external controller source changed during experiment")
    if (
        external_runtime_controller
        and _external_runtime_controller(source, external_policy) != external_runtime_controller
    ):
        raise RuntimeError("official SocialGym2 external controller binary changed during experiment")

    fields = list(records[0]) if records else []
    metrics_path = output / "episode_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    trace_path = output / "belief_trace.csv"
    trace_fields = [
        "policy",
        "seed",
        "episode",
        "step",
        "track_id",
        "observed",
        "missed",
        "active_tracks",
        "observed_tracks",
        "selected_candidate",
        "action_vector",
        "belief_entropy",
        "belief_mode_0",
        "belief_mode_1",
        "belief_mode_2",
        "min_distance",
        "fallback",
        "belief_selection_changed",
        "belief_action_changed",
        "belief_score_l1",
        "belief_selected_regret",
        "stagnant_robot_count",
        "external_raw_action",
    ]
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trace_fields)
        writer.writeheader()
        writer.writerows(trace_rows)
    evidence_path = output / "belief_evidence.csv"
    evidence_fields = [
        "policy",
        "seed",
        "episode_seed",
        "scenario",
        "scenario_hash",
        "initial_state_hash",
        "episode",
        "step",
        "track_id",
        "sim_steps_executed",
        "decision_log_belief",
        "factual_mode_log_likelihood",
        "posterior_log_belief",
    ]
    with evidence_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=evidence_fields)
        writer.writeheader()
        writer.writerows(evidence_rows)
    metrics = (
        "success",
        "team_success",
        "robot_success_fraction",
        "collision_free_team_success",
        "initial_collision",
        "collision",
        "task_progress",
        "mean_reward",
        "min_distance",
        "fallback_rate",
        "stagnation_window_rate",
        "stagnation_go_actions",
        "final_belief_entropy",
        "action_change_rate",
        "belief_selection_audit_windows",
        "belief_selection_change_rate",
        "belief_action_change_rate",
        "belief_score_l1",
        "belief_selected_regret",
    )
    summary = {"policies": {}}
    for index, policy in enumerate(args.policies):
        selected = [row for row in records if row["policy"] == policy]
        summary["policies"][policy] = {
            metric: _bootstrap([float(row[metric]) for row in selected], args.bootstrap_seed + index * 31 + metric_index)
            for metric_index, metric in enumerate(metrics)
        }
    persistent = {(row["seed"], row["episode"]): row for row in records if row["policy"] == "tracer-persistent"}
    summary["paired_contrasts"] = {}
    for policy in args.policies:
        if policy == "tracer-persistent":
            continue
        compared = {(row["seed"], row["episode"]): row for row in records if row["policy"] == policy}
        keys = persistent.keys() & compared.keys()
        summary["paired_contrasts"][f"{policy}-persistent:task_progress"] = _bootstrap(
            [float(compared[key]["task_progress"] - persistent[key]["task_progress"]) for key in keys],
            args.bootstrap_seed + 500 + len(summary["paired_contrasts"]),
        )
    source_commit = os.environ.get("SOCIALGYM_SOURCE_COMMIT", "unavailable")
    try:
        source_label = str(source.relative_to(root))
    except ValueError:
        source_label = str(source)
    counterfactual_enabled = bool(getattr(args, "belief_selection_audit", False))
    execution_roles = socialgym2_policy_roles(args.policies, counterfactual_enabled)
    claim_boundary = (
            "Official SocialGym2/PIPS external controller executed in the same patched "
            "SocialGym2 ROS runtime with its native four-action GoAlone/Halt/Follow/Pass "
            "interface; this is a same-environment external-controller comparison, not a "
            "continuous-action SICNav comparison"
        if external_policy
        else "TRACER response-belief-planning executed in official SocialGym2 ROS with official simulator/human dynamics; action space is projected to GoAlone/Halt; executed robot-prefix likelihoods use the observed simulator transition over a model-clock planning window; behavior-response/navigation integration evidence only, not real-human causality or continuous-H3 equivalence"
    )
    metadata = {
        "protocol": "tracer-socialgym2-joint-belief-synchronous-v5",
        "joint_geometry_filter": {
            "enabled": geometry_segments is not None,
            "map_sha256": geometry_map_hash,
            "robot_separation": 0.4,
            "static_radius": 0.22,
            "execution_position_margin": args.execution_position_margin,
            "prediction": "same observable execution-prefix kinematics as response candidates, evaluated at simulator tick resolution; runtime Halt immediately zeros velocity",
            "boundary": "Additional geometric screen, not a safety guarantee or calibrated unseen acceleration envelope; response predictor and executed evidence semantics unchanged.",
        },
        "observed_human_geometry_filter": {
            "enabled": bool(geometry_segments is not None and getattr(args, "observed_human_geometry_filter", True)),
            "robust_separation": 0.4 + args.execution_position_margin,
            "hard_separation": 0.4,
            "prediction": "constant current world velocity over the next executed planning window for currently observed tracks only",
            "boundary": "Observable near-field swept guard complementing learned response risk; no hidden human state and no guarantee beyond one execution window.",
        },
        "candidate_prefix_model": {
            "name": args.candidate_prefix_model,
            "executable": args.candidate_prefix_model == "executable",
            "max_speed": args.candidate_max_speed,
            "nominal_start_speed": args.candidate_nominal_start_speed,
            "first_model_step": "measured current twist for GoAlone and immediate zero velocity for Halt",
            "later_model_steps": "observable local graph-navigation target with bounded linear/angular acceleration",
            "boundary": "Later prefix is an observable kinematic extrapolation, not hidden simulator rollout or a safety guarantee.",
        },
        "claim_boundary": claim_boundary,
        "git": git,
        "algorithm_sources": algorithm_sources,
        "runtime_contract": runtime_contract,
        "checkpoint_hashes": checkpoint_hashes,
        "external_repository": {"path": source_label, "commit": source_commit},
        "external_controller": {
            "policy": external_policy,
            "source": "SocialGym2 official submodules/pips",
            "source_hashes": external_source_hashes,
            "runtime": external_runtime_controller,
            "service_names": (
                [f"/tracer_external_{index}/SocialPipsSrv" for index in range(args.num_agents)]
                if external_policy
                else []
            ),
            "raw_action_names": EXTERNAL_ACTION_NAMES,
            "raw_to_executed_action": {str(action): action for action in EXTERNAL_ACTION_NAMES},
            "binary_projection_for_legacy_audit": EXTERNAL_ACTION_TO_BINARY,
            "mapping": (
                "Native SocialGym2 actions are executed unchanged: GoAlone=0, Halt=1, "
                "Follow=2, Pass=3"
                if external_policy
                else None
            ),
            "comparison_type": "same-environment external controller" if external_policy else None,
        },
        "device": args.device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "parameters": vars(args),
        "tracked_entity_identity": "greedy spatial association over official ID-free observations, with missed-track grace",
        "official_action_encoding": (
            {str(action): name for action, name in EXTERNAL_ACTION_NAMES.items()}
            if external_policy
            else {"0": "GoAlone", "1": "Halt"}
        ),
        "metric_contract": {
            "success": "at least one robot reaches its goal (legacy, not team success)",
            "team_success": "every robot reaches its goal during the episode",
            "collision_free_team_success": "team success without initial or subsequent observed collision",
            "initial_collision": "collision reported by the official reset observation; retained, not silently excluded",
        },
        "sage_candidate_adapter": (
            "not used for native external-controller action selection"
            if external_policy
            else "continuous SAGE joint proposals used to order the exhaustive official binary action lattice"
        ),
        "joint_response_contract": {
            "controllable_robots": int(args.num_agents),
            "response_checkpoint_robots": int(model.config.num_robots),
            "pair_interactions": True,
            "pair_interactions_by_policy": {
                policy: role["pair_interactions"]
                for policy, role in execution_roles.items()
            },
            "no_pair_ablation_policies": [policy for policy in args.policies if policy == "tracer-no-pair"],
            "candidate_space": (
                (
                    f"joint binary lattice of size at most {2 ** int(args.num_agents)}; TRACER fixes completed robots to Halt"
                    if getattr(args, "completed_robot_halt", False)
                    else f"joint binary lattice of size {2 ** int(args.num_agents)}"
                )
                if not external_policy
                else "native external policy output; TRACER binary candidate lattice is not executed"
            ),
            "belief_state": "factorized entity-indexed categorical posteriors whose product is scored over every joint robot candidate",
            "belief_update": "each applicable TRACER policy updates observed human posteriors from the executed joint candidate response prefix",
            "all_humans_scored_on_same_joint_candidate": True,
            "same_state_uniform_counterfactual": counterfactual_enabled,
            "policy_role_schema": POLICY_ROLE_SCHEMA,
            "policy_roles_by_policy": execution_roles,
        },
        "runtime_health_contract": {
            "required_nodes": ["/simulator", "/pedsim_simulator", "/graph_navigation"],
            "required_services": ["/utmrsStepper", "/utmrsReset", "/graphNavSrv", "/utmrsObserve", "/utmrsClock"],
            "abort_on_missing_runtime": True,
        },
        "clock_contract": {
            "simulator_delta_t": float(args.sim_dt),
            "simulator_steps_per_planning_window": int(args.sim_steps_per_window),
            "planning_window_dt": float(args.env_dt),
            "observed_transition_used_for_belief_update": True,
            "zero_step_reset": True,
            "post_step_observation": True,
            "clock_verified_every_window": True,
            "pedestrian_rng_seeded_per_episode": True,
            "all_controllable_robots_exposed_to_pedestrians": True,
            "reset_configuration_verified": True,
            "partial_terminal_windows_excluded_from_belief_evidence": True,
            "runtime_boundary": "upstream force/motion equations with shared opt-in synchronization, seeded reset, and multi-robot state transport; not an unmodified official binary",
        },
        "calibration_contract": {
            "artifact": args.calibration_artifact,
            "artifact_sha256": (
                _sha256((root / args.calibration_artifact).resolve())
                if args.calibration_artifact
                else None
            ),
            "selected_scale": (
                calibration_artifact["selected"]["likelihood_scale"]
                if calibration_artifact
                else None
            ),
            "validation_only": True,
        },
        "pairing_contract": {
            "scenario_rng_seed": "seed * 1000 + episode",
            "scenario_regenerated_before_each_episode": True,
            "initial_state_hash_recorded": True,
            "cross_policy_state_hash_required": True,
            "ros_reset_seed_note": "shared synchronous reset seeds simulator and pedestrian RNGs with the explicit episode seed",
            "independent_scene_stream": True,
        },
        "records": len(records),
        "belief_evidence_rows": len(evidence_rows),
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_lines = [
        "# TRACER on official SocialGym2",
        "",
        f"Main commit `{git['commit']}`; official source `{source_commit}`.",
        (
            "The official ROS simulator supplies reset/step, graph navigation, pedsim humans, "
            "and the native GoAlone/Halt/Follow/Pass action space. The official PIPS "
            "controller supplies the external action decisions."
            if external_policy
            else
            "The official ROS simulator supplies reset/step, graph navigation, pedsim humans, "
            "and the GoAlone/Halt action space. TRACER supplies SAGE proposal ordering, a "
            "joint multi-robot response model with pair interactions, per-track belief "
            "updates, and risk-aware branch selection."
        ),
        "",
        "## Results",
        "",
    ]
    for policy in args.policies:
        item = summary["policies"][policy]
        report_lines.append(
            f"- `{policy}`: success `{item['success']['mean']:.3f}`, collision `{item['collision']['mean']:.3f}`, progress `{item['task_progress']['mean']:.3f}`, min distance `{item['min_distance']['mean']:.3f}`, belief entropy `{item['final_belief_entropy']['mean']:.3f}`."
        )
    report_lines.extend([
        "",
        (
            "This is a native SocialGym2 external-controller run: three controllable robots "
            "execute the official PIPS action returned for each robot; no TRACER response "
            "selection, belief update, geometry filter, or binary projection is applied."
            if external_policy
            else
            "This is a joint external H3 protocol: three controllable robots are scored "
            "through the same response model with pair interactions enabled, and Persistent "
            "is paired against a same-state Uniform-belief counterfactual over the full "
            "joint action lattice."
        ),
        (
            "The native four-action external interface is compared in the same patched "
            "SocialGym2 runtime; it is not presented as a continuous-action H3 replication."
            if external_policy
            else
            "The binary action projection and ID-free track association remain explicit "
            "external-adapter boundaries; this is not presented as a continuous-action H3 replication."
        ),
        "",
    ])
    report_path = output / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    artifacts = {
        name: _sha256(output / name)
        for name in (
            "metadata.json",
            "summary.json",
            "episode_metrics.csv",
            "belief_trace.csv",
            "belief_evidence.csv",
            "report.md",
        )
    }
    manifest = {
        "status": "COMPLETED" if records else "FAILED",
        "protocol": metadata["protocol"],
        "git": git,
        "external_repository": metadata["external_repository"],
        "records": len(records),
        "artifacts": artifacts,
        "claim_boundary": metadata["claim_boundary"],
    }
    if execution_diagnostics is not None:
        for name in ("execution_diagnostics.jsonl", "execution_diagnostics_manifest.json"):
            artifacts[name] = _sha256(output / name)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "records": len(records), "policies": list(args.policies)}, indent=2))
    return {"metadata": metadata, "summary": summary, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="raw_data/external/source_repos/socialgym2_official")
    parser.add_argument("--output-dir", default="experiments/socialgym2_tracer_20260903")
    parser.add_argument("--response-checkpoint", default="experiments/formal_v2/h2/persistent_trajectory_suite_tracer_effect/seed2027/response.pt")
    parser.add_argument("--sage-checkpoint", default="experiments/pilot/h3_sage_generator_p0_seed2027/synthetic_sage_epoch20.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-humans", type=int, default=2)
    parser.add_argument("--partial-observation", action="store_true")
    parser.add_argument("--scenario", choices=("envs/scenario/intersection", "envs/scenario/open"), default="envs/scenario/intersection")
    parser.add_argument("--env-dt", type=float, default=0.4)
    parser.add_argument("--model-dt", type=float, default=0.4)
    parser.add_argument(
        "--task-steps",
        type=int,
        default=2,
        help="Number of observable candidate steps used for route-progress scoring; execution remains one planning window.",
    )
    parser.add_argument(
        "--response-cost-steps",
        type=int,
        default=None,
        help="Optional response-cost horizon; affects ranking and model risk, not belief evidence timing or the one-window geometry guard.",
    )
    parser.add_argument(
        "--candidate-prefix-model",
        choices=("executable", "goal-line-legacy"),
        default="executable",
        help="Observable execution-consistent prefix (default) or the retained legacy final-goal approximation.",
    )
    parser.add_argument(
        "--candidate-max-speed",
        type=float,
        default=1.5,
        help="Maximum speed used by the observable executable candidate prefix.",
    )
    parser.add_argument(
        "--candidate-nominal-start-speed",
        type=float,
        default=1.5,
        help="Rest-to-GoAlone target speed used by the executable candidate prefix.",
    )
    parser.add_argument("--sim-dt", type=float, default=0.025)
    parser.add_argument("--sim-steps-per-window", type=int, default=16)
    parser.add_argument("--sage-candidates", type=int, default=8)
    parser.add_argument("--sage-step-stride", type=int, default=10)
    parser.add_argument("--sage-goal-speed", type=float, default=1.2)
    parser.add_argument("--candidate-seed", type=int, default=839731)
    parser.add_argument("--bootstrap-seed", type=int, default=849731)
    parser.add_argument("--human-cost-weight", type=float, default=0.6)
    parser.add_argument(
        "--task-aggregation",
        choices=("mean", "lower-tail"),
        default="mean",
        help="Aggregate per-robot task progress; lower-tail emphasizes the least-progressing unfinished robot.",
    )
    parser.add_argument(
        "--task-min-blend",
        type=float,
        default=0.5,
        help="Blend weight for the minimum unfinished-robot progress in lower-tail task aggregation.",
    )
    parser.add_argument(
        "--belief-cost-leverage",
        type=float,
        default=1.0,
        help="Scale only the belief-dependent deviation from uniform response costs; 1.0 preserves the reference protocol.",
    )
    parser.add_argument("--safety-weight", type=float, default=2.0)
    parser.add_argument("--collision-margin", type=float, default=0.65)
    parser.add_argument("--risk-alpha", type=float, default=0.1)
    parser.add_argument("--risk-samples", type=int, default=16)
    parser.add_argument(
        "--hard-risk-filter",
        action="store_true",
        help="Hard-reject candidates above risk-alpha; default is a soft risk penalty plus official safety checks.",
    )
    parser.add_argument(
        "--risk-weight",
        type=float,
        default=0.5,
        help="Additional ranking penalty for predicted violation probability among safe candidates.",
    )
    parser.add_argument(
        "--smoothness-weight",
        type=float,
        default=0.0,
        help="Penalty on the first-action velocity jump relative to current velocity.",
    )
    parser.add_argument(
        "--stagnation-trigger-seconds",
        type=float,
        default=1.2,
        help="Observed stationary unfinished duration before safe Go candidates receive a recovery preference.",
    )
    parser.add_argument(
        "--stagnation-motion-threshold",
        type=float,
        default=0.02,
        help="Maximum per-window displacement counted as stationary.",
    )
    parser.add_argument(
        "--stagnation-halt-penalty",
        type=float,
        default=1.0,
        help="Cost penalty for halting an observably stalled unfinished robot; safety filters remain authoritative.",
    )
    parser.add_argument(
        "--stagnation-release-hard-safe",
        action="store_true",
        help="After observed stagnation, admit hard-safe Go recovery even when robust clearance decreases; hard geometry remains authoritative.",
    )
    parser.add_argument(
        "--stagnation-release-require-robust-robot",
        action="store_true",
        help="Restrict stagnation recovery to candidates whose swept robust robot clearance remains nonnegative.",
    )
    parser.add_argument(
        "--stagnation-release-require-nonworsening-robot",
        action="store_true",
        help="Restrict stagnation recovery to candidates whose terminal robot clearance is no worse than the current clearance.",
    )
    parser.add_argument(
        "--stagnation-release-robot-clearance-floor",
        type=float,
        default=None,
        help="Optional global swept robot-clearance floor for stagnation recovery; hard geometry remains authoritative.",
    )
    parser.add_argument(
        "--stagnation-recovery-safety-first",
        action="store_true",
        help="If the response-cost recovery choice crosses the robust robot margin, rank hard-safe recovery candidates by swept clearance.",
    )
    parser.add_argument("--likelihood-scale", type=float, default=0.05)
    parser.add_argument("--calibration-artifact", default=None)
    parser.add_argument("--runtime-manifest", default=None)
    parser.add_argument("--execution-diagnostics", action="store_true")
    parser.add_argument("--joint-geometry-filter", action="store_true")
    parser.add_argument(
        "--completed-robot-halt",
        action="store_true",
        help="Command Halt after observed goal success; predict that same stationary prefix while retaining completed robots in safety checks.",
    )
    parser.add_argument(
        "--observed-human-geometry-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add an observable one-window swept robot-human guard to joint geometry.",
    )
    parser.add_argument(
        "--execution-position-margin",
        type=float,
        default=0.1,
        help="Measured short-prefix position-error allowance added to robot and static geometry thresholds.",
    )
    parser.add_argument("--hazard-rate", type=float, default=0.18)
    parser.add_argument("--forgetting-rate", type=float, default=0.18)
    parser.add_argument("--change-point-threshold", type=float, default=5.0)
    parser.add_argument("--max-missed", type=int, default=3)
    parser.add_argument(
        "--belief-selection-audit",
        action="store_true",
        help="Record same-state Uniform-belief score, selection, and joint-action counterfactuals.",
    )
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
