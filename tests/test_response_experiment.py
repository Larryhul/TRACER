from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from tracer.data.heterogeneous_scene import AgentRole, HeterogeneousSceneBatch
from tracer.response.data import (
    build_entity_response_batch,
    shuffle_robot_candidates,
    shuffle_robot_candidates_within_groups,
)
from tracer.response.causal_synthetic import CausalResponseSpec, CausalTeamResponseDataset
from tracer.response.closed_loop import (
    ClosedLoopSpec,
    PlanningCostConfig,
    SyntheticClosedLoopSimulator,
    expected_candidate_costs,
)
from tracer.response.factorial import (
    factorial_second_difference,
    interaction_recovery_metrics,
    pairwise_factorial_interactions,
)
from tracer.response.metrics import gaussian_nll_per_sample, response_metrics
from tracer.response.models import ResponseModelConfig, TeamResponsePredictor
from tracer.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
    TrajectoryRecurrentBelief,
    TrajectoryRecurrentConfig,
    belief_weighted_mean,
    executed_prefix_mode_log_likelihood,
    mode_conditioned_trajectory_loss,
    stationary_reference_candidates,
    trajectory_mode_log_likelihood,
)
from tracer.response.persistent_trajectory import (
    MultiEntityPersistentTrajectoryEpisodeDataset,
    MultiEntityPersistentTrajectorySpec,
    PersistentTrajectoryEpisodeDataset,
    PersistentTrajectorySpec,
    candidate_independent_active_mask,
)
from tracer.response.sage_candidate_adapter import (
    SAGECandidateAdapter,
    SAGECandidateAdapterConfig,
)
from tracer.experiments.train_response_baselines import main as train_response_main
from tracer.experiments.train_persistent_trajectory_response import (
    _flatten_entities,
    _shuffled_identity_filter,
)
from tracer.experiments.formal_manifest import candidate_pool_sha256
from tracer.experiments.eval_persistent_closed_loop import (
    _candidate_and_reference_outputs,
    _kendall_tau_b,
)


def make_scene_batch() -> HeterogeneousSceneBatch:
    batch_size, agents, history_steps, horizon = 2, 4, 3, 2
    history = torch.zeros(batch_size, agents, history_steps, 2)
    current = torch.tensor(
        [
            [[1.0, 0.0], [4.0, 0.0], [0.0, 0.0], [0.0, 2.0]],
            [[-3.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [0.0, -2.0]],
        ]
    )
    history[:, :, -1] = current
    history[:, :, -2] = current - 0.4
    history[:, :, -3] = current - 0.8
    future = torch.zeros(batch_size, agents, horizon, 2)
    future[0, 0, :, 0] = 1.0
    future[0, 1, :, 1] = 1.0
    future[1, 0, :, 0] = -2.0
    future[1, 1, :, 1] = -2.0
    future[:, 2:, :, 0] = 0.5
    roles = torch.tensor(
        [[AgentRole.ROBOT, AgentRole.ROBOT, AgentRole.PHA, AgentRole.PHA]] * batch_size
    )
    robot_mask = torch.tensor([[True, True, False, False]] * batch_size)
    entity_mask = ~robot_mask
    agent_mask = torch.ones(batch_size, agents, dtype=torch.bool)
    return HeterogeneousSceneBatch(
        history=history,
        future=future,
        current_pos=current,
        roles=roles.long(),
        agent_mask=agent_mask,
        robot_mask=robot_mask,
        entity_mask=entity_mask,
        loss_mask=entity_mask.clone(),
        dt=0.4,
    )


class ResponseDataTest(unittest.TestCase):
    def test_flatten_and_nearest_robot_order(self) -> None:
        response = build_entity_response_batch(make_scene_batch(), num_robots=2, interaction_radius=5.0)
        self.assertEqual(response.entity_history.shape, (4, 3, 2))
        self.assertEqual(response.robot_current.shape, (4, 2, 4))
        self.assertEqual(response.robot_candidates.shape, (4, 2, 2, 4))
        self.assertEqual(response.target_future_vel.shape, (4, 2, 2))
        torch.testing.assert_close(response.robot_current[0, :, :2], torch.tensor([[1.0, 0.0], [4.0, 0.0]]))
        torch.testing.assert_close(response.robot_current[2, :, :2], torch.tensor([[-1.0, 0.0], [-3.0, 0.0]]))
        self.assertTrue(response.high_interaction.all())

    def test_shuffle_changes_candidate_velocity_without_changing_current(self) -> None:
        response = build_entity_response_batch(make_scene_batch(), num_robots=2)
        shuffled = shuffle_robot_candidates(response)
        torch.testing.assert_close(shuffled.robot_current, response.robot_current)
        self.assertTrue(torch.equal(shuffled.robot_mask, response.robot_mask))
        self.assertFalse(torch.equal(shuffled.robot_candidates[..., 2:], response.robot_candidates[..., 2:]))

    def test_within_group_shuffle_preserves_factorial_geometry_and_group_boundaries(self) -> None:
        response = build_entity_response_batch(make_scene_batch(), num_robots=2)
        grouped = shuffle_robot_candidates_within_groups(
            response,
            group_size=4,
            branch_size=2,
        )
        torch.testing.assert_close(grouped.robot_current, response.robot_current)
        torch.testing.assert_close(grouped.entity_current, response.entity_current)
        self.assertTrue(torch.equal(grouped.robot_mask, response.robot_mask))
        torch.testing.assert_close(
            grouped.robot_candidates[:2, ..., 2:], response.robot_candidates[2:4, ..., 2:]
        )
        torch.testing.assert_close(
            grouped.robot_candidates[2:4, ..., 2:], response.robot_candidates[:2, ..., 2:]
        )
        torch.testing.assert_close(
            grouped.robot_candidates[4:6, ..., 2:], response.robot_candidates[6:8, ..., 2:]
        )
        torch.testing.assert_close(
            grouped.robot_candidates[6:8, ..., 2:], response.robot_candidates[4:6, ..., 2:]
        )


class ResponseModelTest(unittest.TestCase):
    def test_all_variants_forward_and_nll_backward(self) -> None:
        response = build_entity_response_batch(make_scene_batch(), num_robots=2)
        config = ResponseModelConfig(history=3, horizon=2, num_robots=2, hidden_dim=16, num_heads=4, dropout=0.0)
        for variant in sorted(TeamResponsePredictor.VALID_VARIANTS):
            model = TeamResponsePredictor(variant, config)
            output = model(response)
            self.assertEqual(output["mean"].shape, (4, 2, 2))
            self.assertTrue(torch.isfinite(output["mean"]).all())
            self.assertTrue(torch.isfinite(output["log_std"]).all())
            loss = gaussian_nll_per_sample(
                output["mean"], output["log_std"], response.target_future_vel
            ).mean()
            loss.backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_tracer_accepts_optional_high_level_robot_actions(self) -> None:
        scene = replace(
            make_scene_batch(),
            robot_actions=torch.tensor(
                [[0, 1, -1, -1], [1, 0, -1, -1]], dtype=torch.long
            ),
        )
        response = build_entity_response_batch(scene, num_robots=2)
        config = ResponseModelConfig(
            history=3,
            horizon=2,
            num_robots=2,
            hidden_dim=16,
            num_heads=4,
            architecture="tracer",
            dropout=0.0,
        )
        model = TeamResponsePredictor("joint", config).eval()
        without_actions = replace(
            response, robot_actions=torch.zeros_like(response.robot_actions)
        )
        with torch.no_grad():
            conditioned = model(response)
            unconditioned = model(without_actions)
        self.assertFalse(torch.equal(conditioned["mean"], unconditioned["mean"]))

    def test_additive_variants_sum_independent_robot_deltas(self) -> None:
        response = build_entity_response_batch(make_scene_batch(), num_robots=2)
        config = ResponseModelConfig(history=3, horizon=2, num_robots=2, hidden_dim=16, num_heads=4, dropout=0.0)
        no_robot = replace(response, robot_mask=torch.zeros_like(response.robot_mask))
        first_only = replace(
            response,
            robot_mask=torch.tensor([[True, False]] * response.num_samples),
        )
        second_only = replace(
            response,
            robot_mask=torch.tensor([[False, True]] * response.num_samples),
        )
        for variant in ("additive", "matched-additive"):
            model = TeamResponsePredictor(variant, config).eval()
            with torch.no_grad():
                base = model(no_robot)
                first = model(first_only)
                second = model(second_only)
                full = model(response)
            torch.testing.assert_close(
                full["mean"], first["mean"] + second["mean"] - base["mean"]
            )
            torch.testing.assert_close(full["log_std"], base["log_std"])

    def test_matched_additive_parameter_count_is_within_one_percent_of_joint(self) -> None:
        config = ResponseModelConfig(
            history=8, horizon=12, num_robots=2, hidden_dim=96, num_heads=4, dropout=0.0
        )
        matched = TeamResponsePredictor("matched-additive", config)
        joint = TeamResponsePredictor("joint", config)
        matched_count = sum(parameter.numel() for parameter in matched.parameters())
        joint_count = sum(parameter.numel() for parameter in joint.parameters())
        self.assertLess(abs(matched_count - joint_count) / joint_count, 0.01)

    def test_tracer_residual_branches_are_trainable_parameter_matched(self) -> None:
        config = ResponseModelConfig(
            history=8,
            horizon=2,
            num_robots=2,
            hidden_dim=96,
            num_heads=4,
            dropout=0.0,
            architecture="tracer",
        )
        matched = TeamResponsePredictor("matched-additive", config)
        joint = TeamResponsePredictor("joint", config)
        matched_count = sum(
            parameter.numel()
            for name, parameter in matched.named_parameters()
            if name.startswith("tracer_capacity_")
        )
        joint_count = sum(
            parameter.numel()
            for name, parameter in joint.named_parameters()
            if name.startswith("tracer_pair_")
            or name.startswith("tracer_temporal_pair_")
        )
        self.assertLess(abs(matched_count - joint_count) / joint_count, 0.01)

    def test_tracer_joint_residual_starts_at_additive_reduction(self) -> None:
        torch.manual_seed(17)
        config = ResponseModelConfig(
            history=3,
            horizon=2,
            num_robots=2,
            hidden_dim=16,
            num_heads=2,
            dropout=0.0,
            architecture="tracer",
        )
        additive = TeamResponsePredictor("additive", config).eval()
        joint = TeamResponsePredictor("joint", config).eval()
        joint.load_state_dict(additive.state_dict(), strict=False)
        response = build_entity_response_batch(
            make_scene_batch(), num_robots=2, interaction_radius=5.0
        )
        with_pair = joint(response, include_pair_interactions=True)
        without_pair = joint(response, include_pair_interactions=False)
        torch.testing.assert_close(with_pair["mean"], without_pair["mean"])
        torch.testing.assert_close(with_pair["log_std"], without_pair["log_std"])

    def test_tracer_matched_residual_starts_at_additive_reduction(self) -> None:
        torch.manual_seed(19)
        config = ResponseModelConfig(
            history=3,
            horizon=2,
            num_robots=2,
            hidden_dim=16,
            num_heads=2,
            dropout=0.0,
            architecture="tracer",
        )
        additive = TeamResponsePredictor("additive", config).eval()
        matched = TeamResponsePredictor("matched-additive", config).eval()
        matched.load_state_dict(additive.state_dict(), strict=False)
        response = build_entity_response_batch(
            make_scene_batch(), num_robots=2, interaction_radius=5.0
        )
        additive_output = additive(response)
        matched_output = matched(response)
        torch.testing.assert_close(matched_output["mean"], additive_output["mean"])
        torch.testing.assert_close(matched_output["log_std"], additive_output["log_std"])

    def test_tracer_dynamic_interaction_features_change_candidate_encoding(self) -> None:
        config = ResponseModelConfig(
            history=3,
            horizon=2,
            num_robots=2,
            hidden_dim=16,
            num_heads=2,
            dropout=0.0,
            architecture="tracer",
            dynamic_interaction_features=True,
        )
        model = TeamResponsePredictor("joint", config).eval()
        response = build_entity_response_batch(
            make_scene_batch(), num_robots=2, interaction_radius=5.0
        )
        shifted = replace(
            response,
            entity_history=response.entity_history.clone(),
        )
        shifted.entity_history[:, -1] = shifted.entity_history[:, -1] + 1.0
        original = model(response)
        changed = model(shifted)
        self.assertFalse(torch.equal(original["mean"], changed["mean"]))

    def test_tracer_temporal_pair_head_is_time_resolved_and_parameter_matched(self) -> None:
        config = ResponseModelConfig(
            history=8,
            horizon=2,
            num_robots=2,
            hidden_dim=96,
            num_heads=4,
            dropout=0.0,
            architecture="tracer",
            geometry_features=True,
            temporal_pooling="attention",
            dynamic_interaction_features=True,
            temporal_pair_interactions=True,
        )
        joint = TeamResponsePredictor("joint", config)
        matched = TeamResponsePredictor("matched-additive", config)
        response = build_entity_response_batch(
            make_scene_batch(), num_robots=2, interaction_radius=5.0
        )
        output = joint(response)
        self.assertEqual(output["mean"].shape, (response.num_samples, 2, 2))
        joint_count = sum(
            parameter.numel()
            for name, parameter in joint.named_parameters()
            if name.startswith("tracer_pair_")
        )
        matched_count = sum(
            parameter.numel()
            for name, parameter in matched.named_parameters()
            if name.startswith("tracer_capacity_")
        )
        self.assertLess(abs(joint_count - matched_count) / joint_count, 0.01)

    def test_conditioned_variants_reduce_to_history_base_for_zero_robot_input(self) -> None:
        response = build_entity_response_batch(make_scene_batch(), num_robots=2)
        zero_response = replace(
            response,
            robot_current=torch.zeros_like(response.robot_current),
            robot_candidates=torch.zeros_like(response.robot_candidates),
        )
        config = ResponseModelConfig(history=3, horizon=2, num_robots=2, hidden_dim=16, num_heads=4, dropout=0.0)
        for variant in (
            "nearest",
            "additive",
            "matched-additive",
            "joint",
            "team-concat",
        ):
            model = TeamResponsePredictor(variant, config).eval()
            with torch.no_grad():
                entity_context = model._encode_history(zero_response)
                base_mean = model.base_mean_decoder(entity_context).reshape(-1, 2, 2)
                output = model(zero_response)
            torch.testing.assert_close(output["mean"], base_mean, atol=1e-6, rtol=1e-6)

    def test_joint_prediction_is_invariant_to_robot_order(self) -> None:
        response = build_entity_response_batch(make_scene_batch(), num_robots=2)
        swapped = replace(
            response,
            robot_current=response.robot_current.flip(1),
            robot_candidates=response.robot_candidates.flip(1),
            robot_mask=response.robot_mask.flip(1),
        )
        config = ResponseModelConfig(
            history=3, horizon=2, num_robots=2, hidden_dim=16, num_heads=4, dropout=0.0
        )
        model = TeamResponsePredictor("joint", config).eval()
        with torch.no_grad():
            original = model(response)
            permuted = model(swapped)
        torch.testing.assert_close(original["mean"], permuted["mean"], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(original["log_std"], permuted["log_std"])

    def test_joint_is_initialized_as_the_nested_additive_model(self) -> None:
        response = build_entity_response_batch(make_scene_batch(), num_robots=2)
        config = ResponseModelConfig(
            history=3, horizon=2, num_robots=2, hidden_dim=16, num_heads=4, dropout=0.0
        )
        torch.manual_seed(31)
        additive = TeamResponsePredictor("additive", config).eval()
        torch.manual_seed(31)
        joint = TeamResponsePredictor("joint", config).eval()
        with torch.no_grad():
            additive_output = additive(response)
            joint_output = joint(response)
        torch.testing.assert_close(joint_output["mean"], additive_output["mean"])
        torch.testing.assert_close(joint_output["log_std"], additive_output["log_std"])

    def test_joint_interaction_is_zero_when_either_robot_condition_is_zero(self) -> None:
        response = build_entity_response_batch(make_scene_batch(), num_robots=2)
        zero_second = replace(
            response,
            robot_current=response.robot_current.clone(),
            robot_candidates=response.robot_candidates.clone(),
        )
        zero_second.robot_current[:, 1] = 0.0
        zero_second.robot_candidates[:, 1] = 0.0
        masked_second = replace(
            zero_second,
            robot_mask=torch.tensor([[True, False]] * response.num_samples),
        )
        config = ResponseModelConfig(
            history=3, horizon=2, num_robots=2, hidden_dim=16, num_heads=4, dropout=0.0
        )
        model = TeamResponsePredictor("joint", config).eval()
        torch.nn.init.normal_(model.pair_delta_decoder[-1].weight)
        torch.nn.init.normal_(model.pair_delta_decoder[-1].bias)
        with torch.no_grad():
            zero_output = model(zero_second)
            masked_output = model(masked_second)
        torch.testing.assert_close(zero_output["mean"], masked_output["mean"], atol=1e-6, rtol=1e-6)

    def test_joint_pair_residual_can_be_disabled_without_changing_uncertainty(self) -> None:
        response = build_entity_response_batch(make_scene_batch(), num_robots=2)
        config = ResponseModelConfig(
            history=3, horizon=2, num_robots=2, hidden_dim=16, num_heads=4, dropout=0.0
        )
        model = TeamResponsePredictor("joint", config).eval()
        torch.nn.init.normal_(model.pair_delta_decoder[-1].weight)
        torch.nn.init.normal_(model.pair_delta_decoder[-1].bias)
        with torch.no_grad():
            joint = model(response)
            without_pair = model(response, include_pair_interactions=False)
        self.assertFalse(torch.equal(joint["mean"], without_pair["mean"]))
        torch.testing.assert_close(joint["log_std"], without_pair["log_std"])

    def test_metrics_are_zero_for_exact_mean_trajectory(self) -> None:
        target = torch.ones(2, 3, 2)
        metrics = response_metrics(target, torch.zeros_like(target), target, dt=0.4)
        torch.testing.assert_close(metrics["ade"], torch.zeros(2))
        torch.testing.assert_close(metrics["fde"], torch.zeros(2))

    def test_resume_existing_accepts_an_entirely_completed_run(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config_path = root / "config.yaml"
            output_dir = root / "outputs"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "dataset": {
                            "source": "causal_response",
                            "history": 3,
                            "horizon": 2,
                            "causal_response": {
                                "train_base_scenes": 2,
                                "val_base_scenes": 1,
                                "interventions_per_scene": 4,
                                "num_robots": 2,
                                "num_entities": 2,
                                "seed": 19,
                            },
                        },
                        "model": {
                            "num_robots": 2,
                            "hidden_dim": 8,
                            "num_heads": 2,
                            "dropout": 0.0,
                        },
                        "training": {
                            "variants": ["additive"],
                            "batch_size": 8,
                            "epochs": 1,
                            "num_workers": 0,
                            "checkpoint_rule": "best_val_nll",
                        },
                        "evaluation": {"interaction_radius": 4.0},
                    }
                ),
                encoding="utf-8",
            )
            arguments = [
                "train_response_baselines",
                "--config",
                str(config_path),
                "--device",
                "cpu",
                "--output-dir",
                str(output_dir),
                "--effect-mode",
                "none",
                "--seed",
                "23",
                "--data-seed",
                "19",
            ]
            with patch("sys.argv", arguments):
                train_response_main()
            with patch("sys.argv", [*arguments, "--resume-existing"]):
                train_response_main()

            metadata = yaml.safe_load((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["checkpoint_rule"], "best_val_nll")
            self.assertTrue(metadata["resume_existing"])


class CausalResponseDatasetTest(unittest.TestCase):
    def _dataset(self, mode: str, split: str = "train") -> CausalTeamResponseDataset:
        return CausalTeamResponseDataset(
            CausalResponseSpec(
                split=split,
                num_base_scenes=2,
                interventions_per_scene=4,
                num_robots=2,
                num_entities=4,
                effect_mode=mode,
                seed=17,
            )
        )

    def test_paired_interventions_hold_history_fixed(self) -> None:
        dataset = self._dataset("none")
        first, second = dataset[0], dataset[1]
        self.assertEqual(first["base_scene_id"], second["base_scene_id"])
        self.assertNotEqual(first["intervention_id"], second["intervention_id"])
        torch.testing.assert_close(first["history"], second["history"])
        self.assertFalse(
            torch.equal(first["future"][first["robot_mask"]], second["future"][second["robot_mask"]])
        )
        torch.testing.assert_close(
            first["future"][first["entity_mask"]], second["future"][second["entity_mask"]]
        )

    def test_factorial_interventions_change_only_the_selected_robot_level(self) -> None:
        dataset = self._dataset("none")
        samples = [dataset[index] for index in range(4)]
        actions = [sample["future"][sample["robot_mask"]] for sample in samples]

        # Bit 0 selects robot 0's action level; bit 1 selects robot 1's.
        torch.testing.assert_close(actions[0][0], actions[2][0])
        torch.testing.assert_close(actions[1][0], actions[3][0])
        self.assertFalse(torch.equal(actions[0][0], actions[1][0]))

        torch.testing.assert_close(actions[0][1], actions[1][1])
        torch.testing.assert_close(actions[2][1], actions[3][1])
        self.assertFalse(torch.equal(actions[0][1], actions[2][1]))

    def test_effect_mechanisms_share_actions_but_change_responses(self) -> None:
        additive = self._dataset("additive")[0]
        nonlinear = self._dataset("nonlinear")[0]
        torch.testing.assert_close(additive["history"], nonlinear["history"])
        torch.testing.assert_close(
            additive["future"][additive["robot_mask"]],
            nonlinear["future"][nonlinear["robot_mask"]],
        )
        self.assertFalse(
            torch.equal(
                additive["future"][additive["entity_mask"]],
                nonlinear["future"][nonlinear["entity_mask"]],
            )
        )

    def test_train_and_validation_base_scene_ids_do_not_overlap(self) -> None:
        train = self._dataset("none", split="train")
        validation = self._dataset("none", split="val")
        train_ids = {train[index]["base_scene_id"] for index in range(len(train))}
        validation_ids = {validation[index]["base_scene_id"] for index in range(len(validation))}
        self.assertTrue(train_ids.isdisjoint(validation_ids))

    def test_cached_samples_are_deterministic_and_reused(self) -> None:
        dataset = CausalTeamResponseDataset(
            replace(self._dataset("nonlinear").spec, cache_samples=True)
        )
        first = dataset[0]
        second = dataset[0]
        self.assertIs(first, second)
        torch.testing.assert_close(first["history"], second["history"])
        torch.testing.assert_close(first["future"], second["future"])

    def test_factorial_truth_is_zero_only_without_pair_interactions(self) -> None:
        for mode in ("none", "additive", "nonlinear"):
            dataset = self._dataset(mode)
            responses = torch.stack(
                [dataset[index]["future"][dataset[index]["entity_mask"]] for index in range(4)]
            )
            interaction = factorial_second_difference(responses)
            if mode == "nonlinear":
                self.assertGreater(float(interaction.square().mean()), 1e-5)
            else:
                torch.testing.assert_close(interaction, torch.zeros_like(interaction), atol=1e-6, rtol=0.0)

    def test_extended_scene_and_response_families_are_paired(self) -> None:
        for scene_family in ("radial", "corridor", "crossing", "bottleneck"):
            for response_family in ("pincer", "speed", "lateral", "threshold"):
                dataset = CausalTeamResponseDataset(
                    CausalResponseSpec(
                        num_base_scenes=1,
                        num_robots=3,
                        interventions_per_scene=8,
                        num_entities=4,
                        scene_family=scene_family,
                        response_family=response_family,
                    )
                )
                samples = [dataset[index] for index in range(8)]
                self.assertEqual(samples[0]["layout"], f"paired_intervention:{scene_family}")
                self.assertEqual(samples[0]["response_family"], response_family)
                torch.testing.assert_close(samples[0]["history"], samples[1]["history"])



class FactorialInteractionTest(unittest.TestCase):
    def test_second_difference_cancels_additive_main_effects(self) -> None:
        base = torch.tensor([2.0, -1.0])
        first = torch.tensor([0.4, 0.7])
        second = torch.tensor([-0.2, 0.5])
        interaction = torch.tensor([0.3, -0.6])
        values = torch.stack(
            [base, base + first, base + second, base + first + second + interaction]
        )
        torch.testing.assert_close(factorial_second_difference(values), interaction)

    def test_second_difference_supports_batched_intervention_dimension(self) -> None:
        values = torch.zeros(2, 4, 3)
        values[:, 3] = 2.0
        torch.testing.assert_close(
            factorial_second_difference(values, dim=1),
            torch.full((2, 3), 2.0),
        )

    def test_pairwise_contrasts_average_over_complement_bits(self) -> None:
        values = torch.zeros(8, 1)
        for intervention in range(8):
            bits = [(intervention >> bit) & 1 for bit in range(3)]
            values[intervention] = (
                bits[0] * bits[1] + 3 * bits[0] * bits[2] + 2 * bits[1] * bits[2]
            )
        contrasts = pairwise_factorial_interactions(values, num_robots=3)
        torch.testing.assert_close(contrasts[:, 0], torch.tensor([1.0, 3.0, 2.0]))

    def test_recovery_metrics_reward_exact_interaction(self) -> None:
        target = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
        metrics = interaction_recovery_metrics(target, target)
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["pearson"], 1.0)
        self.assertAlmostEqual(metrics["cosine"], 1.0)
        self.assertAlmostEqual(metrics["zero_baseline_r2"], 1.0)

    def test_recovery_metrics_do_not_normalize_zero_interaction(self) -> None:
        target = torch.zeros(3, 2)
        prediction = torch.full_like(target, 0.1)
        metrics = interaction_recovery_metrics(prediction, target)
        self.assertTrue(torch.isnan(torch.tensor(metrics["relative_rmse"])))
        self.assertTrue(torch.isnan(torch.tensor(metrics["zero_baseline_r2"])))


class PersistentTrajectoryIntegrationTest(unittest.TestCase):
    def _dataset(self, split: str = "train", policy: str = "iid") -> PersistentTrajectoryEpisodeDataset:
        return PersistentTrajectoryEpisodeDataset(
            PersistentTrajectorySpec(
                split=split,
                num_episodes=8,
                windows=4,
                history=5,
                horizon=6,
                seed=41,
                action_policy=policy,
            )
        )

    def test_episode_shapes_identity_and_split_separation(self) -> None:
        train = self._dataset("train")
        validation = self._dataset("val")
        sample = train[0]
        self.assertEqual(sample["entity_history"].shape, (4, 5, 2))
        self.assertEqual(sample["robot_current"].shape, (4, 2, 4))
        self.assertEqual(sample["robot_candidates"].shape, (4, 2, 6, 4))
        self.assertEqual(sample["target_future_vel"].shape, (4, 6, 2))
        self.assertNotEqual(train[0]["entity_id"], validation[0]["entity_id"])
        self.assertFalse(torch.equal(sample["entity_history"][0], sample["entity_history"][1]))

    def test_ood_policy_correlates_robot_activity(self) -> None:
        iid = self._dataset(policy="iid")
        correlated = self._dataset(split="ood", policy="correlated")
        iid_activity = torch.cat([iid[index]["activities"] for index in range(len(iid))])
        ood_activity = torch.cat(
            [correlated[index]["activities"] for index in range(len(correlated))]
        )
        iid_correlation = torch.corrcoef(iid_activity.T)[0, 1]
        ood_correlation = torch.corrcoef(ood_activity.T)[0, 1]
        self.assertLess(abs(float(iid_correlation)), 0.5)
        self.assertGreater(float(ood_correlation), 0.9)

    def test_three_robot_episode_and_model_are_permutation_invariant(self) -> None:
        dataset = PersistentTrajectoryEpisodeDataset(
            PersistentTrajectorySpec(
                split="train",
                num_episodes=2,
                windows=4,
                history=5,
                horizon=6,
                num_robots=3,
                seed=43,
            )
        )
        sample = dataset[0]
        self.assertEqual(sample["robot_current"].shape, (4, 3, 4))
        self.assertEqual(sample["robot_candidates"].shape, (4, 3, 6, 4))
        model = BeliefConditionedTeamResponseModel(
            PersistentTrajectoryModelConfig(
                history=5,
                horizon=6,
                num_robots=3,
                hidden_dim=24,
                dropout=0.0,
            )
        ).eval()
        history = sample["entity_history"][None]
        current = sample["robot_current"][None]
        candidates = sample["robot_candidates"][None]
        permutation = torch.tensor([2, 0, 1])
        with torch.no_grad():
            original = model(history, current, candidates)
            permuted = model(
                history,
                current[:, :, permutation],
                candidates[:, :, permutation],
            )
        torch.testing.assert_close(original["mean"], permuted["mean"], atol=1e-6, rtol=1e-6)

    def test_persistent_scene_and_response_families_preserve_contract(self) -> None:
        for scene_family in ("radial", "corridor", "crossing", "bottleneck"):
            for response_family in ("pincer", "speed", "lateral", "threshold"):
                dataset = PersistentTrajectoryEpisodeDataset(
                    PersistentTrajectorySpec(
                        num_episodes=1,
                        windows=4,
                        history=5,
                        horizon=6,
                        num_robots=3,
                        scene_family=scene_family,
                        response_family=response_family,
                        seed=67,
                    )
                )
                sample = dataset[0]
                self.assertEqual(sample["scene_family"], scene_family)
                self.assertEqual(sample["response_family"], response_family)
                self.assertEqual(sample["robot_current"].shape, (4, 3, 4))

    def test_one_robot_model_has_no_pair_interaction(self) -> None:
        model = BeliefConditionedTeamResponseModel(
            PersistentTrajectoryModelConfig(
                history=5,
                horizon=6,
                num_robots=1,
                hidden_dim=24,
                dropout=0.0,
            )
        ).eval()
        history = torch.randn(2, 4, 5, 2)
        current = torch.randn(2, 4, 1, 4)
        candidates = torch.randn(2, 4, 1, 6, 4)
        with torch.no_grad():
            additive = model(
                history, current, candidates, include_pair_interactions=False
            )
            nominal_joint = model(
                history, current, candidates, include_pair_interactions=True
            )
        torch.testing.assert_close(additive["mean"], nominal_joint["mean"])
        torch.testing.assert_close(additive["log_std"], nominal_joint["log_std"])

    def test_robot_mask_matches_explicit_smaller_team(self) -> None:
        torch.manual_seed(13)
        model = BeliefConditionedTeamResponseModel(
            PersistentTrajectoryModelConfig(
                history=5,
                horizon=6,
                num_robots=3,
                hidden_dim=24,
                dropout=0.0,
            )
        ).eval()
        history = torch.randn(2, 4, 5, 2)
        current = torch.randn(2, 4, 3, 4)
        candidates = torch.randn(2, 4, 3, 6, 4)
        robot_mask = torch.tensor([[[True, True, False]] * 4] * 2)
        with torch.no_grad():
            masked = model(history, current, candidates, robot_mask=robot_mask)
            smaller = model(history, current[:, :, :2], candidates[:, :, :2])
        torch.testing.assert_close(masked["mean"], smaller["mean"], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(masked["log_std"], smaller["log_std"])

    def test_reference_action_preserves_current_geometry(self) -> None:
        torch.manual_seed(17)
        model = BeliefConditionedTeamResponseModel(
            PersistentTrajectoryModelConfig(
                history=5,
                horizon=6,
                num_robots=2,
                hidden_dim=24,
                dropout=0.0,
            )
        ).eval()
        history = torch.randn(2, 4, 5, 2)
        current = torch.randn(2, 4, 2, 4)
        reference = stationary_reference_candidates(current, horizon=6)
        self.assertEqual(reference.shape, (2, 4, 2, 6, 4))
        torch.testing.assert_close(reference[..., :2], current[..., None, :2].expand_as(reference[..., :2]))
        torch.testing.assert_close(reference[..., 2:], torch.zeros_like(reference[..., 2:]))
        with torch.no_grad():
            default_reference = model(history, current, reference)
            explicit_reference = model(
                history,
                current,
                reference,
                reference_candidates=reference,
            )
        torch.testing.assert_close(
            default_reference["mean"], explicit_reference["mean"], atol=1e-6, rtol=1e-6
        )

    def test_multi_entity_episode_has_id_reentry_and_candidate_independent_top_k(self) -> None:
        dataset = MultiEntityPersistentTrajectoryEpisodeDataset(
            MultiEntityPersistentTrajectorySpec(
                split="train",
                num_episodes=2,
                windows=8,
                history=5,
                horizon=6,
                num_robots=3,
                num_entities=10,
                active_entities=3,
                seed=47,
            )
        )
        sample = dataset[0]
        self.assertEqual(sample["entity_history"].shape, (10, 8, 5, 2))
        self.assertEqual(sample["robot_current"].shape, (10, 8, 3, 4))
        self.assertEqual(sample["robot_candidates"].shape, (10, 8, 3, 6, 4))
        self.assertTrue(torch.equal(sample["active_mask"].sum(dim=0), torch.full((8,), 3)))
        self.assertTrue(bool(sample["active_mask"][0, :2].all()))
        self.assertFalse(bool(sample["presence_mask"][0, 2:4].any()))
        self.assertTrue(bool(sample["active_mask"][0, 4:].all()))

        changed_candidates = sample["robot_candidates"].clone()
        changed_candidates.uniform_(-100.0, 100.0)
        recomputed, score = candidate_independent_active_mask(
            sample["robot_current"],
            sample["presence_mask"],
            active_entities=3,
        )
        torch.testing.assert_close(recomputed, sample["active_mask"])
        torch.testing.assert_close(score, sample["selection_score"])
        self.assertFalse(torch.equal(changed_candidates, sample["robot_candidates"]))

    def test_cached_multi_entity_samples_are_bitwise_identical(self) -> None:
        kwargs = dict(
            split="test",
            num_episodes=2,
            windows=5,
            history=5,
            horizon=6,
            num_robots=3,
            num_entities=4,
            active_entities=2,
            shared_team_actions=True,
            seed=61,
        )
        uncached = MultiEntityPersistentTrajectoryEpisodeDataset(
            MultiEntityPersistentTrajectorySpec(**kwargs, cache_samples=False)
        )
        cached = MultiEntityPersistentTrajectoryEpisodeDataset(
            MultiEntityPersistentTrajectorySpec(**kwargs, cache_samples=True)
        )
        for index in range(2):
            uncached_sample = uncached[index]
            cached_sample = cached[index]
            self.assertEqual(set(uncached_sample), set(cached_sample))
            for key, value in uncached_sample.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, cached_sample[key], atol=0.0, rtol=0.0)
                else:
                    self.assertEqual(value, cached_sample[key])
            self.assertIs(cached[index], cached[index])

    def test_shared_team_vectorized_response_matches_entity_reference(self) -> None:
        spec = MultiEntityPersistentTrajectorySpec(
            num_episodes=1,
            windows=8,
            history=5,
            horizon=6,
            num_robots=3,
            num_entities=4,
            active_entities=2,
            shared_team_actions=True,
            response_noise=0.18,
            influence_radius=4.0,
            execute_steps=3,
            entity_max_speed=3.0,
            seed=67,
        )
        dataset = MultiEntityPersistentTrajectoryEpisodeDataset(spec)
        rng = np.random.default_rng(71)
        entities, robots, horizon = 4, 3, 6
        modes = rng.integers(0, 3, size=entities)
        entity_position = rng.uniform(-1.4, 1.4, size=(entities, 2)).astype(np.float32)
        preferred_velocity = rng.normal(size=(entities, 2)).astype(np.float32)
        robot_position = rng.normal(size=(robots, 2)).astype(np.float32)
        future_velocity = rng.normal(size=(robots, horizon, 2)).astype(np.float32)
        current = np.concatenate(
            [
                robot_position[None] - entity_position[:, None],
                np.zeros((entities, robots, 2), dtype=np.float32),
            ],
            axis=-1,
        )
        relative_future = (
            robot_position[None, :, None, :]
            + np.cumsum(future_velocity * spec.dt, axis=1)[None]
            - entity_position[:, None, None, :]
        )
        candidates = np.concatenate(
            [relative_future, np.broadcast_to(future_velocity, (entities, robots, horizon, 2))],
            axis=-1,
        )
        reference_rng = np.random.default_rng(73)
        reference = np.stack(
            [
                dataset._tracks._response(
                    reference_rng,
                    int(modes[entity]),
                    entity_position[entity],
                    preferred_velocity[entity],
                    current[entity],
                    candidates[entity],
                )
                for entity in range(entities)
            ]
        )
        vectorized = dataset._responses(
            np.random.default_rng(73),
            modes,
            entity_position,
            preferred_velocity,
            robot_position,
            future_velocity,
        )
        np.testing.assert_allclose(reference, vectorized, atol=2e-6, rtol=0.0)

    def test_multi_entity_batch_flattening_preserves_episode_identity(self) -> None:
        dataset = MultiEntityPersistentTrajectoryEpisodeDataset(
            MultiEntityPersistentTrajectorySpec(
                num_episodes=2,
                windows=5,
                history=5,
                horizon=6,
                num_robots=3,
                num_entities=4,
                active_entities=2,
                seed=53,
            )
        )
        batch = _flatten_entities(next(iter(DataLoader(dataset, batch_size=2))))
        self.assertEqual(batch["entity_history"].shape[:2], (8, 5))
        torch.testing.assert_close(
            batch["episode_id"], torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        )
        torch.testing.assert_close(
            batch["entity_slot"], torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        )

    def test_multi_entity_shared_team_actions_are_globally_consistent(self) -> None:
        dataset = MultiEntityPersistentTrajectoryEpisodeDataset(
            MultiEntityPersistentTrajectorySpec(
                num_episodes=1,
                windows=5,
                history=5,
                horizon=6,
                num_robots=3,
                num_entities=4,
                active_entities=2,
                shared_team_actions=True,
                seed=59,
            )
        )
        sample = dataset[0]
        entity_current = sample["entity_history"][:, :, -1]
        global_current = (
            sample["robot_current"][..., :2] + entity_current[:, :, None, :]
        )
        global_future = (
            sample["robot_candidates"][..., :2]
            + entity_current[:, :, None, None, :]
        )
        for entity in range(1, 4):
            torch.testing.assert_close(global_current[0], global_current[entity])
            torch.testing.assert_close(global_future[0], global_future[entity])
        self.assertTrue(
            torch.equal(sample["active_mask"].sum(dim=0), torch.full((5,), 2))
        )

    def test_shuffled_identity_filter_breaks_cross_window_association(self) -> None:
        likelihoods = torch.tensor(
            [
                [[4.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                [[0.0, 4.0, 0.0], [0.0, 4.0, 0.0], [0.0, 4.0, 0.0]],
                [[0.0, 0.0, 4.0], [0.0, 0.0, 4.0], [0.0, 0.0, 4.0]],
            ]
        )
        active = torch.ones(3, 3, dtype=torch.bool)
        prior = torch.full((3, 3), 1.0 / 3.0).log()
        shuffled_before, shuffled_after = _shuffled_identity_filter(
            likelihoods,
            active,
            episode_ids=torch.zeros(3, dtype=torch.long),
            entity_slots=torch.arange(3),
            prior_log_belief=prior,
        )
        self.assertLess(float(shuffled_before[0, 1, 0].exp()), 0.1)
        torch.testing.assert_close(
            shuffled_after.exp().sum(dim=-1), torch.ones(3, 3)
        )

    def test_mode_response_and_recurrent_models_forward_backward(self) -> None:
        samples = [self._dataset()[index] for index in range(3)]
        history = torch.stack([sample["entity_history"] for sample in samples])
        current = torch.stack([sample["robot_current"] for sample in samples])
        candidates = torch.stack([sample["robot_candidates"] for sample in samples])
        target = torch.stack([sample["target_future_vel"] for sample in samples])
        config = PersistentTrajectoryModelConfig(
            history=5,
            horizon=6,
            hidden_dim=24,
            dropout=0.0,
        )
        model = BeliefConditionedTeamResponseModel(config)
        output = model(history, current, candidates)
        self.assertEqual(output["mean"].shape, (3, 4, 3, 6, 2))
        likelihood = trajectory_mode_log_likelihood(output["mean"], output["log_std"], target)
        self.assertEqual(likelihood.shape, (3, 4, 3))
        belief = torch.softmax(torch.randn(3, 4, 3), dim=-1)
        self.assertEqual(belief_weighted_mean(output["mean"], belief).shape, target.shape)
        (-likelihood.mean()).backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

        recurrent = TrajectoryRecurrentBelief(
            TrajectoryRecurrentConfig(history=5, horizon=6, hidden_dim=16)
        )
        recurrent_output = recurrent(current, candidates, target)
        self.assertEqual(recurrent_output.shape, (3, 4, 3))
        (-recurrent_output[..., 0].mean()).backward()
        self.assertTrue(any(parameter.grad is not None for parameter in recurrent.parameters()))

    def test_mode_discrimination_loss_rewards_the_true_mode(self) -> None:
        likelihood = torch.tensor(
            [[[4.0, 1.0, -1.0], [2.0, 0.0, -2.0]]], requires_grad=True
        )
        losses = mode_conditioned_trajectory_loss(
            likelihood,
            torch.tensor([0]),
            coordinates=2,
            discrimination_weight=0.1,
            temperature=0.2,
        )
        self.assertGreater(float(losses["mode_discrimination"].detach()), 0.0)
        self.assertGreater(
            float(losses["loss"].detach()), float(losses["trajectory_nll"].detach())
        )
        losses["loss"].backward()
        self.assertLess(float(likelihood.grad[..., 0].mean()), 0.0)

    def test_mode_conditioned_loss_ignores_inactive_windows(self) -> None:
        likelihood = torch.tensor(
            [[[2.0, 0.0], [-100.0, 100.0]]], requires_grad=True
        )
        masked = mode_conditioned_trajectory_loss(
            likelihood,
            torch.tensor([0]),
            coordinates=2,
            observation_mask=torch.tensor([[True, False]]),
        )
        reference = mode_conditioned_trajectory_loss(
            likelihood[:, :1],
            torch.tensor([0]),
            coordinates=2,
        )
        torch.testing.assert_close(masked["trajectory_nll"], reference["trajectory_nll"])

    def test_executed_prefix_likelihood_ignores_unexecuted_suffix(self) -> None:
        mean = torch.zeros(2, 4, 3, 6, 2)
        log_std = torch.zeros_like(mean)
        target = torch.randn(2, 4, 6, 2)
        changed = target.clone()
        changed[..., 2:, :] += 100.0
        original = executed_prefix_mode_log_likelihood(
            mean, log_std, target, execute_steps=2
        )
        masked = executed_prefix_mode_log_likelihood(
            mean, log_std, changed, execute_steps=2
        )
        torch.testing.assert_close(original, masked)

    def test_recurrent_belief_ignores_unexecuted_response_suffix(self) -> None:
        samples = [self._dataset()[index] for index in range(2)]
        current = torch.stack([sample["robot_current"] for sample in samples])
        candidates = torch.stack([sample["robot_candidates"] for sample in samples])
        target = torch.stack([sample["target_future_vel"] for sample in samples])
        changed = target.clone()
        changed[..., 2:, :] += 100.0
        recurrent = TrajectoryRecurrentBelief(
            TrajectoryRecurrentConfig(
                history=5,
                horizon=6,
                hidden_dim=16,
                execute_steps=2,
            )
        ).eval()
        with torch.no_grad():
            original = recurrent(current, candidates, target)
            masked = recurrent(current, candidates, changed)
        torch.testing.assert_close(original, masked)

    def test_matched_recurrent_uses_entity_history_and_requires_it(self) -> None:
        samples = [self._dataset()[index] for index in range(2)]
        history = torch.stack([sample["entity_history"] for sample in samples])
        current = torch.stack([sample["robot_current"] for sample in samples])
        candidates = torch.stack([sample["robot_candidates"] for sample in samples])
        target = torch.stack([sample["target_future_vel"] for sample in samples])
        recurrent = TrajectoryRecurrentBelief(
            TrajectoryRecurrentConfig(
                history=5,
                horizon=6,
                hidden_dim=16,
                execute_steps=2,
                use_entity_history=True,
                dt=0.4,
            )
        )
        with self.assertRaises(ValueError):
            recurrent(current, candidates, target)
        output = recurrent(
            current,
            candidates,
            target,
            entity_history=history,
        )
        self.assertEqual(output.shape, (2, 4, 3))

    def test_entity_velocity_encoding_uses_configured_dt(self) -> None:
        torch.manual_seed(5)
        fast_sampling = BeliefConditionedTeamResponseModel(
            PersistentTrajectoryModelConfig(history=5, horizon=6, hidden_dim=24, dropout=0.0, dt=0.2)
        ).eval()
        slow_sampling = BeliefConditionedTeamResponseModel(
            PersistentTrajectoryModelConfig(history=5, horizon=6, hidden_dim=24, dropout=0.0, dt=0.4)
        ).eval()
        slow_sampling.load_state_dict(fast_sampling.state_dict())
        history = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
        with torch.no_grad():
            fast_context = fast_sampling._entity_context(history)
            slow_context = slow_sampling._entity_context(history)
        self.assertFalse(torch.equal(fast_context, slow_context))

    def test_closed_loop_candidates_share_state_and_step_advances_history(self) -> None:
        simulator = SyntheticClosedLoopSimulator(
            ClosedLoopSpec(history=5, horizon=6, num_candidates=8, execute_steps=2),
            episode_index=3,
        )
        pool = simulator.candidate_pool()
        self.assertEqual(pool.robot_candidates.shape, (8, 2, 6, 4))
        torch.testing.assert_close(
            pool.robot_current,
            pool.robot_current[0:1].expand_as(pool.robot_current),
        )
        old_history = simulator.history.copy()
        realized = simulator.step(pool, 0, PlanningCostConfig())
        self.assertEqual(realized.response.shape, (6, 2))
        self.assertFalse(torch.equal(torch.from_numpy(old_history), torch.from_numpy(simulator.history)))
        self.assertEqual(simulator.window, 1)

    def test_closed_loop_supports_three_robot_joint_candidates(self) -> None:
        simulator = SyntheticClosedLoopSimulator(
            ClosedLoopSpec(
                history=5,
                horizon=6,
                num_robots=3,
                num_candidates=8,
                execute_steps=2,
            ),
            episode_index=3,
        )
        pool = simulator.candidate_pool()
        self.assertEqual(pool.robot_current.shape, (8, 3, 4))
        self.assertEqual(pool.robot_candidates.shape, (8, 3, 6, 4))
        realized = simulator.step(pool, selected=0, cost=PlanningCostConfig())
        self.assertEqual(realized.response.shape, (6, 2))
        self.assertEqual(simulator.robot_positions.shape, (3, 2))

    def test_closed_loop_scene_and_response_families_are_finite(self) -> None:
        for scene_family in ("radial", "corridor", "crossing", "bottleneck"):
            for response_family in ("pincer", "speed", "lateral", "threshold"):
                simulator = SyntheticClosedLoopSimulator(
                    ClosedLoopSpec(
                        history=5,
                        horizon=6,
                        num_candidates=8,
                        execute_steps=2,
                        scene_family=scene_family,
                        response_family=response_family,
                    ),
                    episode_index=5,
                )
                pool = simulator.candidate_pool()
                outcomes = simulator.counterfactuals(pool, PlanningCostConfig())
                self.assertEqual(len(outcomes), 8)
                self.assertTrue(all(torch.isfinite(outcome.response).all() for outcome in outcomes))

    def test_closed_loop_counterfactuals_preserve_state_and_match_step(self) -> None:
        simulator = SyntheticClosedLoopSimulator(
            ClosedLoopSpec(
                history=5,
                horizon=6,
                num_robots=3,
                num_candidates=8,
                execute_steps=2,
            ),
            episode_index=7,
        )
        pool = simulator.candidate_pool()
        entity_position = simulator.entity_position.copy()
        robot_positions = simulator.robot_positions.copy()
        robot_velocities = simulator.robot_velocities.copy()
        history = simulator.history.copy()
        window = simulator.window

        counterfactuals = simulator.counterfactuals(pool, PlanningCostConfig())

        self.assertEqual(len(counterfactuals), 8)
        np.testing.assert_array_equal(simulator.entity_position, entity_position)
        np.testing.assert_array_equal(simulator.robot_positions, robot_positions)
        np.testing.assert_array_equal(simulator.robot_velocities, robot_velocities)
        np.testing.assert_array_equal(simulator.history, history)
        self.assertEqual(simulator.window, window)
        selected = 5
        realized = simulator.step(pool, selected, PlanningCostConfig())
        torch.testing.assert_close(
            realized.response,
            counterfactuals[selected].response,
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(realized.task_progress, counterfactuals[selected].task_progress)
        self.assertEqual(realized.safety_cost, counterfactuals[selected].safety_cost)
        self.assertEqual(realized.human_cost, counterfactuals[selected].human_cost)

    def test_closed_loop_batched_response_matches_scalar_reference(self) -> None:
        simulator = SyntheticClosedLoopSimulator(
            ClosedLoopSpec(
                history=5,
                horizon=6,
                num_robots=3,
                num_candidates=8,
                execute_steps=2,
            ),
            episode_index=11,
        )
        pool = simulator.candidate_pool()
        current = pool.robot_current.numpy()
        candidates = pool.robot_candidates.numpy()
        noise = np.random.default_rng(83).normal(
            0.0, simulator.spec.response_noise, size=(simulator.spec.horizon, 2)
        ).astype(np.float32)
        scalar = np.stack(
            [simulator._response(current[i], candidates[i], noise=noise) for i in range(8)]
        )
        batched = simulator._response_batch(current, candidates, noise=noise)
        np.testing.assert_array_equal(scalar, batched)

    def test_closed_loop_batched_counterfactuals_match_each_step(self) -> None:
        simulator = SyntheticClosedLoopSimulator(
            ClosedLoopSpec(
                history=5,
                horizon=6,
                num_robots=3,
                num_candidates=8,
                execute_steps=2,
            ),
            episode_index=13,
        )
        pool = simulator.candidate_pool()
        counterfactuals = simulator.counterfactuals(pool, PlanningCostConfig())
        realized = simulator.step(pool, selected=4, cost=PlanningCostConfig())
        torch.testing.assert_close(
            realized.response,
            counterfactuals[4].response,
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(realized.human_cost, counterfactuals[4].human_cost)

    def test_closed_loop_replay_and_candidate_hash_are_deterministic(self) -> None:
        spec = ClosedLoopSpec(
            history=5,
            horizon=6,
            num_robots=3,
            num_candidates=8,
            execute_steps=2,
            seed=71,
        )
        first = SyntheticClosedLoopSimulator(spec, episode_index=9)
        second = SyntheticClosedLoopSimulator(spec, episode_index=9)
        first_pool = first.candidate_pool()
        second_pool = second.candidate_pool()
        self.assertEqual(
            candidate_pool_sha256(first_pool), candidate_pool_sha256(second_pool)
        )
        first_realized = first.step(first_pool, selected=3, cost=PlanningCostConfig())
        second_realized = second.step(second_pool, selected=3, cost=PlanningCostConfig())
        torch.testing.assert_close(
            first_realized.response, second_realized.response, rtol=0.0, atol=0.0
        )
        self.assertEqual(
            candidate_pool_sha256(first.candidate_pool()),
            candidate_pool_sha256(second.candidate_pool()),
        )

        changed_pool = replace(
            first_pool,
            robot_candidates=first_pool.robot_candidates.clone(),
        )
        changed_pool.robot_candidates[0, 0, 0, 0] += 1.0
        self.assertNotEqual(
            candidate_pool_sha256(first_pool), candidate_pool_sha256(changed_pool)
        )

    def test_sage_candidate_adapter_preserves_joint_candidate_shape(self) -> None:
        class FakeSAGE:
            config = SimpleNamespace(diffusion=SimpleNamespace(horizon=6))

            def eval(self):
                return self

            def generate(self, batch, guided=True, step_stride=1):
                velocity = torch.zeros(batch.batch_size, batch.num_agents, 6, 2)
                velocity[:, :3, :, 0] = 0.5
                position = batch.current_pos[:, :, None] + torch.cumsum(
                    velocity * batch.dt, dim=2
                )
                return {"future_vel": velocity, "future_pos": position}

        adapter = SAGECandidateAdapter(
            FakeSAGE(),
            "cpu",
            SAGECandidateAdapterConfig(num_candidates=5, step_stride=2),
        )
        history = np.stack(
            [np.asarray([0.1 * step, 0.0], dtype=np.float32) for step in range(5)]
        )
        positions = np.asarray([[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        velocities = np.asarray([[-0.5, 0.0], [0.5, 0.0], [0.0, -0.5]], dtype=np.float32)
        pool = adapter.candidate_pool(
            history, positions, velocities, dt=0.4, sample_seed=17
        )
        self.assertEqual(pool.robot_current.shape, (5, 3, 4))
        self.assertEqual(pool.robot_candidates.shape, (5, 3, 6, 4))
        self.assertEqual(pool.activities.shape, (5, 3))
        torch.testing.assert_close(
            pool.robot_current,
            pool.robot_current[0:1].expand_as(pool.robot_current),
        )

    def test_sage_candidate_adapter_cache_is_content_addressed(self) -> None:
        class FakeSAGE:
            config = SimpleNamespace(diffusion=SimpleNamespace(horizon=4))

            def __init__(self):
                self.generate_calls = 0

            def eval(self):
                return self

            def generate(self, batch, guided=True, step_stride=1):
                self.generate_calls += 1
                velocity = torch.zeros(batch.batch_size, batch.num_agents, 4, 2)
                position = batch.current_pos[:, :, None] + torch.cumsum(
                    velocity * batch.dt, dim=2
                )
                return {"future_vel": velocity, "future_pos": position}

        model = FakeSAGE()
        adapter = SAGECandidateAdapter(
            model,
            "cpu",
            SAGECandidateAdapterConfig(
                num_candidates=2,
                cache_candidate_pools=True,
            ),
        )
        history = np.zeros((5, 2), dtype=np.float32)
        positions = np.asarray([[2.0, 0.0], [-2.0, 0.0]], dtype=np.float32)
        velocities = np.zeros_like(positions)
        first = adapter.candidate_pool(
            history, positions, velocities, dt=0.4, sample_seed=17
        )
        second = adapter.candidate_pool(
            history.copy(), positions.copy(), velocities.copy(), dt=0.4, sample_seed=17
        )
        self.assertIs(first, second)
        self.assertEqual(model.generate_calls, 1)
        changed = positions.copy()
        changed[0, 0] += 0.1
        adapter.candidate_pool(
            history, changed, velocities, dt=0.4, sample_seed=17
        )
        self.assertEqual(model.generate_calls, 2)

    def test_sage_candidate_adapter_batch_preserves_seeded_pool_hashes(self) -> None:
        class FakeSAGE:
            config = SimpleNamespace(
                diffusion=SimpleNamespace(horizon=4, num_steps=4),
            )
            generation_mode = "joint"

            def eval(self):
                return self

            def generate(self, batch, guided=True, step_stride=1, noise_schedules=None):
                if noise_schedules is None:
                    velocity = torch.randn(
                        batch.batch_size,
                        batch.num_agents,
                        4,
                        2,
                    )
                else:
                    velocity = noise_schedules[0]
                position = batch.current_pos[:, :, None] + torch.cumsum(
                    velocity * batch.dt, dim=2
                )
                return {"future_vel": velocity, "future_pos": position}

        adapter = SAGECandidateAdapter(
            FakeSAGE(),
            "cpu",
            SAGECandidateAdapterConfig(
                num_candidates=3,
                oversample_factor=2,
                cache_candidate_pools=False,
            ),
        )
        history = np.zeros((5, 2), dtype=np.float32)
        positions = np.asarray([[2.0, 0.0], [-2.0, 0.0]], dtype=np.float32)
        velocities = np.zeros_like(positions)
        scenes = [
            (history, positions, velocities, 0.4),
            (history + 0.25, positions, velocities, 0.4),
        ]
        seeds = [17, 23]
        scalar = [
            adapter.candidate_pool(*scene, sample_seed=seed)
            for scene, seed in zip(scenes, seeds)
        ]
        batched = adapter.candidate_pools(scenes, seeds)
        self.assertEqual(len(batched), len(scalar))
        for expected, actual in zip(scalar, batched):
            torch.testing.assert_close(
                expected.robot_candidates,
                actual.robot_candidates,
                rtol=1e-5,
                atol=1e-5,
            )

    def test_sage_candidate_adapter_prefilters_oversampled_task_progress(self) -> None:
        class FakeSAGE:
            config = SimpleNamespace(diffusion=SimpleNamespace(horizon=4))

            def eval(self):
                return self

            def generate(self, batch, guided=True, step_stride=1):
                direction = -batch.current_pos[:, :2]
                direction = direction / torch.linalg.vector_norm(
                    direction, dim=-1, keepdim=True
                ).clamp_min(1e-6)
                scales = torch.tensor([-1.0, 0.0, 0.5, 1.0])[:, None, None]
                velocity = torch.zeros(batch.batch_size, batch.num_agents, 4, 2)
                velocity[:, :2] = direction[:, :, None] * scales[:, :, None]
                position = batch.current_pos[:, :, None] + torch.cumsum(
                    velocity * batch.dt, dim=2
                )
                return {"future_vel": velocity, "future_pos": position}

        adapter = SAGECandidateAdapter(
            FakeSAGE(),
            "cpu",
            SAGECandidateAdapterConfig(
                num_candidates=2,
                oversample_factor=2,
                task_prefilter=True,
                task_prefix_steps=2,
                goal_direction="entity",
            ),
        )
        history = np.zeros((5, 2), dtype=np.float32)
        positions = np.asarray([[2.0, 0.0], [-2.0, 0.0]], dtype=np.float32)
        velocities = np.zeros_like(positions)
        pool = adapter.candidate_pool(
            history, positions, velocities, dt=0.4, sample_seed=17
        )
        relative = pool.robot_current[:, :, :2]
        toward = -relative / torch.linalg.vector_norm(
            relative, dim=-1, keepdim=True
        ).clamp_min(1e-6)
        displacement = pool.robot_candidates[:, :, 1, :2] - relative
        progress = (displacement * toward).sum(dim=-1).mean(dim=-1)
        self.assertTrue((progress > 0.0).all())

    def test_sage_candidate_adapter_can_add_task_anchor(self) -> None:
        class FakeSAGE:
            config = SimpleNamespace(diffusion=SimpleNamespace(horizon=4))

            def eval(self):
                return self

            def generate(self, batch, guided=True, step_stride=1):
                velocity = torch.zeros(batch.batch_size, batch.num_agents, 4, 2)
                position = batch.current_pos[:, :, None] + torch.cumsum(
                    velocity * batch.dt, dim=2
                )
                return {"future_vel": velocity, "future_pos": position}

        adapter = SAGECandidateAdapter(
            FakeSAGE(),
            "cpu",
            SAGECandidateAdapterConfig(
                num_candidates=2,
                oversample_factor=2,
                task_prefilter=True,
                task_prefix_steps=2,
                goal_direction="entity",
                include_task_anchor=True,
                task_anchor_speed_scales=(0.4, 0.7, 1.0, 1.3),
            ),
        )
        history = np.zeros((5, 2), dtype=np.float32)
        positions = np.asarray([[2.0, 0.0], [-2.0, 0.0]], dtype=np.float32)
        velocities = np.zeros_like(positions)
        pool = adapter.candidate_pool(
            history, positions, velocities, dt=0.4, sample_seed=17
        )
        relative = pool.robot_current[:, :, :2]
        toward = -relative / torch.linalg.vector_norm(
            relative, dim=-1, keepdim=True
        ).clamp_min(1e-6)
        displacement = pool.robot_candidates[:, :, 1, :2] - relative
        progress = (displacement * toward).sum(dim=-1).mean(dim=-1)
        self.assertEqual(pool.robot_candidates.shape[0], 2)
        self.assertTrue((progress > 0.0).any())

    def test_closed_loop_costs_are_finite_for_joint_candidate_pool(self) -> None:
        simulator = SyntheticClosedLoopSimulator(
            ClosedLoopSpec(history=5, horizon=6, num_candidates=8, execute_steps=2),
            episode_index=4,
        )
        pool = simulator.candidate_pool()
        mode_means = torch.randn(8, 3, 6, 2)
        reference_mode_means = mode_means * 0.2
        costs = expected_candidate_costs(
            mode_means,
            torch.tensor([0.2, 0.3, 0.5]),
            pool.robot_current,
            pool.robot_candidates,
            reference_mode_means,
            0.4,
            PlanningCostConfig(),
        )
        self.assertEqual(costs["total"].shape, (8,))
        self.assertTrue(torch.isfinite(costs["total"]).all())

    def test_closed_loop_task_cost_uses_configured_execution_prefix(self) -> None:
        current = torch.tensor([[[2.0, 0.0, 0.0, 0.0]]]).expand(2, -1, -1).clone()
        candidates = torch.zeros(2, 1, 4, 4)
        candidates[:, :, :, 0] = torch.tensor(
            [[1.5, 1.0, 0.5, 0.0], [2.5, 2.0, 1.0, 0.0]]
        )[:, None]
        mode_means = torch.zeros(2, 3, 4, 2)
        costs = expected_candidate_costs(
            mode_means,
            torch.full((3,), 1.0 / 3.0),
            current,
            candidates,
            mode_means,
            0.4,
            PlanningCostConfig(task_steps=2),
        )
        self.assertLess(float(costs["task"][0]), float(costs["task"][1]))
        self.assertLess(float(costs["task"][0]), 0.0)
        self.assertEqual(float(costs["task"][1]), 0.0)

    def test_closed_loop_reference_output_uses_stationary_action(self) -> None:
        torch.manual_seed(79)
        config = PersistentTrajectoryModelConfig(
            history=5,
            horizon=6,
            num_robots=2,
            hidden_dim=24,
            dropout=0.0,
        )
        model = BeliefConditionedTeamResponseModel(config).eval()
        history = torch.randn(3, 1, 5, 2)
        current = torch.randn(3, 2, 4)
        candidates = torch.randn(3, 2, 6, 4)
        mask = torch.ones(3, 1, 2, dtype=torch.bool)
        with torch.no_grad():
            output, reference_output = _candidate_and_reference_outputs(
                model, history, current, candidates, mask, True
            )
            stationary = stationary_reference_candidates(current[:, None], 6)
            expected_reference = model(
                history,
                current[:, None],
                stationary,
                robot_mask=mask,
                include_pair_interactions=True,
            )
        torch.testing.assert_close(
            reference_output["mean"], expected_reference["mean"]
        )
        self.assertFalse(torch.equal(output["mean"], reference_output["mean"]))

    def test_closed_loop_costs_mix_modes_before_risk_reduction(self) -> None:
        robot_current = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        robot_candidates = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
        mode_means = torch.tensor(
            [
                [
                    [[1.0, 0.0]],
                    [[-1.0, 0.0]],
                ]
            ]
        )
        reference = torch.zeros_like(mode_means)
        costs = expected_candidate_costs(
            mode_means,
            torch.tensor([0.5, 0.5]),
            robot_current,
            robot_candidates,
            reference,
            1.0,
            PlanningCostConfig(task_weight=0.0, safety_weight=1.0, human_weight=0.0),
        )
        collapsed_response = (mode_means * torch.tensor([0.5, 0.5])[None, :, None, None]).sum(dim=1)
        collapsed_distance = torch.linalg.vector_norm(
            robot_candidates[..., :2] - torch.cumsum(collapsed_response, dim=1)[:, None],
            dim=-1,
        ).flatten(start_dim=1).min(dim=1).values
        collapsed_safety = torch.exp(-collapsed_distance)
        self.assertGreater(float(costs["safety"][0]), float(collapsed_safety[0]))

    def test_persistent_model_can_disable_pair_residual(self) -> None:
        samples = [self._dataset()[index] for index in range(2)]
        history = torch.stack([sample["entity_history"] for sample in samples])
        current = torch.stack([sample["robot_current"] for sample in samples])
        candidates = torch.stack([sample["robot_candidates"] for sample in samples])
        model = BeliefConditionedTeamResponseModel(
            PersistentTrajectoryModelConfig(history=5, horizon=6, hidden_dim=24, dropout=0.0)
        ).eval()
        torch.nn.init.normal_(model.pair_decoder[-1].weight)
        torch.nn.init.normal_(model.pair_decoder[-1].bias)
        with torch.no_grad():
            additive = model(history, current, candidates, include_pair_interactions=False)
            joint = model(history, current, candidates, include_pair_interactions=True)
        self.assertEqual(additive["mean"].shape, joint["mean"].shape)
        self.assertFalse(torch.equal(additive["mean"], joint["mean"]))

    def test_tracer_response_model_uses_neighbors_and_preserves_robot_permutation(self) -> None:
        dataset = MultiEntityPersistentTrajectoryEpisodeDataset(
            MultiEntityPersistentTrajectorySpec(
                num_episodes=1,
                windows=5,
                history=5,
                horizon=6,
                num_robots=3,
                num_entities=4,
                active_entities=3,
                seed=901,
            )
        )
        batch = _flatten_entities(next(iter(DataLoader(dataset, batch_size=1))))
        model = BeliefConditionedTeamResponseModel(
            PersistentTrajectoryModelConfig(
                history=5,
                horizon=6,
                num_robots=3,
                hidden_dim=24,
                num_heads=4,
                architecture="tracer",
                dropout=0.0,
            )
        ).eval()
        robot_mask = torch.ones(
            batch["entity_history"].shape[0],
            batch["entity_history"].shape[1],
            3,
            dtype=torch.bool,
        )
        with torch.no_grad():
            original = model(
                batch["entity_history"],
                batch["robot_current"],
                batch["robot_candidates"],
                robot_mask=robot_mask,
                neighbor_history=batch["neighbor_history"],
                neighbor_mask=batch["neighbor_mask"],
            )
            permutation = torch.tensor([2, 0, 1])
            permuted = model(
                batch["entity_history"],
                batch["robot_current"][:, :, permutation],
                batch["robot_candidates"][:, :, permutation],
                robot_mask=robot_mask[:, :, permutation],
                neighbor_history=batch["neighbor_history"],
                neighbor_mask=batch["neighbor_mask"],
            )
        torch.testing.assert_close(original["mean"], permuted["mean"], atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.isfinite(original["log_std"]).all())

    def test_probabilistic_safety_filters_candidates_before_selection(self) -> None:
        mode_means = torch.zeros(2, 2, 4, 2)
        mode_log_stds = torch.full_like(mode_means, -4.0)
        robot_current = torch.tensor(
            [[[3.0, 0.0, 0.0, 0.0]], [[3.0, 0.0, 0.0, 0.0]]]
        )
        robot_candidates = torch.zeros(2, 1, 4, 4)
        robot_candidates[0, 0, :, 0] = 3.0
        costs = expected_candidate_costs(
            mode_means,
            torch.tensor([0.5, 0.5]),
            robot_current,
            robot_candidates,
            mode_means,
            0.4,
            PlanningCostConfig(
                task_weight=0.0,
                safety_weight=1.0,
                human_weight=0.0,
                violation_probability_alpha=0.1,
                risk_samples=16,
            ),
            mode_log_stds=mode_log_stds,
        )
        self.assertTrue(bool(costs["safe_mask"][0]))
        self.assertFalse(bool(costs["safe_mask"][1]))
        self.assertTrue(torch.isinf(costs["total"][1]))

    def test_kendall_rank_handles_rejected_infinite_candidates(self) -> None:
        value = _kendall_tau_b(
            np.asarray([float("inf"), float("inf"), 1.0]),
            np.asarray([2.0, 3.0, 1.0]),
        )
        self.assertTrue(np.isfinite(value))


if __name__ == "__main__":
    unittest.main()
