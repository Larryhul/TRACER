from sage.experiments.assess_reliable_navigation import assess_reliable_navigation


def rows():
    return [dict(seed=2027, case=case, initial_scene_hash=str(case), environment_dt=.25,
                 robot_kinematics="unicycle", robot_initial_heading=0., policy="tracer-persistent",
                 success=1., collision_free_success=1., collision=0., wall_collision=0., timeout=0.,
                 total_time=5., task_progress=.95, stagnation_time=.5,
                 human_velocity_variation=.2, terminal_goal_distance=.2) for case in range(32)]


def test_faster_but_less_safe_cannot_pass_reliability_screen():
    parent, candidate = rows(), rows()
    for row in candidate:
        row["total_time"] = 4.
    candidate[0].update(collision_free_success=0., collision=1.)
    result = assess_reliable_navigation(parent, candidate)
    assert not result["development_screen_passed"]
    assert result["worsened_safety_or_success_cases"] == [0]


def test_reliability_gain_does_not_require_faster_completion():
    parent, candidate = rows(), rows()
    parent[0].update(success=0., collision_free_success=0., timeout=1.)
    for row in candidate:
        row["total_time"] = 6.
    result = assess_reliable_navigation(parent, candidate)
    assert result["development_screen_passed"]
    assert result["first_improved_priority_tier"] == 0
    assert result["independent_confirmation_required"]


def test_time_gain_does_not_offset_worse_human_perturbation():
    parent, candidate = rows(), rows()
    for row in candidate:
        row.update(human_velocity_variation=.3, total_time=4.)
    assert not assess_reliable_navigation(parent, candidate)["development_screen_passed"]


def test_four_scene_pilot_and_missing_telemetry_do_not_pass():
    parent, candidate = rows()[:4], rows()[:4]
    for row in candidate:
        row["total_time"] = 4.
    assert not assess_reliable_navigation(parent, candidate)["development_screen_passed"]
    del candidate[0]["human_velocity_variation"]
    result = assess_reliable_navigation(parent, candidate)
    assert result["reliability_status"] == "INCOMPLETE_TELEMETRY"
    assert not result["development_screen_passed"]


def test_success_saturates_progress_inside_goal_radius():
    parent, candidate = rows(), rows()
    candidate[0].update(task_progress=.90, terminal_goal_distance=.29, human_velocity_variation=.1)
    result = assess_reliable_navigation(parent, candidate)
    assert result["metrics"]["task_progress"]["candidate_minus_parent"] < 0.
    assert result["metrics"]["goal_shortfall"]["candidate_minus_parent"] == 0.
    assert result["development_screen_passed"]
    assert result["first_improved_priority_tier"] == 2
