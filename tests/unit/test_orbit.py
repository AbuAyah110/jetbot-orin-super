from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from jetbot_agent.robot_loop.orbit import (
    OrbitConfig,
    bearing_deg,
    run_orbit,
)


@dataclass
class FakeSighting:
    visible: bool
    center_x: float
    pixels: int


class OrbitWorld:
    """Flat kinematic model: target at the origin, robot on the plane.

    Exists so the controller can be checked against geometry rather than
    against a live chassis. A live run costs a minute and cannot be repeated
    exactly; this can.
    """

    def __init__(
        self,
        distance: float = 1.0,
        step: float = 0.35,
        config: OrbitConfig | None = None,
        pixels_at_unit_distance: int = 29000,
        turn_error: float = 0.0,
    ):
        self.config = config or OrbitConfig()
        # Start directly south of the target, facing it.
        self.x = 0.0
        self.y = -distance
        self.heading = math.pi / 2
        self.step = step
        self.pixels_at_unit_distance = pixels_at_unit_distance
        self.turn_error = turn_error
        self.turns = 0
        self.drives = 0

    @property
    def distance(self) -> float:
        return math.hypot(self.x, self.y)

    def _bearing(self) -> float:
        to_target = math.atan2(-self.y, -self.x)
        return (to_target - self.heading + math.pi) % (2 * math.pi) - math.pi

    def locate(self, _phase: str):
        bearing = math.degrees(self._bearing())
        half_fov = self.config.hfov_deg / 2.0
        if abs(bearing) > half_fov:
            return FakeSighting(visible=False, center_x=-1.0, pixels=0)
        width = self.config.frame_width
        center_x = width / 2.0 - bearing / self.config.hfov_deg * width
        pixels = int(self.pixels_at_unit_distance / max(self.distance, 0.05) ** 2)
        return FakeSighting(visible=True, center_x=center_x, pixels=pixels)

    def turn(self, direction: str, pulses: int) -> None:
        self.turns += pulses
        amount = math.radians(pulses * self.config.pulse_deg * (1.0 + self.turn_error))
        self.heading += amount if direction == 'left' else -amount

    def forward(self) -> None:
        self.drives += 1
        self.x += self.step * math.cos(self.heading)
        self.y += self.step * math.sin(self.heading)

    def orbit_angle_travelled(self) -> float:
        """Ground truth: how far around the target the robot actually moved."""
        return math.degrees(
            (math.atan2(self.y, self.x) - math.atan2(-1.0, 0.0)) % (2 * math.pi)
        )


def test_bearing_is_positive_when_the_target_sits_left_of_centre():
    config = OrbitConfig()

    assert bearing_deg(0.0, config) > 0
    assert bearing_deg(config.frame_width, config) < 0
    assert bearing_deg(config.frame_width / 2.0, config) == pytest.approx(0.0)


def test_pure_rotation_does_not_count_as_orbit_progress():
    # Turning right moves the target left by the same angle, so the heading and
    # bearing terms must cancel exactly. An earlier design counted rotation as
    # progress and would have reported success without leaving the spot.
    world = OrbitWorld()
    config = OrbitConfig(target_deg=360.0, max_cycles=1)
    world.forward = lambda: None  # rotation only

    result = run_orbit(
        locate=world.locate, turn=world.turn, forward=world.forward, config=config
    )

    assert result.cycles == 1
    assert result.orbit_deg == pytest.approx(0.0, abs=0.5)


def test_orbit_reaches_behind_the_target_and_agrees_with_ground_truth():
    world = OrbitWorld(distance=1.0, step=0.35)

    result = run_orbit(
        locate=world.locate, turn=world.turn, forward=world.forward
    )

    assert result.aborted == ''
    assert result.reached_behind is True
    # The reported angle is the sum of commanded turns and measured bearings, so
    # against exact geometry it should equal the true angle travelled, not merely
    # exceed the threshold.
    assert result.orbit_deg == pytest.approx(world.orbit_angle_travelled(), abs=2.0)
    assert world.drives > 0


def test_orbit_holds_its_distance_rather_than_spiralling_in():
    world = OrbitWorld(distance=1.0, step=0.35)

    result = run_orbit(
        locate=world.locate, turn=world.turn, forward=world.forward
    )

    assert result.aborted == ''
    # Contact is unrecoverable without a bumper, so closing in is the failure
    # that matters most.
    assert world.distance > 0.5


def test_orbit_aborts_when_the_target_leaves_the_view():
    world = OrbitWorld()
    calls = {'n': 0}

    def flaky_locate(phase):
        calls['n'] += 1
        if calls['n'] > 2:
            return FakeSighting(visible=False, center_x=-1.0, pixels=0)
        return world.locate(phase)

    result = run_orbit(
        locate=flaky_locate, turn=world.turn, forward=world.forward
    )

    assert result.aborted == 'target_lost'
    assert result.reached_behind is False


def test_orbit_aborts_before_closing_on_the_target():
    world = OrbitWorld(distance=1.0, step=0.35)
    calls = {'n': 0}

    def closing_locate(phase):
        # Grow the target after the baseline is taken, as an inward spiral
        # would. Contact is unrecoverable without a bumper, so this is the
        # abort that matters most.
        sighting = world.locate(phase)
        calls['n'] += 1
        scale = 1 if calls['n'] == 1 else 4
        return FakeSighting(
            visible=sighting.visible,
            center_x=sighting.center_x,
            pixels=sighting.pixels * scale,
        )

    result = run_orbit(
        locate=closing_locate, turn=world.turn, forward=world.forward
    )

    assert result.aborted == 'closing_on_target'
    assert result.reached_behind is False


def test_orbit_refuses_to_start_without_a_sighting():
    result = run_orbit(
        locate=lambda _phase: FakeSighting(False, -1.0, 0),
        turn=lambda _d, _p: pytest.fail('must not move'),
        forward=lambda: pytest.fail('must not move'),
    )

    assert result.aborted == 'target_not_located'
    assert result.cycles == 0


def test_orbit_stops_immediately_on_request():
    world = OrbitWorld()

    result = run_orbit(
        locate=world.locate,
        turn=world.turn,
        forward=world.forward,
        should_stop=lambda: True,
    )

    assert result.aborted == 'stop_requested'
    assert world.drives == 0


def test_orbit_degrades_safely_when_the_turn_pulse_is_miscalibrated():
    # Degrees per pulse was measured, not derived, so a wrong value must not
    # produce a collision. At 25% error the simulated orbit drifts outward and
    # the range guard stops it at about 144 degrees, which is the outcome to
    # want: short of the goal, reported honestly, nowhere near the target.
    world = OrbitWorld(distance=1.0, step=0.35, turn_error=0.25)

    result = run_orbit(
        locate=world.locate, turn=world.turn, forward=world.forward
    )

    assert result.aborted in {'', 'cycle_budget_exhausted', 'target_too_far'}
    assert world.distance > 0.5
    if result.aborted:
        assert result.reached_behind is False


def test_acquire_sweeps_until_the_target_is_found_and_centres_it():
    from jetbot_agent.robot_loop.orbit import acquire_target

    # Start facing away from the target so a sweep is required.
    world = OrbitWorld()
    world.heading = math.pi / 2 + math.radians(120)

    seen = acquire_target(locate=world.locate, turn=world.turn)

    assert seen is not None and seen.visible
    config = OrbitConfig()
    assert abs(bearing_deg(seen.center_x, config)) < config.pulse_deg
    assert world.drives == 0


def test_acquire_gives_up_after_one_revolution():
    from jetbot_agent.robot_loop.orbit import acquire_target

    world = OrbitWorld()
    turns = []

    def never_visible(_phase):
        return FakeSighting(False, -1.0, 0)

    seen = acquire_target(
        locate=never_visible,
        turn=lambda d, p: turns.append((d, p)),
        max_pulses=16,
    )

    assert seen is None
    # A full revolution is about 15.5 pulses; spinning past that is pointless.
    assert len(turns) == 16
    assert world.drives == 0


def test_acquire_prefers_the_strongest_sighting_over_the_first_one():
    from jetbot_agent.robot_loop.orbit import acquire_target

    # Reproduces a live failure: a 292 px red artefact appeared first and was
    # locked onto while the actual truck, tens of thousands of pixels, sat
    # further round the sweep.
    frames = [
        FakeSighting(False, -1.0, 0),
        FakeSighting(True, 28.0, 292),
        FakeSighting(True, 224.0, 26000),
        FakeSighting(True, 300.0, 400),
    ]
    calls = []
    turns = []

    def scripted_locate(phase):
        calls.append(phase)
        if phase == 'acquire_best' or phase == 'acquire_centred':
            return FakeSighting(True, 224.0, 26000)
        index = min(len(calls) - 1, len(frames) - 1)
        return frames[index]

    seen = acquire_target(
        locate=scripted_locate,
        turn=lambda d, p: turns.append((d, p)),
        strong_pixels=4000,
    )

    assert seen is not None
    assert seen.pixels == 26000
    # It must turn back to the strong viewpoint, not stay where it stopped.
    assert ('left', 0) not in turns
