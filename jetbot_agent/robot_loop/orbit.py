"""Closed-loop orbit around a colour-grounded target.

Getting *behind* an object is a different problem from stepping aside from it:
it needs a measure of how far around the object the robot has actually
travelled, and this chassis has no odometry, no depth, and no working obstacle
gate.

The usable identity: the robot's angular position around the target is the
direction from the target to the robot, which equals the robot's heading plus
the target's bearing in the camera, plus a constant. So

    orbit angle travelled = (change in heading) + (change in bearing)

Heading change comes from calibrated turn pulses; bearing change is measured
directly from where the target sits in the frame. Pure rotation contributes
nothing, because the two terms cancel: turning right moves the target left by
the same angle. Driving forward while the target is off to one side is what
actually accumulates orbit angle. That is why this controller holds the target
*off* centre rather than centred, and why it does not depend on re-centring
landing exactly on a pulse boundary.

Two earlier designs failed against the simulator in ``tests/unit/test_orbit.py``
and are worth recording:

1. "Re-centre, turn out one pulse, drive." With a 23 degree pulse the
   re-centring rounded straight back to one pulse, so net heading change per
   cycle was zero and the robot circled forever without progressing.
2. "Hold the target 22 degrees off centre and drive." A target visible at 22
   degrees leaves ``cos(22) = 0.93`` of the velocity pointing at it, so the
   robot spirals inward and closes on the object.

The second failure is geometric, not a tuning problem: orbiting means driving
perpendicular to the target, and a 69 degree forward camera cannot see anything
at 90 degrees. Circling therefore *requires* short moves during which the target
is out of view. Each cycle turns away until the target is roughly abeam, drives
one blind tangential pulse, then turns back by the same amount to re-acquire it
and measure the progress. Aiming a little beyond abeam keeps the radius from
shrinking.

Those blind pulses carry no obstacle protection whatsoever. Nothing on this
robot does; the monocular path gate was measured to be a constant. The cycle is
kept short and the radius is watched through apparent area so a failure is a
stopped robot rather than a collision.

Calibration, measured on this chassis at the calibrated duty with a 0.15 s
pulse: the target crossed the frame at 142-150 px per pulse, and a full
revolution took about 15.5 pulses. That gives roughly 23 degrees per pulse and
a horizontal field of view near 69 degrees. The two agree, which is what lets a
pixel error be converted into a pulse count.

This module never opens I2C and never captures a frame. Motion and perception
arrive as callables so the controller can be tested against simulated geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol


class Sighting(Protocol):
    """Minimal view of ``color_grounding.ColorGrounding``."""

    visible: bool
    center_x: float
    pixels: int


@dataclass(frozen=True)
class OrbitConfig:
    pulse_deg: float = 23.0
    hfov_deg: float = 69.0
    frame_width: int = 448
    target_deg: float = 180.0
    # Simulated across tangential step sizes of 0.25 to 0.5 of the target
    # distance, a half orbit took 8 to 15 cycles. 18 leaves margin for the slow
    # end without letting a misbehaving run wander indefinitely.
    max_cycles: int = 18
    # Where to point relative to the target before the blind pulse. 90 degrees
    # is exactly abeam; a little beyond it gives the drive a slight outward
    # component so the radius holds instead of decaying. Simulated at 95 the
    # radius stays within a few percent over a half orbit.
    tangent_aim_deg: float = 95.0
    # Adaptive trim: apparent area is the only range proxy available, so aim
    # further out when the target grows and further in when it shrinks.
    aim_trim_deg: float = 12.0
    min_turn_pulses: int = 2
    max_turn_pulses: int = 5
    # Orbit progress shows up as a growing bearing, so turning back by the same
    # amount as the turn out walks the target out of the frame within a few
    # cycles. Turn back far enough to cancel both the current bearing error and
    # the progress the next drive is expected to add. A fixed one-pulse extra
    # was tried first and drifted the bearing negative by 4-7 degrees a cycle
    # until the target was lost. The accumulator is correct for any asymmetry,
    # so this only steers where the target sits, never the reported angle.
    max_return_extra_pulses: int = 2
    # Apparent area guards the radius, but loosely, because it is a poor range
    # proxy for anything that is not round. Measured circling a toy truck: the
    # matched area ran 20500, 4077, 8209, 2649 across three cycles as the
    # silhouette turned from face-on to side-on, and a 0.12 floor aborted the
    # run at 22 degrees for no real reason. These bounds now only catch gross
    # failures - a target filling the view, or one reduced to noise.
    max_pixel_growth: float = 3.0
    min_pixel_fraction: float = 0.04


@dataclass
class OrbitResult:
    cycles: int = 0
    orbit_deg: float = 0.0
    aborted: str = ''
    log: List[dict] = field(default_factory=list)

    @property
    def reached_behind(self) -> bool:
        return not self.aborted and self.orbit_deg >= 180.0


def bearing_deg(center_x: float, config: OrbitConfig) -> float:
    """Target bearing in degrees, positive when the target is left of centre."""
    offset = config.frame_width / 2.0 - center_x
    return offset / config.frame_width * config.hfov_deg


def _turn_deg(direction: str, pulses: int, config: OrbitConfig) -> float:
    """Leftward heading change is positive."""
    return (1.0 if direction == 'left' else -1.0) * pulses * config.pulse_deg


def _return_pulses(
    *,
    bearing: float,
    out_pulses: int,
    progress: float,
    config: OrbitConfig,
) -> int:
    """Left pulses that put the target back near centre after the blind pulse.

    After turning out and driving, the bearing is roughly
    ``bearing + out_pulses * pulse_deg + progress``. Turning back by the turn-out
    alone leaves the progress uncancelled, which is what walked the target out of
    frame in an earlier version.
    """
    extra = int(round((bearing + progress) / config.pulse_deg))
    extra = max(0, min(config.max_return_extra_pulses, extra))
    return out_pulses + extra


def _aim_trim(*, sighting_pixels: int, baseline: int, config: OrbitConfig) -> float:
    """Nudge the aim outward when the target grows, inward when it shrinks.

    Apparent area is the only range signal available, and it is noisy, so this
    is a two-sided nudge rather than a proportional controller.
    """
    # Wide thresholds: silhouette change during an orbit moves the area far more
    # than range does, so a tight trim would just chase aspect noise.
    if sighting_pixels > baseline * 1.8:
        return config.aim_trim_deg
    if sighting_pixels < baseline * 0.4:
        return -config.aim_trim_deg
    return 0.0


def acquire_target(
    *,
    locate: Callable[[str], Optional[Sighting]],
    turn: Callable[[str, int], None],
    config: Optional[OrbitConfig] = None,
    max_pulses: int = 16,
    strong_pixels: int = 4000,
) -> Optional[Sighting]:
    """Sweep in place, then face the strongest sighting and centre it.

    A full revolution measured about 15.5 pulses, so a 16-pulse budget looks
    everywhere once and gives up rather than spinning on the spot forever.
    Centring means the orbit starts from a known bearing.

    Taking the *first* sighting above threshold was tried live and locked onto a
    292 px red artefact at the frame edge while the actual truck, tens of
    thousands of pixels, sat elsewhere in the room. Keeping the strongest
    sighting across the sweep fixes that. A clearly strong sighting ends the
    sweep early so the ordinary case stays quick.
    """
    config = config or OrbitConfig()
    best_index = -1
    best_pixels = 0
    index = 0

    seen = locate('acquire_0')
    if seen is not None and seen.visible:
        best_index, best_pixels = 0, seen.pixels
    while best_pixels < strong_pixels and index < max_pulses:
        turn('right', 1)
        index += 1
        seen = locate('acquire_{0}'.format(index))
        if seen is not None and seen.visible and seen.pixels > best_pixels:
            best_index, best_pixels = index, seen.pixels
    if best_index < 0:
        return None

    # Turn back to the viewpoint that saw the target best. Only possible
    # because degrees per pulse is calibrated.
    if index - best_index:
        turn('left', index - best_index)
    seen = locate('acquire_best')
    if seen is None or not seen.visible:
        return None

    bearing = bearing_deg(seen.center_x, config)
    correction = int(round(bearing / config.pulse_deg))
    if correction:
        # Turning left decreases the bearing, so a target left of centre
        # (positive bearing) is centred by turning left.
        turn('left' if correction > 0 else 'right', abs(correction))
        seen = locate('acquire_centred')
    if seen is None or not seen.visible:
        return None
    return seen


def run_orbit(
    *,
    locate: Callable[[str], Optional[Sighting]],
    turn: Callable[[str, int], None],
    forward: Callable[[], None],
    config: Optional[OrbitConfig] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> OrbitResult:
    """Circle a target until the robot is behind it, or stop and say why.

    ``locate`` takes a phase label, captures a settled frame, and returns the
    current sighting or ``None``. Losing the target aborts: a robot that cannot
    see what it is circling must not keep driving around it.
    """
    config = config or OrbitConfig()
    result = OrbitResult()
    stop = should_stop or (lambda: False)

    first = locate('orbit_start')
    if first is None or not first.visible:
        result.aborted = 'target_not_located'
        return result
    baseline_pixels = max(1, first.pixels)
    bearing = bearing_deg(first.center_x, config)
    bearing_pixels = first.pixels
    progress_est = config.pulse_deg

    while result.orbit_deg < config.target_deg and result.cycles < config.max_cycles:
        if stop():
            result.aborted = 'stop_requested'
            return result

        # Aim past abeam so the blind pulse runs tangentially. Turning right
        # pans the camera right, which moves the target left in the image and so
        # increases its bearing.
        aim = config.tangent_aim_deg + _aim_trim(
            sighting_pixels=bearing_pixels, baseline=baseline_pixels, config=config
        )
        pulses = int(round((aim - bearing) / config.pulse_deg))
        pulses = max(config.min_turn_pulses, min(config.max_turn_pulses, pulses))

        turn('right', pulses)
        result.orbit_deg += _turn_deg('right', pulses, config)
        forward()
        back = _return_pulses(
            bearing=bearing, out_pulses=pulses, progress=progress_est, config=config
        )
        turn('left', back)
        result.orbit_deg += _turn_deg('left', back, config)

        before = result.orbit_deg
        sighting = locate('orbit_cycle_{0}'.format(result.cycles))
        if sighting is None or not sighting.visible:
            result.aborted = 'target_lost'
            return result
        if sighting.pixels > baseline_pixels * config.max_pixel_growth:
            result.aborted = 'closing_on_target'
            return result
        if sighting.pixels < baseline_pixels * config.min_pixel_fraction:
            result.aborted = 'target_too_far'
            return result

        new_bearing = bearing_deg(sighting.center_x, config)
        result.orbit_deg += new_bearing - bearing
        bearing = new_bearing
        bearing_pixels = sighting.pixels
        # Progress achieved this cycle, used to predict the next turn back.
        progress_est = result.orbit_deg - before + _turn_deg('left', back, config) \
            + _turn_deg('right', pulses, config)
        progress_est = max(0.0, progress_est)
        result.cycles += 1
        result.log.append(
            {
                'cycle': result.cycles,
                'aim_deg': round(aim, 1),
                'out_pulses': pulses,
                'back_pulses': back,
                'orbit_deg': round(result.orbit_deg, 1),
                'bearing_deg': round(bearing, 1),
                'pixels': sighting.pixels,
            }
        )

    if result.orbit_deg < config.target_deg:
        result.aborted = 'cycle_budget_exhausted'
    return result
