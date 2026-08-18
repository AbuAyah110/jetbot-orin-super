from __future__ import annotations

"""Differential-drive helpers: Twist → wheel velocities."""


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def limit_twist(
    linear: float,
    angular: float,
    max_linear: float,
    max_angular: float,
) -> tuple[float, float]:
    return (
        clamp(linear, -max_linear, max_linear),
        clamp(angular, -max_angular, max_angular),
    )


def twist_to_wheel_speeds(
    linear: float,
    angular: float,
    wheel_separation_m: float,
    max_wheel: float = 1.0,
) -> tuple[float, float]:
    """Convert body twist to normalized wheel speeds in [-max_wheel, max_wheel].

    Uses the common unicycle model:
      v_l = linear - angular * (track / 2)
      v_r = linear + angular * (track / 2)
    then scales into [-1, 1] relative to ``max_wheel`` as a simple bring-up mapping
    where 1.0 ≈ ``max_linear`` order of magnitude. Callers should already clamp Twist.
    """
    half = float(wheel_separation_m) / 2.0
    left = float(linear) - float(angular) * half
    right = float(linear) + float(angular) * half
    peak = max(abs(left), abs(right), 1e-9)
    # Map so that max_linear≈max_wheel when angular=0: scale by treating linear units
    # as already roughly normalized if |linear|<=max_wheel; otherwise normalize.
    scale = 1.0
    if peak > max_wheel:
        scale = max_wheel / peak
    left *= scale
    right *= scale
    return (
        clamp(left, -max_wheel, max_wheel),
        clamp(right, -max_wheel, max_wheel),
    )
