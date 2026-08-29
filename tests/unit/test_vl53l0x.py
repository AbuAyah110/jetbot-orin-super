from __future__ import annotations

from jetbot_agent.hardware.vl53l0x import (
    CLEAR_MM,
    OUT_OF_RANGE_MM,
    STOP_MM,
    decode_range_mm,
    interpret_range_mm,
)
from jetbot_agent.robot_loop.demos import occupancy_allows_creep


def test_tof_close_range_blocks_creep():
    allowed, detail = occupancy_allows_creep(b'', range_mm=80)
    assert allowed is False
    assert detail['source'] == 'tof'
    assert detail['blocked'] is True
    assert detail['clear'] is False


def test_tof_far_range_allows_one_pulse():
    allowed, detail = occupancy_allows_creep(b'', range_mm=CLEAR_MM)
    assert allowed is True
    assert detail['clear'] is True
    assert detail['range_mm'] == CLEAR_MM


def test_tof_uncertain_and_out_of_range_fail_closed():
    mid = (STOP_MM + CLEAR_MM) // 2
    allowed, detail = occupancy_allows_creep(b'', range_mm=mid)
    assert allowed is False
    assert detail['rejected'] == 'uncertain_band'
    allowed, detail = occupancy_allows_creep(b'', range_mm=OUT_OF_RANGE_MM)
    assert allowed is False
    assert detail['ok'] is False


def test_interpret_range_none_is_not_clear():
    detail = interpret_range_mm(None)
    assert detail['clear'] is False
    assert detail['ok'] is False


def test_gy530_status_11_keeps_a_real_range():
    assert decode_range_mm(status=11, raw_mm=553) == 553
    assert decode_range_mm(status=9, raw_mm=400) == 400
    assert decode_range_mm(status=0, raw_mm=120) == 120


def test_hardware_fail_and_wrap_are_out_of_range():
    assert decode_range_mm(status=5, raw_mm=120) == OUT_OF_RANGE_MM
    assert decode_range_mm(status=11, raw_mm=8191) == OUT_OF_RANGE_MM
    assert decode_range_mm(status=4, raw_mm=8191) == OUT_OF_RANGE_MM
