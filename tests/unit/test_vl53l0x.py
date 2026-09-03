from __future__ import annotations

from jetbot_agent.hardware.vl53l0x import (
    APPROACH_STOP_MM,
    CLEAR_MM,
    OUT_OF_RANGE_MM,
    READING_FAULT,
    READING_NO_TARGET,
    READING_VALID,
    STOP_MM,
    approach_stop_reply,
    classify_reading,
    creep_refusal_reply,
    decode_range_mm,
    interpret_range_mm,
    tof_near_field_blocks,
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


def test_empty_field_is_evidence_of_clearance_not_a_failed_read():
    # Status 4 with the wrap value is the ordinary answer over open floor.
    # Treating it as unreadable made creep refuse exactly where it should go.
    kind, millimetres = classify_reading(status=4, raw_mm=8191)
    assert kind == READING_NO_TARGET
    assert millimetres == OUT_OF_RANGE_MM

    allowed, detail = occupancy_allows_creep(b'', range_mm=millimetres, kind=kind)
    assert allowed is True
    assert detail['clear'] is True
    assert detail['reason'] == 'no_target_in_range'


def test_a_broken_sensor_still_refuses_the_pulse():
    kind, millimetres = classify_reading(status=5, raw_mm=120)
    assert kind == READING_FAULT

    allowed, detail = occupancy_allows_creep(b'', range_mm=millimetres, kind=kind)
    assert allowed is False
    assert detail['rejected'] == 'sensor_fault'


def test_a_measured_obstacle_outranks_an_empty_field():
    kind, millimetres = classify_reading(status=11, raw_mm=150)
    assert kind == READING_VALID

    allowed, detail = occupancy_allows_creep(b'', range_mm=millimetres, kind=kind)
    assert allowed is False
    assert detail['blocked'] is True


def test_uncertain_band_blocks_creep_but_not_approach():
    blocked, detail = tof_near_field_blocks(310, kind=READING_VALID)
    assert blocked is True
    assert detail['rejected'] == 'uncertain_band'

    blocked, detail = tof_near_field_blocks(
        310, kind=READING_VALID, for_approach=True
    )
    assert blocked is False

    blocked, detail = tof_near_field_blocks(
        140, kind=READING_VALID, for_approach=True
    )
    assert blocked is True
    assert detail.get('approach_stop') is True
    assert 'close enough' in approach_stop_reply(detail).lower()

    blocked, detail = tof_near_field_blocks(
        200, kind=READING_VALID, for_approach=True
    )
    assert blocked is False


def test_uncertain_band_blocks_approach_and_creep():
    blocked, detail = tof_near_field_blocks(285, kind=READING_VALID)
    assert blocked is True
    assert detail['rejected'] == 'uncertain_band'
    assert '28 centimetres' in creep_refusal_reply(detail)


def test_gy530_status_11_keeps_a_real_range():
    assert decode_range_mm(status=11, raw_mm=553) == 553
    assert decode_range_mm(status=9, raw_mm=400) == 400
    assert decode_range_mm(status=0, raw_mm=120) == 120


def test_hardware_fail_and_wrap_are_out_of_range():
    assert decode_range_mm(status=5, raw_mm=120) == OUT_OF_RANGE_MM
    assert decode_range_mm(status=11, raw_mm=8191) == OUT_OF_RANGE_MM
    assert decode_range_mm(status=4, raw_mm=8191) == OUT_OF_RANGE_MM
