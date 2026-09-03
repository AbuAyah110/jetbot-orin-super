"""VL53L0X ranging on Linux I2C. Never opens motors or PWM.

Adapted from the Pololu / Adafruit CircuitPython VL53L0X driver (MIT).
Default on this Jetson: ``/dev/i2c-1`` address ``0x29`` — 40-pin pins 27 (SDA)
and 28 (SCL), shared with the onboard FUSB301 at ``0x25`` and INA3221 at
``0x40``. The motor HAT stays on bus 7, which is the *other* header pin pair
(3 and 5), so PWM traffic never shares this bus.
"""

from __future__ import annotations

import math
import time
from typing import Optional

from jetbot_agent._stage import StageNotReady

DEFAULT_BUS = 1
DEFAULT_ADDRESS = 0x29
MODEL_ID = 0xEE
REVISION_ID = 0x10
# ST marks "no object / wrap" as this sentinel.
OUT_OF_RANGE_MM = 8190
# RESULT_RANGE_STATUS bits 6:3. ST's PAL table maps raw 9 → "range valid".
# Cheap GY-530 boards commonly report 11 with a usable millimetre value.
# 5 is analog/VCSEL hardware fail.
_DEVICE_STATUS_HARDWARE_FAIL = 5
_DEVICE_STATUS_OK = frozenset({0, 9, 11})
# "The ranging sequence ran and found nothing within range" is a real
# observation, not a failed read. Measured on this GY-530: a hand at 115-366 mm
# reports 11, and removing it reports 4 with the wrap value. Keeping 4 lumped in
# with hardware faults made an empty floor indistinguishable from a dead laser,
# so creep refused in the one situation it exists for.
_DEVICE_STATUS_NO_TARGET = frozenset({3, 4})

READING_VALID = 'valid'
READING_NO_TARGET = 'no_target'
READING_FAULT = 'fault'
# Creep / near-field stop. Indoor JetBot should not lunge into something closer.
STOP_MM = 250
CLEAR_MM = 400

_SYSRANGE_START = 0x00
_SYSTEM_SEQUENCE_CONFIG = 0x01
_SYSTEM_INTERRUPT_CONFIG_GPIO = 0x0A
_SYSTEM_INTERRUPT_CLEAR = 0x0B
_RESULT_INTERRUPT_STATUS = 0x13
_RESULT_RANGE_STATUS = 0x14
_MSRC_CONFIG_CONTROL = 0x60
_PRE_RANGE_CONFIG_VCSEL_PERIOD = 0x50
_PRE_RANGE_CONFIG_TIMEOUT_MACROP_HI = 0x51
_FINAL_RANGE_CONFIG_VCSEL_PERIOD = 0x70
_FINAL_RANGE_CONFIG_TIMEOUT_MACROP_HI = 0x71
_FINAL_RANGE_CONFIG_MIN_COUNT_RATE_RTN_LIMIT = 0x44
_MSRC_CONFIG_TIMEOUT_MACROP = 0x46
_IDENTIFICATION_MODEL_ID = 0xC0
_IDENTIFICATION_REVISION_ID = 0xC2
_GPIO_HV_MUX_ACTIVE_HIGH = 0x84
_GLOBAL_CONFIG_SPAD_ENABLES_REF_0 = 0xB0
_DYNAMIC_SPAD_NUM_REQUESTED_REF_SPAD = 0x4E
_DYNAMIC_SPAD_REF_EN_START_OFFSET = 0x4F
_GLOBAL_CONFIG_REF_EN_START_SELECT = 0xB6
_VCSEL_PERIOD_PRE_RANGE = 0
_VCSEL_PERIOD_FINAL_RANGE = 1


def _decode_timeout(val: int) -> float:
    return float(val & 0xFF) * math.pow(2.0, ((val & 0xFF00) >> 8)) + 1


def _encode_timeout(timeout_mclks: float) -> int:
    timeout_mclks = int(timeout_mclks) & 0xFFFF
    ls_byte = 0
    ms_byte = 0
    if timeout_mclks > 0:
        ls_byte = timeout_mclks - 1
        while ls_byte > 255:
            ls_byte >>= 1
            ms_byte += 1
        return ((ms_byte << 8) | (ls_byte & 0xFF)) & 0xFFFF
    return 0


def _timeout_mclks_to_microseconds(timeout_period_mclks: int, vcsel_period_pclks: int) -> int:
    macro_period_ns = ((2304 * vcsel_period_pclks * 1655) + 500) // 1000
    return ((timeout_period_mclks * macro_period_ns) + (macro_period_ns // 2)) // 1000


def _timeout_microseconds_to_mclks(timeout_period_us: int, vcsel_period_pclks: int) -> int:
    macro_period_ns = ((2304 * vcsel_period_pclks * 1655) + 500) // 1000
    return ((timeout_period_us * 1000) + (macro_period_ns // 2)) // macro_period_ns


def classify_reading(*, status: int, raw_mm: int) -> tuple[str, int]:
    """Separate a measured distance, an empty field, and a broken sensor.

    Collapsing the last two into one sentinel is what made an open floor
    unreadable: both arrived as "out of range", and the only safe response to a
    possibly dead sensor is to refuse.
    """
    code = int(status)
    millimetres = int(raw_mm)
    if code == _DEVICE_STATUS_HARDWARE_FAIL:
        return READING_FAULT, OUT_OF_RANGE_MM
    if code in _DEVICE_STATUS_NO_TARGET:
        return READING_NO_TARGET, OUT_OF_RANGE_MM
    if code in _DEVICE_STATUS_OK:
        if 0 < millimetres < OUT_OF_RANGE_MM:
            return READING_VALID, millimetres
        # A status the device calls good, carrying the wrap value, still means
        # the beam found nothing.
        return READING_NO_TARGET, OUT_OF_RANGE_MM
    return READING_FAULT, OUT_OF_RANGE_MM


def interpret_range_mm(
    range_mm: Optional[int],
    *,
    kind: str = '',
) -> dict:
    """Fail-closed near-field policy for one creep pulse.

    ``kind`` carries the reading class from :func:`classify_reading`. A
    confirmed empty field permits the pulse; anything unexplained does not.
    """
    detail = {
        'ok': False,
        'range_mm': range_mm,
        'blocked': False,
        'clear': False,
        'rejected': '',
    }
    if kind == READING_NO_TARGET:
        # Nothing within the sensor's reach. This is the ordinary open-floor
        # answer, and it is the evidence a creep pulse needs. It is not proof
        # for a matte black or steeply angled surface, which can also return
        # nothing; the one-pulse limit remains the guard for that.
        detail['ok'] = True
        detail['clear'] = True
        detail['range_mm'] = OUT_OF_RANGE_MM
        detail['reason'] = 'no_target_in_range'
        return detail
    if kind == READING_FAULT:
        detail['rejected'] = 'sensor_fault'
        return detail
    if range_mm is None:
        detail['rejected'] = 'no_reading'
        return detail
    try:
        millimetres = int(range_mm)
    except (TypeError, ValueError):
        detail['rejected'] = 'bad_reading'
        return detail
    detail['range_mm'] = millimetres
    if millimetres <= 0 or millimetres >= OUT_OF_RANGE_MM:
        detail['rejected'] = 'out_of_range'
        return detail
    detail['ok'] = True
    if millimetres < STOP_MM:
        detail['blocked'] = True
        return detail
    if millimetres >= CLEAR_MM:
        detail['clear'] = True
        return detail
    detail['rejected'] = 'uncertain_band'
    return detail


def tof_near_field_blocks(
    range_mm: Optional[int],
    *,
    kind: str = '',
) -> tuple[bool, dict]:
    """True when the ToF says something is too close to drive into.

    Approach and creep both use this so contact is stopped before the wheels
    push an object, not only when the path was already blocked at the gate.
    """
    detail = interpret_range_mm(range_mm, kind=kind)
    if detail.get('blocked'):
        return True, detail
    if detail.get('rejected') == 'uncertain_band':
        return True, detail
    return False, detail


def creep_refusal_reply(detail: dict) -> str:
    """Spoken explanation when a creep pulse is refused."""
    if detail.get('rejected') == 'sensor_fault':
        return 'My distance sensor is not answering, so I stayed put.'
    millimetres = detail.get('range_mm')
    if detail.get('blocked') or detail.get('rejected') == 'uncertain_band':
        if isinstance(millimetres, int) and 0 < millimetres < OUT_OF_RANGE_MM:
            return (
                'My distance sensor sees something {0} centimetres '
                'ahead, so I stopped.'
            ).format(max(1, round(millimetres / 10.0)))
        return 'My distance sensor reports an obstacle, so I stopped.'
    return (
        'I cannot tell if the floor is clear without a distance sensor, '
        'so I am staying put.'
    )


def decode_range_mm(*, status: int, raw_mm: int) -> int:
    """Keep a usable millimetre field; fail closed on hardware death or wrap."""
    if int(status) == _DEVICE_STATUS_HARDWARE_FAIL:
        return OUT_OF_RANGE_MM
    millimetres = int(raw_mm)
    if millimetres <= 0 or millimetres >= OUT_OF_RANGE_MM:
        return OUT_OF_RANGE_MM
    if int(status) in _DEVICE_STATUS_OK:
        return millimetres
    return OUT_OF_RANGE_MM


class VL53L0X:
    """Single-shot millimetre ranging. Not thread-safe."""

    def __init__(
        self,
        bus: int = DEFAULT_BUS,
        address: int = DEFAULT_ADDRESS,
        io_timeout_s: float = 1.0,
    ) -> None:
        try:
            from smbus2 import SMBus
        except ImportError as exc:
            raise StageNotReady('smbus2 is required for VL53L0X') from exc
        self.bus_id = int(bus)
        self.address = int(address)
        self.io_timeout_s = float(io_timeout_s)
        self._bus = SMBus(self.bus_id)
        self._stop_variable = 0
        self._measurement_timing_budget_us = 0
        self._data_ready = False
        self.last_status = -1
        self.last_raw_mm = 0
        self.last_kind = ''
        model = self._read_u8(_IDENTIFICATION_MODEL_ID)
        revision = self._read_u8(_IDENTIFICATION_REVISION_ID)
        if model != MODEL_ID:
            raise StageNotReady(
                'VL53L0X model id {0:#x} on bus {1} addr {2:#x}, expected {3:#x}'.format(
                    model, self.bus_id, self.address, MODEL_ID
                )
            )
        self.revision = revision
        self._init_sensor()

    def close(self) -> None:
        try:
            self._bus.close()
        except Exception:
            pass

    def __enter__(self) -> 'VL53L0X':
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def _read_u8(self, register: int) -> int:
        return self._bus.read_byte_data(self.address, register & 0xFF)

    def _write_u8(self, register: int, value: int) -> None:
        self._bus.write_byte_data(self.address, register & 0xFF, value & 0xFF)

    def _read_u16(self, register: int) -> int:
        data = self._bus.read_i2c_block_data(self.address, register & 0xFF, 2)
        return (data[0] << 8) | data[1]

    def _write_u16(self, register: int, value: int) -> None:
        self._bus.write_i2c_block_data(
            self.address, register & 0xFF, [(value >> 8) & 0xFF, value & 0xFF]
        )

    def _init_sensor(self) -> None:
        # 2.8 V I2C pull-ups are common on VL53L0X breakouts.
        self._write_u8(0x89, self._read_u8(0x89) | 0x01)
        for pair in ((0x88, 0x00), (0x80, 0x01), (0xFF, 0x01), (0x00, 0x00)):
            self._write_u8(pair[0], pair[1])
        self._stop_variable = self._read_u8(0x91)
        for pair in ((0x00, 0x01), (0xFF, 0x00), (0x80, 0x00)):
            self._write_u8(pair[0], pair[1])
        config_control = self._read_u8(_MSRC_CONFIG_CONTROL) | 0x12
        self._write_u8(_MSRC_CONFIG_CONTROL, config_control)
        self.signal_rate_limit = 0.25
        self._write_u8(_SYSTEM_SEQUENCE_CONFIG, 0xFF)
        spad_count, spad_is_aperture = self._get_spad_info()
        ref_spad_map = bytearray(self._bus.read_i2c_block_data(
            self.address, _GLOBAL_CONFIG_SPAD_ENABLES_REF_0, 6
        ))
        for pair in (
            (0xFF, 0x01),
            (_DYNAMIC_SPAD_REF_EN_START_OFFSET, 0x00),
            (_DYNAMIC_SPAD_NUM_REQUESTED_REF_SPAD, 0x2C),
            (0xFF, 0x00),
            (_GLOBAL_CONFIG_REF_EN_START_SELECT, 0xB4),
        ):
            self._write_u8(pair[0], pair[1])
        first_spad_to_enable = 12 if spad_is_aperture else 0
        spads_enabled = 0
        for i in range(48):
            index = i // 8
            bit = 1 << (i % 8)
            if i < first_spad_to_enable or spads_enabled == spad_count:
                ref_spad_map[index] &= ~bit & 0xFF
            elif (ref_spad_map[index] >> (i % 8)) & 0x1:
                spads_enabled += 1
        self._bus.write_i2c_block_data(
            self.address, _GLOBAL_CONFIG_SPAD_ENABLES_REF_0, list(ref_spad_map)
        )
        for pair in (
            (0xFF, 0x01),
            (0x00, 0x00),
            (0xFF, 0x00),
            (0x09, 0x00),
            (0x10, 0x00),
            (0x11, 0x00),
            (0x24, 0x01),
            (0x25, 0xFF),
            (0x75, 0x00),
            (0xFF, 0x01),
            (0x4E, 0x2C),
            (0x48, 0x00),
            (0x30, 0x20),
            (0xFF, 0x00),
            (0x30, 0x09),
            (0x54, 0x00),
            (0x31, 0x04),
            (0x32, 0x03),
            (0x40, 0x83),
            (0x46, 0x25),
            (0x60, 0x00),
            (0x27, 0x00),
            (0x50, 0x06),
            (0x51, 0x00),
            (0x52, 0x96),
            (0x56, 0x08),
            (0x57, 0x30),
            (0x61, 0x00),
            (0x62, 0x00),
            (0x64, 0x00),
            (0x65, 0x00),
            (0x66, 0xA0),
            (0xFF, 0x01),
            (0x22, 0x32),
            (0x47, 0x14),
            (0x49, 0xFF),
            (0x4A, 0x00),
            (0xFF, 0x00),
            (0x7A, 0x0A),
            (0x7B, 0x00),
            (0x78, 0x21),
            (0xFF, 0x01),
            (0x23, 0x34),
            (0x42, 0x00),
            (0x44, 0xFF),
            (0x45, 0x26),
            (0x46, 0x05),
            (0x40, 0x40),
            (0x0E, 0x06),
            (0x20, 0x1A),
            (0x43, 0x40),
            (0xFF, 0x00),
            (0x34, 0x03),
            (0x35, 0x44),
            (0xFF, 0x01),
            (0x31, 0x04),
            (0x4B, 0x09),
            (0x4C, 0x05),
            (0x4D, 0x04),
            (0xFF, 0x00),
            (0x44, 0x00),
            (0x45, 0x20),
            (0x47, 0x08),
            (0x48, 0x28),
            (0x67, 0x00),
            (0x70, 0x04),
            (0x71, 0x01),
            (0x72, 0xFE),
            (0x76, 0x00),
            (0x77, 0x00),
            (0xFF, 0x01),
            (0x0D, 0x01),
            (0xFF, 0x00),
            (0x80, 0x01),
            (0x01, 0xF8),
            (0xFF, 0x01),
            (0x8E, 0x01),
            (0x00, 0x01),
            (0xFF, 0x00),
            (0x80, 0x00),
        ):
            self._write_u8(pair[0], pair[1])
        self._write_u8(_SYSTEM_INTERRUPT_CONFIG_GPIO, 0x04)
        gpio_hv_mux_active_high = self._read_u8(_GPIO_HV_MUX_ACTIVE_HIGH)
        self._write_u8(_GPIO_HV_MUX_ACTIVE_HIGH, gpio_hv_mux_active_high & ~0x10)
        self._write_u8(_SYSTEM_INTERRUPT_CLEAR, 0x01)
        self._measurement_timing_budget_us = self.measurement_timing_budget
        self._write_u8(_SYSTEM_SEQUENCE_CONFIG, 0xE8)
        self.measurement_timing_budget = self._measurement_timing_budget_us
        self._write_u8(_SYSTEM_SEQUENCE_CONFIG, 0x01)
        self._perform_single_ref_calibration(0x40)
        self._write_u8(_SYSTEM_SEQUENCE_CONFIG, 0x02)
        self._perform_single_ref_calibration(0x00)
        self._write_u8(_SYSTEM_SEQUENCE_CONFIG, 0xE8)

    def _wait(self, predicate) -> None:
        start = time.monotonic()
        while not predicate():
            if self.io_timeout_s > 0 and (time.monotonic() - start) >= self.io_timeout_s:
                raise StageNotReady('timeout waiting for VL53L0X')
            time.sleep(0.001)

    def _get_spad_info(self) -> tuple[int, bool]:
        for pair in ((0x80, 0x01), (0xFF, 0x01), (0x00, 0x00), (0xFF, 0x06)):
            self._write_u8(pair[0], pair[1])
        self._write_u8(0x83, self._read_u8(0x83) | 0x04)
        for pair in ((0xFF, 0x07), (0x81, 0x01), (0x80, 0x01), (0x94, 0x6B), (0x83, 0x00)):
            self._write_u8(pair[0], pair[1])
        self._wait(lambda: self._read_u8(0x83) != 0x00)
        self._write_u8(0x83, 0x01)
        tmp = self._read_u8(0x92)
        count = tmp & 0x7F
        is_aperture = ((tmp >> 7) & 0x01) == 1
        for pair in ((0x81, 0x00), (0xFF, 0x06)):
            self._write_u8(pair[0], pair[1])
        self._write_u8(0x83, self._read_u8(0x83) & ~0x04)
        for pair in ((0xFF, 0x01), (0x00, 0x01), (0xFF, 0x00), (0x80, 0x00)):
            self._write_u8(pair[0], pair[1])
        return count, is_aperture

    def _perform_single_ref_calibration(self, vhv_init_byte: int) -> None:
        self._write_u8(_SYSRANGE_START, 0x01 | (vhv_init_byte & 0xFF))
        self._wait(lambda: (self._read_u8(_RESULT_INTERRUPT_STATUS) & 0x07) != 0)
        self._write_u8(_SYSTEM_INTERRUPT_CLEAR, 0x01)
        self._write_u8(_SYSRANGE_START, 0x00)

    def _get_vcsel_pulse_period(self, vcsel_period_type: int) -> int:
        if vcsel_period_type == _VCSEL_PERIOD_PRE_RANGE:
            val = self._read_u8(_PRE_RANGE_CONFIG_VCSEL_PERIOD)
            return ((val + 1) & 0xFF) << 1
        if vcsel_period_type == _VCSEL_PERIOD_FINAL_RANGE:
            val = self._read_u8(_FINAL_RANGE_CONFIG_VCSEL_PERIOD)
            return ((val + 1) & 0xFF) << 1
        return 255

    def _get_sequence_step_enables(self) -> tuple[bool, bool, bool, bool, bool]:
        sequence_config = self._read_u8(_SYSTEM_SEQUENCE_CONFIG)
        tcc = (sequence_config >> 4) & 0x1 > 0
        dss = (sequence_config >> 3) & 0x1 > 0
        msrc = (sequence_config >> 2) & 0x1 > 0
        pre_range = (sequence_config >> 6) & 0x1 > 0
        final_range = (sequence_config >> 7) & 0x1 > 0
        return tcc, dss, msrc, pre_range, final_range

    def _get_sequence_step_timeouts(self, pre_range: int):
        pre_range_vcsel_period_pclks = self._get_vcsel_pulse_period(_VCSEL_PERIOD_PRE_RANGE)
        msrc_dss_tcc_mclks = (self._read_u8(_MSRC_CONFIG_TIMEOUT_MACROP) + 1) & 0xFF
        msrc_dss_tcc_us = _timeout_mclks_to_microseconds(
            msrc_dss_tcc_mclks, pre_range_vcsel_period_pclks
        )
        pre_range_mclks = _decode_timeout(self._read_u16(_PRE_RANGE_CONFIG_TIMEOUT_MACROP_HI))
        pre_range_us = _timeout_mclks_to_microseconds(
            pre_range_mclks, pre_range_vcsel_period_pclks
        )
        final_range_vcsel_period_pclks = self._get_vcsel_pulse_period(_VCSEL_PERIOD_FINAL_RANGE)
        final_range_mclks = _decode_timeout(self._read_u16(_FINAL_RANGE_CONFIG_TIMEOUT_MACROP_HI))
        if pre_range:
            final_range_mclks -= pre_range_mclks
        final_range_us = _timeout_mclks_to_microseconds(
            final_range_mclks, final_range_vcsel_period_pclks
        )
        return (
            msrc_dss_tcc_us,
            pre_range_us,
            final_range_us,
            final_range_vcsel_period_pclks,
            pre_range_mclks,
        )

    @property
    def signal_rate_limit(self) -> float:
        return self._read_u16(_FINAL_RANGE_CONFIG_MIN_COUNT_RATE_RTN_LIMIT) / (1 << 7)

    @signal_rate_limit.setter
    def signal_rate_limit(self, val: float) -> None:
        self._write_u16(_FINAL_RANGE_CONFIG_MIN_COUNT_RATE_RTN_LIMIT, int(val * (1 << 7)))

    @property
    def measurement_timing_budget(self) -> int:
        budget_us = 1910 + 960
        tcc, dss, msrc, pre_range, final_range = self._get_sequence_step_enables()
        step_timeouts = self._get_sequence_step_timeouts(pre_range)
        msrc_dss_tcc_us, pre_range_us, final_range_us, _, _ = step_timeouts
        if tcc:
            budget_us += msrc_dss_tcc_us + 590
        if dss:
            budget_us += 2 * (msrc_dss_tcc_us + 690)
        elif msrc:
            budget_us += msrc_dss_tcc_us + 660
        if pre_range:
            budget_us += pre_range_us + 660
        if final_range:
            budget_us += final_range_us + 550
        self._measurement_timing_budget_us = budget_us
        return budget_us

    @measurement_timing_budget.setter
    def measurement_timing_budget(self, budget_us: int) -> None:
        used_budget_us = 1320 + 960
        tcc, dss, msrc, pre_range, final_range = self._get_sequence_step_enables()
        step_timeouts = self._get_sequence_step_timeouts(pre_range)
        msrc_dss_tcc_us, pre_range_us, _ = step_timeouts[:3]
        final_range_vcsel_period_pclks, pre_range_mclks = step_timeouts[3:]
        if tcc:
            used_budget_us += msrc_dss_tcc_us + 590
        if dss:
            used_budget_us += 2 * (msrc_dss_tcc_us + 690)
        elif msrc:
            used_budget_us += msrc_dss_tcc_us + 660
        if pre_range:
            used_budget_us += pre_range_us + 660
        if final_range:
            used_budget_us += 550
        if used_budget_us > budget_us:
            raise ValueError('requested VL53L0X timeout too big')
        final_range_timeout_us = budget_us - used_budget_us
        final_range_timeout_mclks = _timeout_microseconds_to_mclks(
            final_range_timeout_us, final_range_vcsel_period_pclks
        )
        if pre_range:
            final_range_timeout_mclks += pre_range_mclks
        self._write_u16(
            _FINAL_RANGE_CONFIG_TIMEOUT_MACROP_HI,
            _encode_timeout(final_range_timeout_mclks),
        )
        self._measurement_timing_budget_us = budget_us

    def range_mm(self) -> int:
        """One single-shot range in millimetres. Does not move the robot."""
        for pair in (
            (0x80, 0x01),
            (0xFF, 0x01),
            (0x00, 0x00),
            (0x91, self._stop_variable),
            (0x00, 0x01),
            (0xFF, 0x00),
            (0x80, 0x00),
            (_SYSRANGE_START, 0x01),
        ):
            self._write_u8(pair[0], pair[1])
        self._wait(lambda: (self._read_u8(_SYSRANGE_START) & 0x01) == 0)
        self._data_ready = False
        self._wait(lambda: (self._read_u8(_RESULT_INTERRUPT_STATUS) & 0x07) != 0)
        range_mm = self._read_u16(_RESULT_RANGE_STATUS + 10)
        status = (self._read_u8(_RESULT_RANGE_STATUS) >> 3) & 0x1F
        self._write_u8(_SYSTEM_INTERRUPT_CLEAR, 0x01)
        self.last_status = status
        self.last_raw_mm = range_mm
        kind, millimetres = classify_reading(status=status, raw_mm=range_mm)
        self.last_kind = kind
        return millimetres
