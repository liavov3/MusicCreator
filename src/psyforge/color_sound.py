"""
Color-to-sound mapping utilities.

Bass RGB:
- R -> harmonic content & drive (sine<->saw + saturation)
- G -> brightness (low-pass cutoff)
- B -> motion (tremolo LFO rate/depth)

Kick RGB:
- R -> click/transient amount + drive (punch)
- G -> brightness vs. tail (start freq & decay time)
- B -> boom/depth (pitch drop & length)

Values are musically constrained to sane ranges.
"""

from __future__ import annotations
from dataclasses import dataclass

# ---------------- Bass ----------------

@dataclass
class BassParams:
    osc_blend: float  # 0=sine, 1=saw
    lp_cut: float     # Hz
    drive: float      # saturation amount
    level: float      # linear gain (0..1.5)
    lfo_rate: float   # Hz
    lfo_depth: float  # 0..1

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def rgb_to_bass_params(r: int, g: int, b: int) -> BassParams:
    R = max(0, min(255, int(r))) / 255.0
    G = max(0, min(255, int(g))) / 255.0
    B = max(0, min(255, int(b))) / 255.0

    lp_cut = _lerp(80.0, 340.0, G)          # deeper->brighter
    osc_blend = R                            # 0..1 (sine->saw)
    drive = _lerp(1.0, 2.8, R)               # more red -> more grit
    level = _lerp(0.70, 1.15, 0.5*R + 0.5*G)

    lfo_rate = _lerp(0.10, 10.0, B)         # motion
    lfo_depth = _lerp(0.00, 0.50, B)

    return BassParams(
        osc_blend=osc_blend, lp_cut=lp_cut, drive=drive,
        level=level, lfo_rate=lfo_rate, lfo_depth=lfo_depth
    )

# ---------------- Kick ----------------

@dataclass
class KickParams:
    length_ms: float
    f_start: float
    f_end: float
    click_ms: float
    body_tau_s: float
    gain: float
    drive: float
    click_gain: float

def rgb_to_kick_params(r: int, g: int, b: int) -> KickParams:
    R = max(0, min(255, int(r))) / 255.0
    G = max(0, min(255, int(g))) / 255.0
    B = max(0, min(255, int(b))) / 255.0

    # Red -> transient & saturation
    click_gain = _lerp(0.6, 2.2, R)
    drive      = _lerp(1.0, 2.8, R)

    # Green -> brightness vs tail
    f_start    = _lerp(78.0, 110.0, G)
    body_tau_s = _lerp(0.09, 0.05, G)   # brighter -> shorter decay
    click_ms   = _lerp(1.0, 3.0, G)

    # Blue -> boom/depth & length
    f_end      = _lerp(55.0, 43.0, B)   # more blue -> deeper drop
    length_ms  = _lerp(165.0, 210.0, B)

    gain       = _lerp(0.90, 1.15, 0.5*R + 0.5*G)

    return KickParams(
        length_ms=length_ms, f_start=f_start, f_end=f_end,
        click_ms=click_ms, body_tau_s=body_tau_s,
        gain=gain, drive=drive, click_gain=click_gain
    )

# ---------------- util ----------------

def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{int(r)&255:02X}{int(g)&255:02X}{int(b)&255:02X}"
