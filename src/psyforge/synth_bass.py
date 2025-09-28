"""
Rolling bass synthesizer:
- Psy: continuous 1/16 notes with short pluck envelopes
- Goa: off-beat 1/8 with longer gates
"""

from __future__ import annotations
import numpy as np
from .dsp import saw, env_pluck, one_pole_lowpass, saturate_tanh, normalize
from .constants import SR


def rolling_bass(bpm: int = 145,
                 root_hz: float = 46.25,
                 bars: int = 2,
                 style: str = "psy") -> np.ndarray:
    """
    Generate a mono rolling bass line for the given style.
    Returns mono array (N,).
    """
    sec_per_beat = 60.0 / float(bpm)
    sec_per_bar = 4.0 * sec_per_beat
    length_s = bars * sec_per_bar
    n = int(SR * length_s)

    out = np.zeros(n, dtype=float)

    if style.lower() == "goa":
        steps = 8 * bars             # eighth notes
        step_len = int(SR * (sec_per_beat / 2.0))
        gate_on = int(0.04 * SR)
        gate_hold = int(0.14 * SR)
        gate_off = int(0.08 * SR)
        osc_freq = root_hz * 2.0
        lp_cut = 160.0
        drive = 1.5
        level = 0.6
    else:
        steps = 16 * bars            # sixteenth notes
        step_len = int(SR * (sec_per_beat / 4.0))
        gate_on = int(0.01 * SR)
        gate_hold = int(0.05 * SR)
        gate_off = int(0.06 * SR)
        osc_freq = root_hz * 2.0
        lp_cut = 140.0
        drive = 1.8
        level = 0.6

    idx = 0
    for _ in range(steps):
        start = idx
        end = min(start + step_len, n)
        seg_len = end - start
        if seg_len <= 0:
            break

        osc = saw(osc_freq, seg_len / SR)
        env = env_pluck(seg_len, gate_on, gate_hold, gate_off)
        voice = one_pole_lowpass(osc * env, lp_cut)
        voice = saturate_tanh(voice, drive=drive)

        out[start:end] += voice * level
        idx += step_len

    return normalize(out, headroom_db=6.0)
