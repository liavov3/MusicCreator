"""
Kick synthesis tailored for Psy/Goa: pitch-swept sine body + short click transient.
"""

from __future__ import annotations
import numpy as np
from .dsp import exp_sweep_sine, env_exp, normalize
from .constants import SR


def psy_kick(length_ms: float = 180.0,
             f_start: float = 85.0,
             f_end: float = 48.0,
             click_ms: float = 2.0,
             body_tau_s: float = 0.06) -> np.ndarray:
    """
    Create a psy-style kick:
    - Exponential frequency sweep sine (body)
    - Ultra-short initial click
    - Normalized with a few dB of headroom
    """
    length_s = length_ms / 1000.0
    body = exp_sweep_sine(f_start, f_end, length_s) * env_exp(length_s, tau_s=body_tau_s)

    click_len = int(SR * (click_ms / 1000.0))
    click = np.zeros_like(body)
    if click_len > 0:
        # short Hann for a tight transient
        click[:click_len] = np.hanning(2 * click_len)[:click_len] * 0.9

    kick = body * 0.9 + click
    return normalize(kick, headroom_db=6.0)
