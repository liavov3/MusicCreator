"""
Low-level DSP utilities: oscillators, envelopes, filters, saturation, normalization, and stereo helpers.
Designed for clarity and educational value, not micro-optimized for speed.

All functions are pure and stateless where possible.
"""

from __future__ import annotations
import numpy as np
from .constants import SR


# ---------- Oscillators ----------

def sine(freq: float, length_s: float) -> np.ndarray:
    """Generate a sine wave."""
    t = np.linspace(0.0, length_s, int(SR * length_s), endpoint=False)
    return np.sin(2.0 * np.pi * freq * t)


def exp_sweep_sine(f_start: float, f_end: float, length_s: float) -> np.ndarray:
    """
    Exponential frequency sweep sine. Good for kick body (pitch drop).
    """
    n = int(SR * length_s)
    t = np.linspace(0.0, length_s, n, endpoint=False)
    # avoid division by zero
    f_start = max(1e-6, float(f_start))
    f_end = max(1e-6, float(f_end))
    k = np.log(f_end / f_start) / length_s
    inst_freq = f_start * np.exp(k * t)
    phase = 2.0 * np.pi * np.cumsum(inst_freq) / SR
    return np.sin(phase)


def saw(freq: float, length_s: float) -> np.ndarray:
    """
    Simple band-limited-ish saw (not fully BLEP/PolyBLEP; acceptable for low registers with post-LP).
    """
    n = int(SR * length_s)
    t = np.linspace(0.0, length_s, n, endpoint=False)
    phase = (freq * t) % 1.0
    return 2.0 * phase - 1.0


# ---------- Envelopes ----------

def env_exp(length_s: float, tau_s: float) -> np.ndarray:
    """
    Exponential decay envelope, commonly used for percussive tails.
    """
    n = int(SR * length_s)
    t = np.arange(n) / SR
    return np.exp(-t / max(1e-6, tau_s))


def env_pluck(n_samples: int, on_samples: int, hold_samples: int, off_samples: int) -> np.ndarray:
    """
    Simple pluck-like envelope: instant-ish attack, short hold, linear release.
    """
    n = int(n_samples)
    env = np.zeros(n, dtype=float)

    a = min(10, on_samples)  # ultra-short attack
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a, endpoint=False)

    hold_end = min(n, a + hold_samples)
    env[a:hold_end] = 1.0

    rel_end = min(n, hold_end + off_samples)
    if rel_end > hold_end:
        env[hold_end:rel_end] = np.linspace(1.0, 0.0, rel_end - hold_end, endpoint=False)

    return env


# ---------- Filters & Nonlinear ----------

def one_pole_lowpass(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """
    One-pole low-pass filter. Not linear phase; adequate for quick tone-shaping.
    """
    if cutoff_hz <= 0.0:
        return np.zeros_like(x)
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    alpha = 1.0 / (1.0 + rc * (1.0 / SR))
    y = np.zeros_like(x)
    for i in range(1, len(x)):
        y[i] = y[i - 1] + alpha * (x[i] - y[i - 1])
    return y


def saturate_tanh(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """
    Soft saturation via tanh. Drive > 1 increases harmonic content.
    """
    return np.tanh(drive * x)


# ---------- Gain & Stereo ----------

def normalize(x: np.ndarray, headroom_db: float = 1.0) -> np.ndarray:
    """
    Normalize to 0 dBFS, then apply headroom (positive dB reduces level).
    """
    peak = float(np.max(np.abs(x))) + 1e-12
    x = x / peak
    gain = 10 ** (-headroom_db / 20.0)
    return x * gain


def stereo_widen_mono(x: np.ndarray, delay_ms: float = 12.0) -> np.ndarray:
    """
    Create a faux-stereo field by delaying the right channel slightly.
    Input is mono; output shape = (2, N).
    """
    delay = int(SR * delay_ms / 1000.0)
    left = x.copy()
    if delay > 0 and delay < len(x):
        right = np.concatenate([np.zeros(delay), x[:-delay]])
    else:
        right = x.copy()
    return np.stack([left, right], axis=0)
