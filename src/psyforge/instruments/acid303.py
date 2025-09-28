"""
Acid303 instrument (MVP):
- Saw oscillator with phase continuity
- Resonant low-pass (biquad) with per-sample cutoff modulation
- Filter envelope (env_mod), Accent (more env + level), Slide (portamento)
- Soft drive, output level

Designed for small real-time blocks.
"""

from __future__ import annotations
import math
import numpy as np
from ..constants import SR


def _tanh_drive(x: np.ndarray, drive: float) -> np.ndarray:
    # Gentle soft-clip, drive in ~[1.0 .. 3.0]
    return np.tanh(drive * x, dtype=np.float32)


class BiquadLP:
    """Simple biquad low-pass, recalculated per-sample for modulated cutoff."""
    def __init__(self, cutoff_hz: float = 800.0, q: float = 0.8):
        self.cutoff = float(cutoff_hz)
        self.q = float(q)
        # z^-1 states (Direct Form I)
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0

    def reset(self) -> None:
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def _coeffs(self, cutoff_hz: float, q: float):
        # RBJ cookbook low-pass
        w0 = 2.0 * math.pi * max(10.0, min(cutoff_hz, SR * 0.45)) / SR
        cosw = math.cos(w0)
        sinw = math.sin(w0)
        alpha = sinw / (2.0 * max(0.05, q))
        b0 = (1.0 - cosw) / 2.0
        b1 = 1.0 - cosw
        b2 = (1.0 - cosw) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cosw
        a2 = 1.0 - alpha
        # normalize
        return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)

    def process_mod(self, x: np.ndarray, cutoff_series: np.ndarray, q: float) -> np.ndarray:
        """Process with per-sample cutoff modulation."""
        y = np.empty_like(x, dtype=np.float32)
        x1 = self.x1; x2 = self.x2; y1 = self.y1; y2 = self.y2
        for i in range(len(x)):
            b0, b1, b2, a1, a2 = self._coeffs(float(cutoff_series[i]), q)
            # Direct Form I
            out = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1 = x1, x[i]
            y2, y1 = y1, out
            y[i] = out
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        return y


class Acid303:
    """
    Basic 303-like monosynth:
    - wave: saw
    - filter: resonant LPF (biquad)
    - params: cutoff, resonance(q), env_mod (Hz), slide_ms, accent_gain, drive, level
    """
    def __init__(self,
                 root_hz: float,
                 cutoff_hz: float = 900.0,
                 resonance: float = 0.85,
                 env_mod_hz: float = 1400.0,
                 slide_ms: float = 35.0,
                 accent_gain: float = 1.6,
                 drive: float = 1.4,
                 level: float = 0.9):
        self.root = float(root_hz)
        self.cutoff = float(cutoff_hz)
        self.q = float(resonance)
        self.env_mod = float(env_mod_hz)
        self.slide_ms = float(slide_ms)
        self.accent_gain = float(accent_gain)
        self.drive = float(drive)
        self.level = float(level)

        self._phase = 0.0         # oscillator phase [0..1)
        self._prev_freq = self.root

        self._lpf = BiquadLP(self.cutoff, self.q)

    # ---- live setters ----
    def set_root(self, hz: float): self.root = float(hz)
    def set_cutoff(self, hz: float): self.cutoff = float(hz)
    def set_resonance(self, q: float): self.q = float(q)
    def set_env_mod(self, hz: float): self.env_mod = float(hz)
    def set_slide_ms(self, ms: float): self.slide_ms = float(ms)
    def set_accent_gain(self, g: float): self.accent_gain = float(g)
    def set_drive(self, d: float): self.drive = float(d)
    def set_level(self, lv: float): self.level = float(lv)

    # ---- helpers ----
    @staticmethod
    def _hz_to_ratio(semitones: float) -> float:
        return 2.0 ** (semitones / 12.0)

    def _saw_vec(self, freq_series: np.ndarray) -> np.ndarray:
        """
        Generate a saw with continuous phase while frequency may vary sample-to-sample.
        phase += freq/ SR ; out = 2*phase-1
        """
        n = len(freq_series)
        out = np.empty(n, dtype=np.float32)
        ph = self._phase
        for i in range(n):
            ph += float(freq_series[i]) / SR
            ph -= math.floor(ph)  # keep [0..1)
            out[i] = 2.0 * ph - 1.0
        self._phase = ph
        return out

    # ---- synthesis ----
    def note_segment(self, length_samples: int, target_hz: float, slide: bool, accent: bool) -> np.ndarray:
        """
        Render one 16th-note segment.
        - target_hz: frequency for this step (before slide)
        - slide: if True, glide from previous note to target over slide_ms
        - accent: if True, stronger filter env + level
        """
        n = int(length_samples)
        t = np.arange(n, dtype=np.float32) / SR

        # Frequency trajectory (slide)
        start_hz = float(self._prev_freq)
        end_hz = float(target_hz)
        if slide:
            tau = max(0.001, self.slide_ms / 1000.0)
            # exponential glide
            k = np.exp(-t / tau)
            freq_series = end_hz + (start_hz - end_hz) * k
        else:
            freq_series = np.full(n, end_hz, dtype=np.float32)

        self._prev_freq = end_hz

        # Oscillator (saw)
        osc = self._saw_vec(freq_series)

        # Filter envelope -> cutoff modulation
        # Fast attack, expo decay
        env = np.exp(-t / 0.07, dtype=np.float32)  # ~70ms decay
        env_amt = self.env_mod * (self.accent_gain if accent else 1.0)
        cutoff_series = self.cutoff + env * env_amt
        cutoff_series = np.clip(cutoff_series, 100.0, SR * 0.45).astype(np.float32)

        # Filter with resonance (per-sample coef update)
        self._lpf.q = self.q
        y = self._lpf.process_mod(osc, cutoff_series, self.q)

        # Drive and level (accent slightly louder)
        y = _tanh_drive(y, self.drive)
        if accent:
            y *= min(1.8, self.level * 1.15)
        else:
            y *= self.level

        return y.astype(np.float32)
