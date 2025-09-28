"""
Real-time monophonic bass synth with raw controls:
- Oscillator blend (sine<->saw)
- Low-pass cutoff
- Soft saturation (drive)
- Tremolo LFO (rate/depth)
- Per-step pluck envelope (psy/goa variants)
"""

from __future__ import annotations
import numpy as np
from ..dsp import one_pole_lowpass, saturate_tanh
from ..constants import SR

def _sine(freq: float, n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SR
    return np.sin(2.0 * np.pi * freq * t).astype(np.float32)

def _saw(freq: float, n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SR
    phase = (freq * t) % 1.0
    return (2.0 * phase - 1.0).astype(np.float32)

def _env_pluck(n: int, on: int, hold: int, off: int) -> np.ndarray:
    env = np.zeros(n, dtype=np.float32)
    a = min(10, on)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a, endpoint=False, dtype=np.float32)
    h_end = min(n, a + hold)
    env[a:h_end] = 1.0
    r_end = min(n, h_end + off)
    if r_end > h_end:
        env[h_end:r_end] = np.linspace(1.0, 0.0, r_end - h_end, endpoint=False, dtype=np.float32)
    return env

class BassSynth:
    def __init__(self,
                 freq_hz: float,
                 osc_blend: float = 1.0,   # 0=sine, 1=saw
                 lp_cut: float = 140.0,
                 drive: float = 1.6,
                 level: float = 0.85,
                 lfo_rate: float = 0.0,    # Hz (tremolo)
                 lfo_depth: float = 0.0):  # 0..1
        self.freq = float(freq_hz)
        self.osc_blend = float(np.clip(osc_blend, 0.0, 1.0))
        self.lp_cut = float(lp_cut)
        self.drive = float(drive)
        self.level = float(level)
        self.lfo_rate = float(lfo_rate)
        self.lfo_depth = float(np.clip(lfo_depth, 0.0, 1.0))

        self._lfo_phase = 0.0  # carry phase across steps

    # --- Live setters ---
    def set_root(self, freq_hz: float) -> None:
        self.freq = float(freq_hz)

    def apply_preset(self, lp_cut: float, drive: float, level: float) -> None:
        self.lp_cut = float(lp_cut); self.drive = float(drive); self.level = float(level)

    def apply_custom(self, **kw) -> None:
        for k, v in kw.items():
            if hasattr(self, k):
                setattr(self, k, float(v))

    # --- Rendering ---
    def _osc_blended(self, n: int) -> np.ndarray:
        if self.osc_blend <= 0.0:   # pure sine
            return _sine(self.freq * 2.0, n)      # above sub
        if self.osc_blend >= 1.0:   # pure saw
            return _saw(self.freq * 2.0, n)
        s1 = _sine(self.freq * 2.0, n)
        s2 = _saw(self.freq * 2.0, n)
        return (1.0 - self.osc_blend) * s1 + self.osc_blend * s2

    def _tremolo(self, n: int) -> np.ndarray:
        if self.lfo_depth <= 0.0 or self.lfo_rate <= 0.0:
            return np.ones(n, dtype=np.float32)
        t = (np.arange(n, dtype=np.float32) / SR)
        phase = self._lfo_phase + 2.0 * np.pi * self.lfo_rate * t
        # 0..1 depth around 1.0: (1 - d) .. (1 + d)
        mod = 1.0 + self.lfo_depth * np.sin(phase, dtype=np.float32)
        # wrap phase
        self._lfo_phase = (phase[-1] + 2.0 * np.pi * self.lfo_rate / SR) % (2.0 * np.pi)
        return mod

    def note_segment(self, length_samples: int, style: str = "psy") -> np.ndarray:
        # Gate/shape per style
        if style == "goa":
            on, hold, off = int(0.04*SR), int(0.14*SR), int(0.08*SR)
        else:
            on, hold, off = int(0.01*SR), int(0.05*SR), int(0.06*SR)

        on = min(on, length_samples)
        hold = min(hold, max(0, length_samples - on))
        off = max(0, length_samples - on - hold)

        n = int(length_samples)
        osc = self._osc_blended(n)
        env = _env_pluck(n, on, hold, off)
        trem = self._tremolo(n)

        # Tone → LP → Drive → Level
        voice = one_pole_lowpass(osc * env, self.lp_cut).astype(np.float32)
        voice = saturate_tanh(voice, drive=self.drive).astype(np.float32)
        voice = voice * trem * self.level
        return voice
