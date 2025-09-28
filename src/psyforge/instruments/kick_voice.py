"""
Real-time Kick voice with stronger punch options:
- Exponential pitch drop sine for body
- Click transient (adjustable)
- Soft saturation (drive)
- RGB / preset-friendly apply_custom()

Designed to rebuild quickly on param change.
"""

from __future__ import annotations
import numpy as np
from ..dsp import exp_sweep_sine, env_exp, normalize, saturate_tanh
from ..constants import SR


class KickVoice:
    def __init__(self,
                 length_ms: float = 180.0,
                 f_start: float = 85.0,
                 f_end: float = 48.0,
                 click_ms: float = 2.0,
                 body_tau_s: float = 0.06,
                 gain: float = 0.95,
                 drive: float = 1.2,
                 click_gain: float = 1.0):
        self.params = dict(
            length_ms=length_ms, f_start=f_start, f_end=f_end,
            click_ms=click_ms, body_tau_s=body_tau_s,
            gain=gain, drive=drive, click_gain=click_gain
        )
        self._build()

    # ---------- synthesis ----------
    def _build(self) -> None:
        p = self.params
        length_s = float(p["length_ms"]) / 1000.0

        # Body: pitch-swept sine with exponential decay
        body = exp_sweep_sine(p["f_start"], p["f_end"], length_s)
        body *= env_exp(length_s, tau_s=float(p["body_tau_s"]))
        body *= float(p["gain"])

        # Click: short Hann window scaled by click_gain
        click_len = int(SR * (float(p["click_ms"]) / 1000.0))
        click = np.zeros_like(body, dtype=np.float32)
        if click_len > 0:
            click[:click_len] = np.hanning(2 * click_len)[:click_len] * float(p["click_gain"])

        # Sum + soft saturation for punch
        x = (body + click).astype(np.float32)
        x = saturate_tanh(x, drive=float(p["drive"])).astype(np.float32)

        # Keep headroom; overall loudness will be controlled by mixer gain (g_kick)
        self.buf = normalize(x, headroom_db=6.0).astype(np.float32)

    # ---------- API ----------
    def apply_preset(self, **kwargs) -> None:
        self.params.update(kwargs); self._build()

    def apply_custom(self, **kwargs) -> None:
        """Set any of: length_ms, f_start, f_end, click_ms, body_tau_s, gain, drive, click_gain."""
        for k, v in kwargs.items():
            if k in self.params:
                self.params[k] = float(v)
        self._build()

    def trigger(self) -> np.ndarray:
        return self.buf.copy()
