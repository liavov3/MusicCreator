"""
Rendering pipeline: place kicks on beats, synth bass, apply simple ducking, mix, and optional stereo widening.
"""

from __future__ import annotations
import numpy as np
from .constants import SR, DEFAULT_HEADROOM_DB
from .synth_kick import psy_kick
from .synth_bass import rolling_bass
from .dsp import normalize, stereo_widen_mono
from .presets import key_to_hz


def _place_kicks(bpm: int, bars: int, kick_wav: np.ndarray) -> np.ndarray:
    """Place the mono kick sample on every beat across the requested number of bars."""
    sec_per_beat = 60.0 / float(bpm)
    sec_per_bar = 4.0 * sec_per_beat
    length_s = bars * sec_per_bar
    n = int(SR * length_s)

    out = np.zeros(n, dtype=float)
    step = int(SR * sec_per_beat)
    klen = len(kick_wav)

    for i in range(0, n, step):
        end = min(i + klen, n)
        out[i:end] += kick_wav[: end - i]

    return out


def _duck_against_kick(bass_track: np.ndarray, bpm: int, amount: float = 0.8, ms: float = 100.0) -> np.ndarray:
    """
    Simple time-domain ducking envelope aligned to each beat.
    'amount' is the lowest multiplier at the start of each beat (0..1).
    """
    sec_per_beat = 60.0 / float(bpm)
    env_len = int(SR * (ms / 1000.0))
    step = int(SR * sec_per_beat)
    duck = np.ones_like(bass_track)

    for i in range(0, len(duck), step):
        end = min(i + env_len, len(duck))
        if end > i:
            duck[i:end] *= np.linspace(amount, 1.0, end - i, endpoint=False)

    return bass_track * duck


def render_loop(style: str = "psy", bpm: int = 145, key: str = "F#", bars: int = 4, stereo: bool = True) -> np.ndarray:
    """
    Render a complete loop (kick + bass).
    Returns stereo (2, N) if `stereo=True`, otherwise mono (N,).
    """
    # Kick by style
    if style.lower() == "goa":
        kick = psy_kick(length_ms=190.0, f_start=80.0, f_end=50.0)
    else:
        kick = psy_kick(length_ms=180.0, f_start=85.0, f_end=48.0)

    kick_track = _place_kicks(bpm, bars, kick)

    # Bass
    root_hz = key_to_hz(key)
    bass_track = rolling_bass(bpm=bpm, root_hz=root_hz, bars=bars, style=style)

    # Duck bass slightly under the kick
    bass_track = _duck_against_kick(bass_track, bpm=bpm, amount=0.2, ms=100.0)

    # Simple mono mix, then optional stereo widening
    mix_mono = normalize(kick_track * 0.9 + bass_track * 0.8, headroom_db=DEFAULT_HEADROOM_DB)
    if stereo:
        return stereo_widen_mono(mix_mono, delay_ms=12.0)
    return mix_mono
