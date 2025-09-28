"""
Musical helpers: note-to-frequency mapping and style defaults.
"""

from __future__ import annotations

NOTE_FREQS = {
    "C": 32.70,  "C#": 34.65, "D": 36.71, "D#": 38.89,
    "E": 41.20,  "F": 43.65,  "F#": 46.25, "G": 49.00,
    "G#": 51.91, "A": 55.00,  "A#": 58.27, "B": 61.74,
}

def key_to_hz(key: str) -> float:
    """Map root key name to low register frequency (approx. octave 1). Defaults to F# if unknown."""
    return NOTE_FREQS.get(key.upper(), 46.25)


def style_defaults(style: str) -> dict:
    """Reasonable defaults per style."""
    if style.lower() == "goa":
        return dict(bpm=142, key="F#", kick_fstart=80.0, kick_fend=50.0)
    return dict(bpm=145, key="F#", kick_fstart=85.0, kick_fend=48.0)
