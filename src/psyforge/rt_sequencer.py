"""
Simple 16-step sequencer and transport.
- Keeps musical time (BPM) as sample positions.
- Exposes step onsets so the synth layer can trigger voices.
"""

from __future__ import annotations
import numpy as np
from .constants import SR

class StepSequencer:
    """
    16-step pattern per track. Multiple tracks can be mixed by the synth layer.
    """
    def __init__(self, bpm: int = 145, steps: int = 16):
        self.bpm = int(bpm)
        self.steps = int(steps)
        self.sample_pos = 0               # global sample cursor
        self._set_timing()
        # Patterns are boolean arrays of length `steps`
        self.patterns: dict[str, np.ndarray] = {
            "kick": np.array([1, 0, 0, 0] * 4, dtype=np.int8),        # 4-on-the-floor
            "bass": np.array([1] * steps, dtype=np.int8),             # rolling by default
        }

    def _set_timing(self):
        spb = 60.0 / float(self.bpm)             # seconds per beat
        self.samples_per_beat = int(SR * spb)    # 4/4 time
        self.samples_per_step = self.samples_per_beat // 4  # 16th notes

    def set_bpm(self, bpm: int):
        self.bpm = int(bpm)
        # keep musical position roughly continuous: recompute relative step
        current_step = self.current_step_index()
        self._set_timing()
        # realign cursor to the same step boundary
        self.sample_pos = current_step * self.samples_per_step

    def current_step_index(self) -> int:
        return (self.sample_pos // self.samples_per_step) % self.steps

    def advance(self, frames: int) -> list[tuple[str, int]]:
        """
        Advance the song position by `frames`. Return list of step-triggers:
        [(track_name, step_index), ...] that occur within this block.
        """
        triggers: list[tuple[str, int]] = []
        start_pos = self.sample_pos
        end_pos = self.sample_pos + frames

        # Check every step boundary crossed in this block
        start_step = start_pos // self.samples_per_step
        end_step   = end_pos // self.samples_per_step

        for step_idx in range(start_step + 1, end_step + 1):
            s = step_idx % self.steps
            for track, pat in self.patterns.items():
                if pat[s]:
                    triggers.append((track, s))

        self.sample_pos = end_pos
        return triggers

    def toggle_step(self, track: str, step: int):
        if track not in self.patterns:
            return
        s = step % self.steps
        self.patterns[track][s] = 0 if self.patterns[track][s] else 1

    def set_pattern(self, track: str, arr):
        self.patterns[track] = np.array(arr, dtype=np.int8)[: self.steps]
