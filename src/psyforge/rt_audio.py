"""
Real-time audio engine using sounddevice.
- Single output stream (stereo) with fixed blocksize.
- Pulls audio by calling a provided generator function that returns (2, N) float32 chunks.
"""

from __future__ import annotations
import threading
import numpy as np
import sounddevice as sd
from .constants import SR

class RtAudioEngine:
    """Minimal real-time audio output engine."""
    def __init__(self, chunk_size: int = 512):
        self.chunk_size = int(chunk_size)
        self._lock = threading.RLock()
        self._running = False
        self._stream: sd.OutputStream | None = None
        # The generator must be a callable: (num_frames:int) -> np.ndarray[shape=(2, num_frames)]
        self._audio_callback_fn = None

    def set_audio_callback(self, fn):
        """Register the generator function that produces audio per callback."""
        self._audio_callback_fn = fn

    def start(self):
        """Start the output stream."""
        if self._running:
            return
        if self._audio_callback_fn is None:
            raise RuntimeError("Audio callback not set.")
        self._running = True

        def _cb(outdata, frames, time, status):
            if status:
                # Dropouts/underflows can be inspected here
                pass
            with self._lock:
                block = self._audio_callback_fn(frames)  # (2, frames)
            # Safety & dtype
            if block.ndim == 1:
                block = np.stack([block, block], axis=0)
            out = np.clip(block, -1.0, 1.0).astype(np.float32).T  # (frames, 2)
            outdata[:] = out

        self._stream = sd.OutputStream(
            samplerate=SR, channels=2, dtype="float32",
            callback=_cb, blocksize=self.chunk_size
        )
        self._stream.start()

    def stop(self):
        """Stop the stream."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def is_running(self) -> bool:
        return self._running
