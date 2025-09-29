"""
PsyForge Live – Kick/Bass/Acid + Drums + Delay + RGB + Scenes + Recording + Sound Painter
Run:
    python src/ui_live.py
"""

from __future__ import annotations
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import numpy as np

# ---- Optional recording ----
try:
    import soundfile as sf  # pip install soundfile
except Exception:
    sf = None

# ---- Engine + instruments (from your project) ----
from psyforge.constants import SR
from psyforge.rt_audio import RtAudioEngine
from psyforge.rt_sequencer import StepSequencer
from psyforge.instruments.kick_voice import KickVoice
from psyforge.instruments.bass_voice import BassSynth
from psyforge.instruments.acid303 import Acid303
from psyforge.presets import NOTE_FREQS

# ---- RGB mapping (with safe fallbacks) ----
try:
    from psyforge.color_sound import (
        rgb_to_bass_params, rgb_to_kick_params, rgb_to_hex
    )
except Exception:
    def rgb_to_hex(r,g,b): return f"#{int(r)&255:02X}{int(g)&255:02X}{int(b)&255:02X}"
    class _B: pass
    def rgb_to_bass_params(r,g,b):
        p=_B(); p.osc_blend=r/255; p.lp_cut=80+(g/255)*260; p.drive=1+(r/255)*1.8
        p.level=0.85; p.lfo_rate=0.2+(b/255)*8; p.lfo_depth=(b/255)*0.4; return p
    class _K: pass
    def rgb_to_kick_params(r,g,b):
        p=_K(); p.length_ms=180; p.f_start=85+(g/255)*18; p.f_end=50-(b/255)*7
        p.click_ms=1+(g/255)*2; p.body_tau_s=0.06; p.gain=0.95; p.drive=1+(r/255)*1.6
        p.click_gain=0.6+(r/255)*1.6; return p

# =============================================================================
# Lightweight synthesized drums (NO samples): Hat & Clap
# =============================================================================

class HatSynth:
    """Short bright tick (noise, highpass, very fast decay)."""
    def __init__(self, length_ms: float = 80.0, hp_cut: float = 4000.0, level: float = 0.8):
        self.length_ms = float(length_ms); self.hp_cut = float(hp_cut); self.level = float(level)

    def trigger(self) -> np.ndarray:
        n = max(1, int(SR * self.length_ms / 1000.0))
        noise = np.random.randn(n).astype(np.float32) * 0.4
        # highpass via (x - one-pole LP)
        alpha = float(np.exp(-2.0 * np.pi * self.hp_cut / SR))
        lp = np.zeros(n, dtype=np.float32)
        for i in range(1, n):
            lp[i] = alpha * lp[i-1] + (1.0 - alpha) * noise[i]
        hp = (noise - lp).astype(np.float32)
        # fast decay ~20ms
        t = np.arange(n, dtype=np.float32) / SR
        env = np.exp(-t / 0.020, dtype=np.float32)
        return (hp * env * self.level).astype(np.float32)

class ClapSynth:
    """
    Smooth 808-ish clap:
    - Band-pass noise (HP -> LP) to avoid harsh top/DC
    - 4 burst-envelopes for the “clap cluster”
    - Tiny dither to kill denormals
    - Short fade in/out to avoid clicks
    """
    def __init__(self,
                 length_ms: float = 140.0,
                 bursts_ms: tuple[float, ...] = (0.0, 12.0, 22.0, 36.0),
                 hp_cut: float = 1200.0,
                 lp_cut: float = 6500.0,
                 decay_s: float = 0.080,
                 level: float = 0.85):
        self.length_ms = float(length_ms)
        self.bursts_ms = tuple(float(x) for x in bursts_ms)
        self.hp_cut = float(hp_cut)
        self.lp_cut = float(lp_cut)
        self.decay_s = float(decay_s)
        self.level = float(level)

    def _onepole_lp(self, x: np.ndarray, fc: float) -> np.ndarray:
        alpha = float(np.exp(-2.0 * np.pi * fc / SR))
        y = np.empty_like(x)
        acc = 0.0
        a1 = alpha
        b0 = (1.0 - alpha)
        for i in range(x.shape[0]):
            acc = a1 * acc + b0 * float(x[i])
            y[i] = acc
        return y

    def _onepole_hp(self, x: np.ndarray, fc: float) -> np.ndarray:
        return (x - self._onepole_lp(x, fc)).astype(np.float32)

    def trigger(self) -> np.ndarray:
        n = max(1, int(SR * self.length_ms / 1000.0))
        noise = np.random.randn(n).astype(np.float32)

        # band-pass
        x = self._onepole_hp(noise, self.hp_cut)
        x = self._onepole_lp(x, self.lp_cut)

        # burst envelopes
        env = np.zeros(n, dtype=np.float32)
        weights = [1.0, 0.82, 0.66, 0.55]
        for j, ms in enumerate(self.bursts_ms):
            off = int(SR * ms / 1000.0)
            if off >= n: continue
            m = n - off
            t = np.arange(m, dtype=np.float32) / SR
            e = np.exp(-t / self.decay_s, dtype=np.float32) * (weights[j] if j < len(weights) else 0.5)
            env[off:off+m] += e

        x *= env

        # fades
        fade = min(64, n // 20)
        if fade > 0:
            w = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            x[:fade] *= w; x[-fade:] *= w[::-1]

        # dither tiny
        x += (1e-8 * np.random.randn(n)).astype(np.float32)

        return (x * self.level).astype(np.float32)

# =============================================================================
# FX: Stereo Delay (mono send -> stereo return)
# =============================================================================

class StereoDelay:
    def __init__(self, time_ms: float = 240.0, feedback: float = 0.35):
        self.SR = SR
        self.set_params(time_ms, feedback)
        self.buf = np.zeros(self.delay_samples, dtype=np.float32)
        self.idx = 0

    def set_params(self, time_ms: float, feedback: float):
        self.time_ms = float(time_ms)
        self.feedback = float(np.clip(feedback, 0.0, 0.95))
        self.delay_samples = max(1, int(self.SR * self.time_ms / 1000.0))

    def _resize_if_needed(self):
        if len(self.buf) != self.delay_samples:
            new = np.zeros(self.delay_samples, dtype=np.float32)
            copy = min(len(self.buf), len(new))
            new[:copy] = self.buf[:copy]
            self.buf = new
            self.idx %= self.delay_samples

    def process_send(self, mono_in: np.ndarray, send_level: float) -> np.ndarray:
        self._resize_if_needed()
        out = np.zeros((2, len(mono_in)), dtype=np.float32)
        idx = self.idx; fb = self.feedback; buf = self.buf
        for i in range(len(mono_in)):
            y = buf[idx]
            buf[idx] = y * fb + float(mono_in[i]) * float(send_level)
            idx += 1
            if idx >= len(buf): idx = 0
            out[0, i] = y; out[1, i] = y
        self.idx = idx
        return out


class PainterLayer:
    """One freehand-drawn oscillator layer."""
    def __init__(self, name="Layer 1", color="#39C2FF"):
        # user params
        self.name = name
        self.color = color
        self.enabled = False
        self.mode = "sine"   # "sine" or "saw"
        self.quantize = False
        self.gain = 0.9
        self.fmin = 80.0
        self.fmax = 2400.0
        self.beats = 4

        # curves (per-sample)
        self.freq = None     # np.float32
        self.amp  = None     # np.float32
        self.loop_len = 0

        # runtime
        self._pos = 0
        self._phase = 0.0

# =============================================================================
# Synth graph & audio callback
# =============================================================================

class LiveSynthGraph:
    """
    Manages instruments, patterns, FX, ducking, scenes, meters, recording & Sound Painter.
    """
    def __init__(self, seq: StepSequencer, style: str = "psy", key: str = "F#"):
        self.seq = seq
        self.style = style
        self.key = key

        root_hz = NOTE_FREQS.get(key.upper(), 46.25)

        # Instruments
        self.kick = KickVoice()
        self.bass = BassSynth(freq_hz=root_hz, lp_cut=140.0, drive=1.6, level=0.85)
        self.acid = Acid303(root_hz=root_hz, cutoff_hz=900.0, resonance=0.88,
                            env_mod_hz=1400.0, slide_ms=35.0,
                            accent_gain=1.6, drive=1.4, level=0.9)
        self.hat  = HatSynth()
        self.clap = ClapSynth()

        # Mixer gains
        self.g_kick = 1.60
        self.g_bass = 0.85
        self.g_acid = 0.95
        self.g_hat  = 0.70
        self.g_clap = 0.85

        # FX
        self.delay = StereoDelay(time_ms=240.0, feedback=0.35)
        self.acid_send = 0.25

        # Scenes
        self.scenes: dict[str, dict[str, np.ndarray]] = {}
        self.queued_scene: str | None = None

        # Ducking (kick -> bass/acid)
        self.duck_value = 0.0
        self.duck_decay = np.exp(-1.0 / (0.060 * SR))  # ~60ms decay
        self.duck_depth = 0.30

        # Recording & meters
        self.recording = False
        self._writer = None
        self.meter_peak = 0.0
        self.meter_rms = 0.0

        # Bar/Beat UI
        self.bar_count = 1
        self.current_step = 0

        # Active voices
        self.active: list[dict] = []

        # helpers
        self._rng = np.random.default_rng(42)
        self.acid_octave = 0
        self.acid_pattern = "Root"

        # --- Sound Painter state ---
        # self.painter_enabled = False
        # self.painter_gain = 0.9
        # self.painter_mode = "sine"  # 'sine' or 'saw'
        # self.painter_freq = None    # np.ndarray
        # self.painter_amp = None     # np.ndarray
        # self.painter_loop_len = 0
        # self._painter_pos = 0
        # self._painter_phase = 0.0

        # --- Sound Painter Layers ---
        self.layers: list[PainterLayer] = [PainterLayer("Layer 1", "#39C2FF")]


    # --- setters / glue ---
    def set_style(self, s: str): self.style = s
    def set_key(self, k: str):
        self.key = k
        hz = NOTE_FREQS.get(k.upper(), 46.25)
        self.bass.set_root(hz); self.acid.set_root(hz)

    def set_kick_gain(self, v: float): self.g_kick = float(v)
    def set_bass_gain(self, v: float): self.g_bass = float(v)
    def set_acid_gain(self, v: float): self.g_acid = float(v)

    def set_bass_lp(self, v: float): self.bass.lp_cut = float(v)
    def set_bass_drive(self, v: float): self.bass.drive = float(v)

    def set_acid_cutoff(self, v: float): self.acid.set_cutoff(float(v))
    def set_acid_resonance(self, v: float): self.acid.set_resonance(float(v))
    def set_acid_envmod(self, v: float): self.acid.set_env_mod(float(v))
    def set_acid_slide_ms(self, v: float): self.acid.set_slide_ms(float(v))
    def set_acid_accent_gain(self, v: float): self.acid.set_accent_gain(float(v))
    def set_acid_octave(self, n: int): self.acid_octave = int(n)
    def set_acid_pattern(self, name: str): self.acid_pattern = name
    def set_delay_time(self, ms: float): self.delay.set_params(time_ms=float(ms), feedback=self.delay.feedback)
    def set_delay_feedback(self, fb: float): self.delay.set_params(time_ms=self.delay.time_ms, feedback=float(fb))
    def set_delay_send(self, s: float): self.acid_send = float(s)

    # --- RGB mapping ---
    def apply_bass_rgb(self, r: int, g: int, b: int):
        p = rgb_to_bass_params(r, g, b)
        self.bass.apply_custom(
            osc_blend=p.osc_blend, lp_cut=p.lp_cut, drive=p.drive,
            level=p.level, lfo_rate=p.lfo_rate, lfo_depth=p.lfo_depth
        )
        self.g_bass = p.level
        return p

    def apply_kick_rgb(self, r: int, g: int, b: int):
        p = rgb_to_kick_params(r, g, b)
        self.kick.apply_custom(
            length_ms=p.length_ms, f_start=p.f_start, f_end=p.f_end,
            click_ms=p.click_ms, body_tau_s=p.body_tau_s,
            gain=p.gain, drive=p.drive, click_gain=p.click_gain
        )
        return p

    # --- scenes, recording ---
    def start_record(self, path: Path):
        if sf is None: return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = sf.SoundFile(str(path), mode="w", samplerate=SR, channels=2)
            self.recording = True
            return True
        except Exception:
            self._writer = None; self.recording = False
            return False

    def stop_record(self):
        if self._writer is not None:
            try: self._writer.close()
            except Exception: pass
        self._writer = None; self.recording = False

    def _apply_scene_now(self, scene: dict[str, np.ndarray]) -> None:
        for k, v in scene.items():
            if k in self.seq.patterns and len(self.seq.patterns[k]) == len(v):
                self.seq.patterns[k] = v.copy()

    # --- Sound Painter API ---
    def set_painter_enabled(self, flag: bool): self.painter_enabled = bool(flag)
    def set_painter_gain(self, v: float): self.painter_gain = float(max(0.0, v))
    def set_painter_mode(self, name: str): self.painter_mode = name

    def set_painter_curve(self, freq_curve_hz: np.ndarray, amp_curve: np.ndarray, loop_samples: int):
        if loop_samples <= 0:
            self.painter_freq = None; self.painter_amp = None; self.painter_loop_len = 0
            return
        self.painter_loop_len = int(loop_samples)
        self.painter_freq = np.asarray(freq_curve_hz, dtype=np.float32).reshape(-1)
        self.painter_amp  = np.asarray(amp_curve, dtype=np.float32).reshape(-1)
        n = min(len(self.painter_freq), len(self.painter_amp), self.painter_loop_len)
        self.painter_freq = self.painter_freq[:n]
        self.painter_amp  = self.painter_amp[:n]
        self.painter_loop_len = n
        self._painter_pos = 0
        self._painter_phase = 0.0

    # --- helpers ---
    def _acid_semitone_for_step(self, s: int) -> int:
        if self.acid_pattern == "Root": return 0
        if self.acid_pattern == "UpDown":
            seq = [0,3,5,7,10,7,5,3,0,-2,-5,-7,-5,-2,0,3]
            return seq[s % 16]
        return int(self._rng.choice([0,2,3,5,7,9,10,12,-2,-5]))
    
    def painter_add_layer(self, name: str, color: str) -> int:
        self.layers.append(PainterLayer(name, color))
        return len(self.layers) - 1

    def painter_remove_layer(self, idx: int):
        if 0 <= idx < len(self.layers) and len(self.layers) > 1:
            self.layers.pop(idx)

    def painter_set_curve(self, idx: int, freq: np.ndarray, amp: np.ndarray, loop_len: int):
        if not (0 <= idx < len(self.layers)): return
        L = self.layers[idx]
        if loop_len <= 0:
            L.freq = None; L.amp = None; L.loop_len = 0; L._pos = 0; L._phase = 0.0
            return
        L.freq = np.asarray(freq, dtype=np.float32).reshape(-1)
        L.amp  = np.asarray(amp, dtype=np.float32).reshape(-1)
        n = min(len(L.freq), len(L.amp), int(loop_len))
        L.freq = L.freq[:n]; L.amp = L.amp[:n]; L.loop_len = n; L._pos = 0; L._phase = 0.0

    def painter_set_enabled(self, idx: int, flag: bool):
        if 0 <= idx < len(self.layers): self.layers[idx].enabled = bool(flag)

    def painter_update_params(self, idx: int, *, mode=None, gain=None, fmin=None, fmax=None, beats=None, quantize=None):
        if not (0 <= idx < len(self.layers)): return
        L = self.layers[idx]
        if mode is not None: L.mode = mode
        if gain is not None: L.gain = float(gain)
        if fmin is not None: L.fmin = float(fmin)
        if fmax is not None: L.fmax = float(fmax)
        if beats is not None: L.beats = int(beats)
        if quantize is not None: L.quantize = bool(quantize)

    # --- audio callback ---
    def render(self, frames: int) -> np.ndarray:
        out = np.zeros((2, frames), dtype=np.float32)
        triggers = self.seq.advance(frames)
        step_len = self.seq.samples_per_step

        kick_in_block = False
        step0_in_block = False

        for track, step in triggers:
            self.current_step = int(step)
            if step == 0:
                step0_in_block = True
            if track == "kick":
                self.active.append({"buf": self.kick.trigger(), "pos": 0, "trk": "kick"})
                kick_in_block = True
            elif track == "bass":
                seg = self.bass.note_segment(step_len, style=self.style)
                self.active.append({"buf": seg, "pos": 0, "trk": "bass"})
            elif track == "acid":
                acc = bool(self.seq.patterns.get("acid_acc", np.zeros(16, dtype=np.int8))[step % 16])
                sld = bool(self.seq.patterns.get("acid_sld", np.zeros(16, dtype=np.int8))[step % 16])
                root = NOTE_FREQS.get(self.key.upper(), 46.25)
                semi = self._acid_semitone_for_step(step) + 12 * int(self.acid_octave)
                hz = root * (2.0 ** (semi / 12.0))
                seg = self.acid.note_segment(step_len, target_hz=hz, slide=sld, accent=acc)
                self.active.append({"buf": seg, "pos": 0, "trk": "acid"})
            elif track == "hat":
                self.active.append({"buf": self.hat.trigger(), "pos": 0, "trk": "hat"})
            elif track == "clap":
                self.active.append({"buf": self.clap.trigger(), "pos": 0, "trk": "clap"})

        # scene switch at bar boundary
        if step0_in_block:
            self.bar_count += 1
            if self.queued_scene and self.queued_scene in self.scenes:
                self._apply_scene_now(self.scenes[self.queued_scene])
                self.queued_scene = None

        # sidechain envelope for this block (exp decay)
        duck = np.empty(frames, dtype=np.float32)
        v = self.duck_value
        for i in range(frames):
            duck[i] = v
            v *= self.duck_decay
        self.duck_value = v
        if kick_in_block:
            self.duck_value = 1.0

        rm = []
        acid_dry = np.zeros(frames, dtype=np.float32)
        for i, vce in enumerate(self.active):
            buf, pos, trk = vce["buf"], vce["pos"], vce["trk"]
            take = min(len(buf) - pos, frames)
            if take > 0:
                seg = buf[pos:pos+take]
                if trk == "kick":
                    g = self.g_kick
                elif trk == "bass":
                    g = self.g_bass * (1.0 - self.duck_depth * duck[:take])
                    out[0, :take] += seg * g; out[1, :take] += seg * g
                    vce["pos"] += take
                    if vce["pos"] >= len(buf): rm.append(i)
                    continue
                elif trk == "acid":
                    g = self.g_acid * (1.0 - self.duck_depth * duck[:take])
                    acid_dry[:take] += seg
                    out[0, :take] += seg * g; out[1, :take] += seg * g
                    vce["pos"] += take
                    if vce["pos"] >= len(buf): rm.append(i)
                    continue
                elif trk == "hat":
                    g = self.g_hat
                elif trk == "clap":
                    g = self.g_clap
                else:
                    g = 1.0
                out[0, :take] += seg * g; out[1, :take] += seg * g
                vce["pos"] += take
            if vce["pos"] >= len(buf): rm.append(i)
        for i in reversed(rm): self.active.pop(i)

        # FX return
        if self.acid_send > 1e-4:
            out += self.delay.process_send(acid_dry, self.acid_send)

        # --- Sound Painter mixing ---
        # if self.painter_enabled and self.painter_loop_len > 0 and self.painter_freq is not None:
        #     n = frames
        #     loop_len = self.painter_loop_len
        #     freq = self.painter_freq
        #     amp = self.painter_amp
        #     pos = self._painter_pos
        #     phase = self._painter_phase
        #     g = float(self.painter_gain)
        #     for i in range(n):
        #         idx = pos % loop_len
        #         f = float(freq[idx])
        #         a = float(amp[idx]) * g
        #         phase += (2.0 * np.pi) * (f / SR)
        #         if phase >= 2.0 * np.pi:
        #             phase -= 2.0 * np.pi
        #         if self.painter_mode == "saw":
        #             sample = (phase / np.pi) - 1.0
        #         else:
        #             sample = np.sin(phase, dtype=np.float32)
        #         s = sample * a
        #         out[0, i] += s; out[1, i] += s
        #         pos += 1
        #     self._painter_pos = pos % loop_len
        #     self._painter_phase = phase

        # --- Sound Painter: mix all enabled layers ---
        for L in self.layers:
            if not (L.enabled and L.loop_len > 0 and L.freq is not None): 
                continue
            n = frames; loop_len = L.loop_len
            freq = L.freq; amp = L.amp
            pos = L._pos; phase = L._phase; g = float(L.gain)
            for i in range(n):
                idx = pos % loop_len
                f = float(freq[idx]); a = float(amp[idx]) * g
                phase += (2.0 * np.pi) * (f / SR)
                if phase >= 2.0 * np.pi: phase -= 2.0 * np.pi
                if L.mode == "saw":
                    sample = (phase / np.pi) - 1.0
                else:
                    sample = np.sin(phase, dtype=np.float32)
                s = sample * a
                out[0, i] += s; out[1, i] += s
                pos += 1
            L._pos = pos % loop_len; L._phase = phase


        # meters
        peak = float(np.max(np.abs(out))) + 1e-9
        self.meter_peak = 0.9 * self.meter_peak + 0.1 * peak
        rms = float(np.sqrt(np.mean(out**2))) + 1e-9
        self.meter_rms = 0.9 * self.meter_rms + 0.1 * rms

        # soft limiter
        out = np.tanh(out * 1.2, dtype=np.float32)

        # recording
        if self.recording and self._writer is not None:
            try: self._writer.write(out.T)
            except Exception: pass

        return out

# =============================================================================
# UI
# =============================================================================

class LiveUI(ttk.Frame):

    # Colors
    STEP_ON_BG  = "#2D7DFF"
    STEP_OFF_BG = "#F4F4F6"
    STEP_ON_FG  = "#FFFFFF"
    STEP_OFF_FG = "#A0A4AB"
    STEP_BORDER = "#C9CDD3"

    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=12)
        # create early so handlers can safely use them
        self.var_status = tk.StringVar(value="Ready.")
        self.var_meters = tk.StringVar(value="")

        master.title("PsyForge Live – Kick/Bass/Acid + Delay + RGB + Painter")
        master.minsize(1240, 860)

        # Sequencer + default patterns
        self.seq = StepSequencer(bpm=145, steps=16)
        for k in ["kick", "bass", "acid", "acid_acc", "acid_sld", "hat", "clap"]:
            if k not in self.seq.patterns:
                self.seq.patterns[k] = np.zeros(16, dtype=np.int8)
        # nice defaults
        self.seq.patterns["acid"][:] = np.array([1,0]*8, dtype=np.int8)      # acid offbeat
        self.seq.patterns["hat"][:]  = np.array([0,1]*8, dtype=np.int8)      # hat offbeat
        clap = np.zeros(16, dtype=np.int8); clap[4] = 1; self.seq.patterns["clap"][:] = clap

        self.graph = LiveSynthGraph(self.seq, style="psy", key="F#")
        self.engine = RtAudioEngine(chunk_size=512)
        self.engine.set_audio_callback(self.graph.render)
        self.playing = False

        # Build UI
        self._build_transport()
        self._build_grid()
        self._build_mixer()
        self._build_acid_controls()
        self._build_delay_controls()
        self._build_rgb_designer()
        self._build_painter_panel()
        self._build_status()

        # layout
        self.grid(sticky="nsew")
        # 2-column layout: main (col 0) + right sidebar (col 1)
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)

        master.columnconfigure(0, weight=1); master.rowconfigure(0, weight=1)

    # ---------------- Transport (Play, REC, Tap, Style, Key, Scenes, BPM readout) ----------------
    def _build_transport(self):
        fx = ttk.Frame(self); fx.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0,8))
        for c in range(16): fx.columnconfigure(c, weight=1)

        self.btn_play = ttk.Button(fx, text="Play", command=self._toggle_play)
        self.btn_play.grid(row=0, column=0, padx=(0,8))

        self.btn_rec  = ttk.Button(fx, text="REC",  command=self._toggle_record)
        self.btn_rec.grid(row=0, column=1, padx=(0,12))

        ttk.Label(fx, text="BPM").grid(row=0, column=2, sticky="e")
        self.var_bpm = tk.IntVar(value=145)
        self.lbl_bpm_val = ttk.Label(fx, textvariable=self.var_bpm, width=4, anchor="w")
        self.lbl_bpm_val.grid(row=0, column=3, sticky="w", padx=(6,6))

        s = ttk.Scale(
            fx, from_=30, to=230, orient="horizontal",
            command=lambda v: self._on_bpm_change(int(float(v)))
        )
        s.set(self.var_bpm.get())
        s.grid(row=0, column=4, sticky="ew", padx=(6,12))

        self.btn_tap = ttk.Button(fx, text="Tap", command=self._tap_tempo)
        self.btn_tap.grid(row=0, column=5, padx=(0,12))

        ttk.Label(fx, text="Style").grid(row=0, column=6, sticky="e")
        self.var_style = tk.StringVar(value="psy")
        ttk.Radiobutton(fx, text="Psy", value="psy", variable=self.var_style, command=self._on_style_change).grid(row=0, column=7, sticky="w")
        ttk.Radiobutton(fx, text="Goa", value="goa", variable=self.var_style, command=self._on_style_change).grid(row=0, column=8, sticky="w")

        ttk.Label(fx, text="Key").grid(row=0, column=9, sticky="e")
        self.var_key = tk.StringVar(value="F#")
        cmb = ttk.Combobox(fx, textvariable=self.var_key, values=list(NOTE_FREQS.keys()), width=4, state="readonly")
        cmb.grid(row=0, column=10, sticky="w"); cmb.bind("<<ComboboxSelected>>", lambda e: self._on_key_change())

        # Scenes A..D
        for j, name in enumerate(["A", "B", "C", "D"]):
            ttk.Button(fx, text=f"Queue {name}", command=lambda n=name: self._queue_scene(n)).grid(row=0, column=11+j, padx=(6,2))

        # Bar/Beat display (right side)
        self.var_pos = tk.StringVar(value="Bar 1 • Beat 1.1")
        ttk.Label(fx, textvariable=self.var_pos).grid(row=0, column=15, sticky="e", padx=(6,0))

    def _tap_tempo(self):
        now = time.time()
        if not hasattr(self, "_tap_times"): self._tap_times = []
        self._tap_times = [t for t in self._tap_times if now - t < 3.0]
        self._tap_times.append(now)
        if len(self._tap_times) >= 2:
            intervals = [self._tap_times[i+1]-self._tap_times[i] for i in range(len(self._tap_times)-1)]
            avg = max(0.05, sum(intervals) / len(intervals))
            bpm = int(round(60.0 / avg))
            bpm = max(30, min(230, bpm))
            self._on_bpm_change(bpm)

    def _toggle_record(self):
        if self.graph.recording:
            self.graph.stop_record()
            self.btn_rec.configure(text="REC"); self.var_status.set("Recording stopped.")
        else:
            if sf is None:
                self.var_status.set("Install 'soundfile' to enable recording: pip install soundfile"); return
            outdir = Path("outputs"); outdir.mkdir(exist_ok=True)
            fname = time.strftime("live_%Y%m%d_%H%M%S.wav")
            ok = self.graph.start_record(outdir / fname)
            if ok:
                self.btn_rec.configure(text="■ Stop"); self.var_status.set(f"Recording → outputs/{fname}")
            else:
                self.var_status.set("Failed to start recording.")

    def _queue_scene(self, name: str):
        if name not in self.graph.scenes:
            self._store_scene(name)
        self.graph.queued_scene = name
        self.var_status.set(f"Scene {name} queued (applies at bar start).")

    def _store_scene(self, name: str):
        scene = {k: v.copy() for k, v in self.seq.patterns.items()}
        self.graph.scenes[name] = scene
        self.var_status.set(f"Scene {name} stored.")

    # ---------------- Sequencer Grid (steps + tools) ----------------
    # --- Step cell visuals (pretty grid boxes) ---
    def _step_update(self, lbl: tk.Label, val: int):
        """Paint a step cell as ON/OFF."""
        if val:
            lbl.configure(bg=self.STEP_ON_BG, fg=self.STEP_ON_FG, text="✓")
        else:
            lbl.configure(bg=self.STEP_OFF_BG, fg=self.STEP_OFF_FG, text="")
        lbl.configure(highlightthickness=0, bd=1, relief="solid")

    def _make_step_cell(self, parent, track: str, step_idx: int, init_val: int):
        """Create a clickable square step cell bound to sequencer pattern."""
        var = tk.IntVar(value=int(init_val))
        lbl = tk.Label(
            parent, width=2, height=1,
            bg=self.STEP_OFF_BG, fg=self.STEP_OFF_FG,
            bd=1, relief="solid", cursor="hand2"
        )
        self._step_update(lbl, var.get())

        def toggle(_evt=None):
            var.set(0 if var.get() else 1)
            self.seq.patterns[track][step_idx] = var.get()
            self._step_update(lbl, var.get())

        lbl.bind("<Button-1>", toggle)
        return var, lbl

    def _build_grid(self):
        g = ttk.LabelFrame(self, text="Sequencer (click to toggle steps)")
        g.grid(row=1, column=0, sticky="nsew")
        self.rowconfigure(1, weight=1)

        for c in range(1, 17):
            g.columnconfigure(c, weight=1, uniform="steps")

        for c in (0, 17, 18, 19, 20, 21, 22): g.columnconfigure(c, weight=0)

        labels = [("Kick", "kick"), ("Bass", "bass"), ("Acid", "acid"),
                  ("Accent", "acid_acc"), ("Slide", "acid_sld"),
                  ("Hat", "hat"), ("Clap", "clap")]
        self.step_btns = {k: [] for _, k in labels}
        self.step_vars = {k: [] for _, k in labels}

        for r, (title, track) in enumerate(labels):
            ttk.Label(g, text=title).grid(row=r, column=0, sticky="w", padx=10)
            pattern = self.seq.patterns[track]
            for c in range(16):
                var, cell = self._make_step_cell(g, track, c, int(pattern[c]))
                cell.grid(row=r, column=c+1, padx=3, pady=6, sticky="nsew")
                self.step_vars[track].append(var)
                self.step_btns[track].append(cell)

            ttk.Button(g, text="Fill All",  command=lambda t=track: self._fill_row(t)).grid(row=r, column=17, padx=(8,4), pady=6, sticky="e")
            ttk.Button(g, text="Clear All", command=lambda t=track: self._clear_row(t)).grid(row=r, column=18, padx=(4,4), pady=6, sticky="w")
            ttk.Button(g, text="Invert",    command=lambda t=track: self._invert_row(t)).grid(row=r, column=19, padx=(4,4), pady=6, sticky="w")
            ttk.Button(g, text="Every 2",   command=lambda t=track: self._pattern_every_n(t, 2)).grid(row=r, column=20, padx=(4,4), pady=6, sticky="w")
            ttk.Button(g, text="Every 3",   command=lambda t=track: self._pattern_every_n(t, 3)).grid(row=r, column=21, padx=(4,4), pady=6, sticky="w")
            ttk.Button(g, text="Every 4",   command=lambda t=track: self._pattern_every_n(t, 4)).grid(row=r, column=22, padx=(4,10), pady=6, sticky="w")

    def _toggle_step(self, track: str, step: int, btn=None):
        pat = self.seq.patterns.get(track)
        var_list = self.step_vars.get(track)
        if pat is None or var_list is None: return
        pat[step] = 1 if int(var_list[step].get()) else 0

    def _fill_row(self, track: str):
        pat = self.seq.patterns.get(track); vars_list = self.step_vars.get(track); cells = self.step_btns.get(track)
        if pat is None or vars_list is None: return
        for i in range(len(pat)):
            pat[i] = 1; vars_list[i].set(1); self._step_update(cells[i], 1)

    def _clear_row(self, track: str):
        pat = self.seq.patterns.get(track); vars_list = self.step_vars.get(track); cells = self.step_btns.get(track)
        if pat is None or vars_list is None: return
        for i in range(len(pat)):
            pat[i] = 0; vars_list[i].set(0); self._step_update(cells[i], 0)

    def _invert_row(self, track: str):
        pat = self.seq.patterns.get(track); vars_list = self.step_vars.get(track); cells = self.step_btns.get(track)
        if pat is None or vars_list is None: return
        for i in range(len(pat)):
            v = 0 if pat[i] else 1
            pat[i] = v; vars_list[i].set(v); self._step_update(cells[i], v)

    def _pattern_every_n(self, track: str, n: int, start: int = 0):
        pat = self.seq.patterns.get(track); vars_list = self.step_vars.get(track); cells = self.step_btns.get(track)
        if pat is None or vars_list is None: return
        for i in range(len(pat)):
            val = 1 if ((i - start) % n == 0) else 0
            pat[i] = val; vars_list[i].set(val); self._step_update(cells[i], val)


    # ---------------- Mixer & Bass tone ----------------
    def _build_mixer(self):
        box = ttk.LabelFrame(self, text="Mixer & Bass Tone")
        box.grid(row=2, column=0, sticky="ew", pady=(8,0))

        ttk.Label(box, text="Kick Level").grid(row=0, column=0, padx=(10,6), sticky="w")
        s = ttk.Scale(box, from_=0.0, to=3.0, orient="horizontal",
                      command=lambda v: self.graph.set_kick_gain(float(v)))
        s.set(self.graph.g_kick); s.grid(row=0, column=1, sticky="ew", padx=(4,16))

        ttk.Label(box, text="Bass Level").grid(row=0, column=2, padx=(10,6), sticky="w")
        s2 = ttk.Scale(box, from_=0.0, to=1.8, orient="horizontal",
                       command=lambda v: self.graph.set_bass_gain(float(v)))
        s2.set(self.graph.g_bass); s2.grid(row=0, column=3, sticky="ew", padx=(4,16))

        ttk.Label(box, text="Bass LP (Hz)").grid(row=0, column=4, sticky="e")
        s3 = ttk.Scale(box, from_=80.0, to=340.0, orient="horizontal",
                       command=lambda v: self.graph.set_bass_lp(float(v)))
        s3.set(self.graph.bass.lp_cut); s3.grid(row=0, column=5, sticky="ew", padx=(4,16))

        ttk.Label(box, text="Bass Drive").grid(row=0, column=6, sticky="e")
        s4 = ttk.Scale(box, from_=0.8, to=2.8, orient="horizontal",
                       command=lambda v: self.graph.set_bass_drive(float(v)))
        s4.set(self.graph.bass.drive); s4.grid(row=0, column=7, sticky="ew", padx=(4,16))

        ttk.Label(box, text="Acid Level").grid(row=1, column=0, padx=(10,6), sticky="w")
        s5 = ttk.Scale(box, from_=0.0, to=1.8, orient="horizontal",
                       command=lambda v: self.graph.set_acid_gain(float(v)))
        s5.set(self.graph.g_acid); s5.grid(row=1, column=1, sticky="ew", padx=(4,16))

        ttk.Label(box, text="Hat Level").grid(row=1, column=2, padx=(10,6), sticky="w")
        s6 = ttk.Scale(box, from_=0.0, to=1.8, orient="horizontal",
                       command=lambda v: setattr(self.graph, "g_hat", float(v)))
        s6.set(self.graph.g_hat); s6.grid(row=1, column=3, sticky="ew", padx=(4,16))

        ttk.Label(box, text="Clap Level").grid(row=1, column=4, padx=(10,6), sticky="w")
        s7 = ttk.Scale(box, from_=0.0, to=1.8, orient="horizontal",
                       command=lambda v: setattr(self.graph, "g_clap", float(v)))
        s7.set(self.graph.g_clap); s7.grid(row=1, column=5, sticky="ew", padx=(4,16))

        for c in range(0, 8): box.columnconfigure(c, weight=1)

    # ---------------- Acid controls ----------------
    def _build_acid_controls(self):
        box = ttk.LabelFrame(self, text="Acid 303 Controls")
        box.grid(row=3, column=0, sticky="ew", pady=(8,0))

        ttk.Label(box, text="Cutoff (Hz)").grid(row=0, column=0, sticky="e", padx=(10,6))
        s1 = ttk.Scale(box, from_=200.0, to=3500.0, orient="horizontal",
                       command=lambda v: self.graph.set_acid_cutoff(float(v)))
        s1.set(self.graph.acid.cutoff); s1.grid(row=0, column=1, sticky="ew", padx=(4,16))

        ttk.Label(box, text="Resonance").grid(row=0, column=2, sticky="e")
        s2 = ttk.Scale(box, from_=0.5, to=1.2, orient="horizontal",
                       command=lambda v: self.graph.set_acid_resonance(float(v)))
        s2.set(self.graph.acid.q); s2.grid(row=0, column=3, sticky="ew", padx=(4,16))

        ttk.Label(box, text="Env Mod (Hz)").grid(row=0, column=4, sticky="e")
        s3 = ttk.Scale(box, from_=0.0, to=3000.0, orient="horizontal",
                       command=lambda v: self.graph.set_acid_envmod(float(v)))
        s3.set(self.graph.acid.env_mod); s3.grid(row=0, column=5, sticky="ew", padx=(4,16))

        ttk.Label(box, text="Slide (ms)").grid(row=0, column=6, sticky="e")
        s4 = ttk.Scale(box, from_=0.0, to=120.0, orient="horizontal",
                       command=lambda v: self.graph.set_acid_slide_ms(float(v)))
        s4.set(self.graph.acid.slide_ms); s4.grid(row=0, column=7, sticky="ew", padx=(4,16))

        ttk.Label(box, text="Accent Amt").grid(row=1, column=0, sticky="e", padx=(10,6))
        s5 = ttk.Scale(box, from_=1.0, to=2.5, orient="horizontal",
                       command=lambda v: self.graph.set_acid_accent_gain(float(v)))
        s5.set(self.graph.acid.accent_gain); s5.grid(row=1, column=1, sticky="ew", padx=(4,16))

        ttk.Label(box, text="Pattern").grid(row=1, column=2, sticky="e")
        self.var_pat = tk.StringVar(value="Root")
        cmb = ttk.Combobox(box, textvariable=self.var_pat, values=["Root","UpDown","Random"], state="readonly", width=10)
        cmb.grid(row=1, column=3, sticky="w"); cmb.bind("<<ComboboxSelected>>", lambda e: self.graph.set_acid_pattern(self.var_pat.get()))

        ttk.Label(box, text="Octave").grid(row=1, column=4, sticky="e")
        self.var_oct = tk.IntVar(value=0)
        s6 = ttk.Scale(box, from_=-1, to=1, orient="horizontal",
                       command=lambda v: self.graph.set_acid_octave(int(float(v))))
        s6.set(self.var_oct.get()); s6.grid(row=1, column=5, sticky="ew", padx=(4,16))

        for c in range(0, 8): box.columnconfigure(c, weight=1)

    # ---------------- Delay send ----------------
    def _build_delay_controls(self):
        box = ttk.LabelFrame(self, text="Acid Delay Send")
        box.grid(row=4, column=0, sticky="ew", pady=(8,0))

        ttk.Label(box, text="Send").grid(row=0, column=0, sticky="e", padx=(10,6))
        s1 = ttk.Scale(box, from_=0.0, to=1.0, orient="horizontal",
                       command=lambda v: self.graph.set_delay_send(float(v)))
        s1.set(self.graph.acid_send); s1.grid(row=0, column=1, sticky="ew", padx=(4,16))

        ttk.Label(box, text="Time (ms)").grid(row=0, column=2, sticky="e")
        s2 = ttk.Scale(box, from_=90.0, to=480.0, orient="horizontal",
                       command=lambda v: self.graph.set_delay_time(float(v)))
        s2.set(self.graph.delay.time_ms); s2.grid(row=0, column=3, sticky="ew", padx=(4,16))

        ttk.Label(box, text="Feedback").grid(row=0, column=4, sticky="e")
        s3 = ttk.Scale(box, from_=0.0, to=0.65, orient="horizontal",
                       command=lambda v: self.graph.set_delay_feedback(float(v)))
        s3.set(self.graph.delay.feedback); s3.grid(row=0, column=5, sticky="ew", padx=(4,16))

        for c in range(0, 6): box.columnconfigure(c, weight=1)

    # ---------------- RGB Sound Designer (labels = real params) ----------------
    def _build_rgb_designer(self):
        box = ttk.LabelFrame(self, text="Sound Designer – Bass & Kick (RGB mapped to real params)")
        box.grid(row=5, column=0, sticky="ew", pady=(8,0))

        # Bass
        ttk.Label(box, text="Bass Sound").grid(row=0, column=0, sticky="w", padx=(10,6))
        labels_bass = ("Harmonics / Drive", "Brightness (LP Cutoff)", "Motion (LFO)")
        self.var_BR = tk.IntVar(value=128); self.var_BG = tk.IntVar(value=160); self.var_BB = tk.IntVar(value=64)
        def _apply_bass():
            self.graph.apply_bass_rgb(self.var_BR.get(), self.var_BG.get(), self.var_BB.get())
            hexc = rgb_to_hex(self.var_BR.get(), self.var_BG.get(), self.var_BB.get())
            self.lbl_bhex.configure(text=hexc, background=hexc)
        self._triple_sliders(box, row=1, labels=labels_bass,
                             vars=(self.var_BR, self.var_BG, self.var_BB), on_apply=_apply_bass)
        self.lbl_bhex = ttk.Label(box, text="#808040", width=10, anchor="center"); self.lbl_bhex.grid(row=1, column=8, sticky="w")

        # Kick
        ttk.Label(box, text="Kick Sound").grid(row=2, column=0, sticky="w", padx=(10,6))
        labels_kick = ("Punch (Click / Drive)", "Brightness / Decay", "Boom (Drop / Length)")
        self.var_KR = tk.IntVar(value=200); self.var_KG = tk.IntVar(value=120); self.var_KB = tk.IntVar(value=100)
        def _apply_kick():
            self.graph.apply_kick_rgb(self.var_KR.get(), self.var_KG.get(), self.var_KB.get())
            hexc = rgb_to_hex(self.var_KR.get(), self.var_KG.get(), self.var_KB.get())
            self.lbl_khex.configure(text=hexc, background=hexc)
        self._triple_sliders(box, row=3, labels=labels_kick,
                             vars=(self.var_KR, self.var_KG, self.var_KB), on_apply=_apply_kick)
        self.lbl_khex = ttk.Label(box, text="#C87864", width=10, anchor="center"); self.lbl_khex.grid(row=3, column=8, sticky="w")

        for c in range(0, 9): box.columnconfigure(c, weight=1)

    def _triple_sliders(self, parent, row: int, labels: tuple[str, str, str], vars, on_apply):
        ttk.Label(parent, text=labels[0]).grid(row=row, column=1, sticky="e")
        s1 = ttk.Scale(parent, from_=0, to=255, orient="horizontal",
                       command=lambda v, var=vars[0]: var.set(int(float(v))))
        s1.set(vars[0].get()); s1.grid(row=row, column=2, sticky="ew", padx=(4,12))

        ttk.Label(parent, text=labels[1]).grid(row=row, column=3, sticky="e")
        s2 = ttk.Scale(parent, from_=0, to=255, orient="horizontal",
                       command=lambda v, var=vars[1]: var.set(int(float(v))))
        s2.set(vars[1].get()); s2.grid(row=row, column=4, sticky="ew", padx=(4,12))

        ttk.Label(parent, text=labels[2]).grid(row=row, column=5, sticky="e")
        s3 = ttk.Scale(parent, from_=0, to=255, orient="horizontal",
                       command=lambda v, var=vars[2]: var.set(int(float(v))))
        s3.set(vars[2].get()); s3.grid(row=row, column=6, sticky="ew", padx=(4,12))

        ttk.Button(parent, text="Apply", command=on_apply).grid(row=row, column=7, sticky="e")

    # ---------------- Sound Painter Panel ----------------
    def _build_painter_panel(self):
        box = ttk.LabelFrame(self, text="Sound Painter – multilayer (X=Amplitude, Y=Frequency)")
        box.grid(row=1, column=1, rowspan=6, sticky="n", padx=(12,0), pady=(0,0))

        # ---------- Left: layers list + buttons ----------
        left = ttk.Frame(box); left.grid(row=0, column=0, sticky="ns", padx=(10,6), pady=8)
        ttk.Label(left, text="Layers").grid(row=0, column=0, columnspan=2, sticky="w")
        self.lst_layers = tk.Listbox(left, height=6, exportselection=False)
        self.lst_layers.grid(row=1, column=0, columnspan=2, sticky="nsew")
        left.rowconfigure(1, weight=1)
        for i, L in enumerate(self.graph.layers):
            self.lst_layers.insert(tk.END, f"{L.name}")
        self.lst_layers.selection_set(0)
        self.current_layer = 0
        self.lst_layers.bind("<<ListboxSelect>>", lambda e: self._layer_select())

        ttk.Button(left, text="+ Add", command=self._layer_add).grid(row=2, column=0, sticky="ew", pady=(6,2))
        ttk.Button(left, text="Duplicate", command=self._layer_dup).grid(row=2, column=1, sticky="ew", pady=(6,2))
        ttk.Button(left, text="Delete", command=self._layer_del).grid(row=3, column=0, sticky="ew")
        ttk.Button(left, text="Clear draw", command=self._layer_clear).grid(row=3, column=1, sticky="ew")

        # ---------- Right: canvas + params ----------
        right = ttk.Frame(box); right.grid(row=0, column=1, sticky="n", padx=(6,10), pady=8)

        # canvas
        self.p_width, self.p_height = 520, 260
        self.cnv = tk.Canvas(right, width=self.p_width, height=self.p_height,
                            bg="#0B0B0B", highlightthickness=1, highlightbackground="#666", cursor="crosshair")
        self.cnv.grid(row=0, column=0, columnspan=6, sticky="w")
        self._p_pts = []; self._p_lines = []
        self.cnv.bind("<Button-1>", self._p_on_down)
        self.cnv.bind("<B1-Motion>", self._p_on_drag)
        self.cnv.bind("<ButtonRelease-1>", self._p_on_up)

        # params (per layer)
        ttk.Label(right, text="Min Freq (Hz)").grid(row=1, column=0, sticky="e", padx=(8,6))
        self.var_fmin = tk.IntVar(value=int(self.graph.layers[0].fmin))
        self.sb_min = ttk.Spinbox(right, from_=20, to=2000, textvariable=self.var_fmin, width=6, command=self._layer_params_changed)
        self.sb_min.grid(row=1, column=1, sticky="w")

        ttk.Label(right, text="Max Freq (Hz)").grid(row=1, column=2, sticky="e", padx=(8,6))
        self.var_fmax = tk.IntVar(value=int(self.graph.layers[0].fmax))
        self.sb_max = ttk.Spinbox(right, from_=200, to=12000, textvariable=self.var_fmax, width=6, command=self._layer_params_changed)
        self.sb_max.grid(row=1, column=3, sticky="w")

        ttk.Label(right, text="Loop (beats)").grid(row=1, column=4, sticky="e", padx=(8,6))
        self.var_beats = tk.IntVar(value=int(self.graph.layers[0].beats))
        self.sb_beats = ttk.Spinbox(right, from_=1, to=16, textvariable=self.var_beats, width=4, command=self._layer_params_changed)
        self.sb_beats.grid(row=1, column=5, sticky="w")

        ttk.Label(right, text="Mode").grid(row=2, column=0, sticky="e", padx=(8,6))
        self.var_mode = tk.StringVar(value=self.graph.layers[0].mode)
        ttk.Combobox(right, textvariable=self.var_mode, state="readonly", width=6, values=["sine","saw"])\
            .grid(row=2, column=1, sticky="w")

        ttk.Label(right, text="Gain").grid(row=2, column=2, sticky="e", padx=(8,6))
        self.var_pgain = tk.DoubleVar(value=self.graph.layers[0].gain)
        ttk.Scale(right, from_=0.0, to=1.6, orient="horizontal",
                command=lambda v: self._layer_gain_changed(float(v)))\
            .grid(row=2, column=3, sticky="ew", padx=(4,10))

        self.var_quant = tk.BooleanVar(value=self.graph.layers[0].quantize)
        ttk.Checkbutton(right, text="Quantize to scale (Key)", variable=self.var_quant).grid(row=2, column=4, sticky="w")

        self.var_enable = tk.BooleanVar(value=self.graph.layers[0].enabled)
        ttk.Checkbutton(right, text="Enable Layer", variable=self.var_enable,
                        command=lambda: self._layer_enable_toggle())\
            .grid(row=2, column=5, sticky="w")

        ttk.Button(right, text="Apply Drawing → Layer", command=self._layer_apply).grid(row=3, column=4, sticky="e", padx=(4,10))
        ttk.Button(right, text="Clear (Layer audio)", command=self._layer_clear_audio).grid(row=3, column=5, sticky="w", padx=(0,10))

        # info line
        self.var_pinfo = tk.StringVar(value="Draw, set params, Apply → Enable.")
        ttk.Label(right, textvariable=self.var_pinfo).grid(row=4, column=0, columnspan=6, sticky="w", pady=(6,0))

        for c in range(0,6): right.columnconfigure(c, weight=1)
        box.columnconfigure(1, weight=1)
        self._p_draw_grid()


    def _p_draw_grid(self):
        """Draw background grid, axes and labels on the painter canvas."""
        c = self.cnv
        c.delete("grid")
        w, h = self.p_width, self.p_height

        # מסגרת
        c.create_rectangle(1, 1, w-2, h-2, outline="#888", width=1, tags="grid")

        # קווי גריד כל 10%
        for i in range(1, 10):
            x = int(w * i/10); y = int(h * i/10)
            c.create_line(x, 2, x, h-2, fill="#222", width=1, tags="grid")
            c.create_line(2, y, w-2, y, fill="#222", width=1, tags="grid")

        # ציר X (אמפליטודה 0..1) — טיקים וטקסט
        for i, lbl in zip([0, 5, 10], ["0", "0.5", "1.0"]):
            x = int(w * i/10)
            c.create_line(x, h-2, x, h-8, fill="#AAA", width=2, tags="grid")
            c.create_text(x+12, h-12, text=lbl, fill="#AAA", font=("Segoe UI", 8), tags="grid")

        # ציר Y (תדר) — תחתית=Min, למעלה=Max
        fmin, fmax = self.var_fmin.get(), self.var_fmax.get()
        c.create_text(8, h-12, text=f"{fmin} Hz", anchor="w", fill="#AAA", font=("Segoe UI", 8), tags="grid")
        c.create_text(8, 12,     text=f"{fmax} Hz", anchor="w", fill="#AAA", font=("Segoe UI", 8), tags="grid")

        # סקאלת זמן (0..beats)
        beats = self.var_beats.get()
        c.create_text(w-10, h-12, text=f"{beats} beats", anchor="e", fill="#AAA", font=("Segoe UI", 8), tags="grid")


    def _p_on_down(self, e):
        self._p_pts = [(e.x, e.y)]
        for lid in self._p_lines: self.cnv.delete(lid)
        self._p_lines = []
        self._p_last = (e.x, e.y)

    def _p_on_drag(self, e):
        x0,y0 = self._p_last
        color = self.graph.layers[self.current_layer].color
        line = self.cnv.create_line(x0,y0, e.x,e.y, fill=color, width=3, capstyle=tk.ROUND, smooth=True, tags=("stroke",))
        self._p_lines.append(line)
        self._p_pts.append((e.x, e.y))
        self._p_last = (e.x, e.y)

    def _p_on_up(self, e):
        pass

    def _p_clear(self):
        # מחיקת השרטוט מהקנבס
        for lid in self._p_lines: self.cnv.delete(lid)
        self._p_lines.clear(); self._p_pts.clear()
        # ציור מחדש של הגריד
        self._p_draw_grid()
        # ניתוק הסאונד הקודם כדי "לשמוע רק את מה שמצויר"
        self.graph.set_painter_curve(np.array([], dtype=np.float32), np.array([], dtype=np.float32), 0)
        # כיבוי Enable
        if hasattr(self, "var_enable"):
            self.var_enable.set(False)
            self.graph.set_painter_enabled(False)
        # מידע
        self.var_pinfo.set("Cleared. Draw a new path and Apply.")
        self.var_status.set("Painter cleared.")


    def _p_apply(self):
        if not self._p_pts:
            self.var_status.set("Painter: draw a path first."); return
        pts = np.array(self._p_pts, dtype=np.float32)

        # arc-length->time
        diffs = np.diff(pts, axis=0)
        seglen = np.sqrt((diffs**2).sum(axis=1))
        arc = np.concatenate([[0.0], np.cumsum(seglen)])
        total = float(arc[-1]) if arc[-1] > 0 else 1.0
        arc /= total

        beats = int(self.var_beats.get())
        bpm = int(self.var_bpm.get())
        loop_samples = max(1, int((60.0 / bpm) * beats * SR))

        t = np.linspace(0.0, 1.0, loop_samples, dtype=np.float32)
        x = np.interp(t, arc, pts[:,0])
        y = np.interp(t, arc, pts[:,1])

        amp = np.clip(x / float(self.p_width), 0.0, 1.0).astype(np.float32)
        fmin = float(self.var_fmin.get()); fmax = float(self.var_fmax.get())
        y_norm = 1.0 - np.clip(y / float(self.p_height), 0.0, 1.0)
        freq = (fmin * (fmax / fmin) ** y_norm).astype(np.float32)

        if bool(self.var_quant.get()):
            root = NOTE_FREQS.get(self.var_key.get().upper(), 46.25)
            notes = []
            for midi in range(-48, 72):
                hz = root * (2.0 ** (midi / 12.0))
                if fmin <= hz <= fmax: notes.append(hz)
            notes = np.array(notes, dtype=np.float32)
            if len(notes):
                idx = np.abs(notes.reshape(1,-1) - freq.reshape(-1,1)).argmin(axis=1)
                freq = notes[idx]

        # התקנה למנוע – זה מחליף כל עקומה קודמת
        self.graph.set_painter_mode(self.var_mode.get())
        self.graph.set_painter_gain(float(self.var_pgain.get()))
        self.graph.set_painter_curve(freq, amp, loop_samples)

        # הפעלה אוטומטית אם מסומן Enable
        if self.var_enable.get():
            self.graph.set_painter_enabled(True)

        dur_sec = (60.0 / bpm) * beats
        self.var_pinfo.set(
            f"Applied: {beats} beats (~{dur_sec:.2f}s) | {loop_samples} samples | "
            f"Freq {int(fmin)}–{int(fmax)} Hz | Amp 0–1"
        )
        self.var_status.set("Painter applied.")
    # ---------------- Add layers ---------------------

    def _layer_select(self):
        sel = self.lst_layers.curselection()
        if not sel: return
        idx = int(sel[0])
        self.current_layer = idx
        L = self.graph.layers[idx]
        self.var_fmin.set(int(L.fmin)); self.var_fmax.set(int(L.fmax)); self.var_beats.set(int(L.beats))
        self.var_mode.set(L.mode); self.var_pgain.set(L.gain); self.var_quant.set(L.quantize); self.var_enable.set(L.enabled)
        self._p_draw_grid()
        self.var_pinfo.set(f"Selected: {L.name} ({L.color})")

    def _layer_add(self):
        if len(self.graph.layers) >= 8:
            self.var_status.set("Max 8 layers."); return
        colors = ["#39C2FF","#FF8C39","#A1E751","#FF4D88","#FFD166","#06D6A0","#9B5DE5","#4EA8DE"]
        col = colors[len(self.graph.layers) % len(colors)]
        idx = self.graph.painter_add_layer(f"Layer {len(self.graph.layers)+1}", col)
        self.lst_layers.insert(tk.END, self.graph.layers[idx].name)
        self.lst_layers.selection_clear(0, tk.END); self.lst_layers.selection_set(idx)
        self._layer_select()

    def _layer_dup(self):
        L = self.graph.layers[self.current_layer]
        idx = self.graph.painter_add_layer(f"{L.name} copy", L.color)
        C = self.graph.layers[idx]
        C.mode=L.mode; C.quantize=L.quantize; C.gain=L.gain; C.fmin=L.fmin; C.fmax=L.fmax; C.beats=L.beats
        C.freq = (None if L.freq is None else L.freq.copy()); C.amp = (None if L.amp is None else L.amp.copy()); C.loop_len = L.loop_len
        self.lst_layers.insert(tk.END, C.name)
        self.lst_layers.selection_clear(0, tk.END); self.lst_layers.selection_set(idx)
        self._layer_select()

    def _layer_del(self):
        if len(self.graph.layers) <= 1: 
            self.var_status.set("Keep at least one layer."); return
        idx = self.current_layer
        self.graph.painter_remove_layer(idx)
        self.lst_layers.delete(idx)
        new_idx = max(0, idx-1)
        self.lst_layers.selection_clear(0, tk.END); self.lst_layers.selection_set(new_idx)
        self.current_layer = new_idx
        self._layer_select()

    def _layer_clear(self):
        # clear drawing (canvas only)
        for lid in self._p_lines: self.cnv.delete(lid)
        self._p_lines.clear(); self._p_pts.clear()
        self._p_draw_grid()

    def _layer_clear_audio(self):
        # clear layer audio curve → won’t play
        self.graph.painter_set_curve(self.current_layer, np.array([],dtype=np.float32), np.array([],dtype=np.float32), 0)
        self.var_pinfo.set("Layer audio cleared. Draw+Apply to replace.")

    def _layer_gain_changed(self, v: float):
        self.graph.painter_update_params(self.current_layer, gain=v)

    def _layer_enable_toggle(self):
        self.graph.painter_set_enabled(self.current_layer, self.var_enable.get())

    def _layer_params_changed(self):
        self.graph.painter_update_params(
            self.current_layer,
            fmin=self.var_fmin.get(), fmax=self.var_fmax.get(),
            beats=self.var_beats.get(), mode=self.var_mode.get(), quantize=self.var_quant.get()
        )
        if hasattr(self, "_p_draw_grid"): self._p_draw_grid()

    def _layer_apply(self):
        if not self._p_pts:
            self.var_status.set("Painter: draw first."); return
        pts = np.array(self._p_pts, dtype=np.float32)
        diffs = np.diff(pts, axis=0)
        seglen = np.sqrt((diffs**2).sum(axis=1))
        arc = np.concatenate([[0.0], np.cumsum(seglen)])
        total = float(arc[-1]) if arc[-1] > 0 else 1.0
        arc /= total

        L = self.graph.layers[self.current_layer]
        beats = int(self.var_beats.get()); bpm = int(self.var_bpm.get())
        loop_samples = max(1, int((60.0 / bpm) * beats * SR))

        t = np.linspace(0.0, 1.0, loop_samples, dtype=np.float32)
        x = np.interp(t, arc, pts[:,0]); y = np.interp(t, arc, pts[:,1])

        amp = np.clip(x / float(self.p_width), 0.0, 1.0).astype(np.float32)
        fmin = float(self.var_fmin.get()); fmax = float(self.var_fmax.get())
        y_norm = 1.0 - np.clip(y / float(self.p_height), 0.0, 1.0)
        freq = (fmin * (fmax / fmin) ** y_norm).astype(np.float32)

        if bool(self.var_quant.get()):
            root = NOTE_FREQS.get(self.var_key.get().upper(), 46.25)
            notes=[]
            for midi in range(-48,72):
                hz = root * (2.0 ** (midi/12.0))
                if fmin <= hz <= fmax: notes.append(hz)
            if notes:
                notes = np.array(notes, dtype=np.float32)
                idx = np.abs(notes.reshape(1,-1) - freq.reshape(-1,1)).argmin(axis=1)
                freq = notes[idx]

        self.graph.painter_update_params(self.current_layer,
                                        mode=self.var_mode.get(), gain=float(self.var_pgain.get()),
                                        fmin=fmin, fmax=fmax, beats=beats, quantize=bool(self.var_quant.get()))
        self.graph.painter_set_curve(self.current_layer, freq, amp, loop_samples)

        # enable if checked
        if self.var_enable.get(): self.graph.painter_set_enabled(self.current_layer, True)

        dur = (60.0/bpm)*beats
        self.var_pinfo.set(f"{L.name}: {beats} beats (~{dur:.2f}s) | {loop_samples} samples | {int(fmin)}–{int(fmax)} Hz")
        self.var_status.set("Layer applied.")


    # ---------------- Status + meters ----------------
    def _build_status(self):
        st = ttk.Frame(self); st.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8,0))
        self.var_status = tk.StringVar(value="Stopped.")
        self.var_meters = tk.StringVar(value="")
        ttk.Label(st, textvariable=self.var_status).grid(row=0, column=0, sticky="w")
        ttk.Label(st, textvariable=self.var_meters).grid(row=0, column=1, sticky="e")
        self.after(200, self._poll_meters)

    def _poll_meters(self):
        def db(x):
            import math
            return max(-80.0, 20.0 * math.log10(max(1e-9, float(x))))
        pk = db(self.graph.meter_peak); rms = db(self.graph.meter_rms)
        self.var_meters.set(f"Peak {pk:.1f} dBFS | RMS {rms:.1f} dBFS")

        # position: 16 steps per bar, 4/4 → Beat = step//4 + 1, Sixteenth = step%4 + 1
        step = int(getattr(self.graph, "current_step", 0))
        beat = (step // 4) + 1
        sixteenth = (step % 4) + 1
        bar = int(getattr(self.graph, "bar_count", 1))
        if hasattr(self, "var_pos"):
            self.var_pos.set(f"Bar {bar} • Beat {beat}.{sixteenth}")

        self.after(200, self._poll_meters)

    # ---------------- Simple handlers ----------------
    def _toggle_play(self):
        if self.playing:
            self.engine.stop(); self.playing = False
            self.btn_play.configure(text="Play"); self.var_status.set("Stopped.")
        else:
            self.engine.start(); self.playing = True
            self.btn_play.configure(text="Stop"); self.var_status.set("Playing…")

    def _on_bpm_change(self, bpm: int):
        self.var_bpm.set(int(bpm))
        self.seq.set_bpm(int(bpm))
        # guard: might be called before status widgets are built
        if hasattr(self, "var_status"):
            self.var_status.set(f"BPM: {int(bpm)}")


    def _on_style_change(self): self.graph.set_style(self.var_style.get())
    def _on_key_change(self): self.graph.set_key(self.var_key.get()); self.var_status.set(f"Key: {self.var_key.get()}")

# -----------------------------------------------------------------------------

def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        theme = "vista" if "vista" in style.theme_names() else "clam"
        style.theme_use(theme)
    except Exception:
        pass
    root.rowconfigure(0, weight=1); root.columnconfigure(0, weight=1)
    app = LiveUI(root); app.grid(sticky="nsew")
    root.mainloop()

if __name__ == "__main__":
    main()
