"""
PsyForge Live – Kick/Bass/Acid + Drums + Delay + RGB + Scenes + Recording
Run:
    python src/ui_live.py
"""

from __future__ import annotations
import os, time
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import numpy as np

# ---- Optional recording ----
try:
    import soundfile as sf  # pip install soundfile
except Exception:
    sf = None

# ---- Engine + instruments you already have ----
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

    # ----- simple one-pole filters (fast & stable) -----
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
        # HP = x - LP(x)
        return (x - self._onepole_lp(x, fc)).astype(np.float32)

    # ----- synthesis -----
    def trigger(self) -> np.ndarray:
        n = max(1, int(SR * self.length_ms / 1000.0))
        # base noise
        noise = np.random.randn(n).astype(np.float32)

        # band-pass (HP then LP)
        x = self._onepole_hp(noise, self.hp_cut)
        x = self._onepole_lp(x, self.lp_cut)

        # multi-burst envelope cluster
        env = np.zeros(n, dtype=np.float32)
        weights = [1.0, 0.82, 0.66, 0.55]  # relative loudness of bursts
        for j, ms in enumerate(self.bursts_ms):
            off = int(SR * ms / 1000.0)
            if off >= n:
                continue
            m = n - off
            t = np.arange(m, dtype=np.float32) / SR
            e = np.exp(-t / self.decay_s, dtype=np.float32) * (weights[j] if j < len(weights) else 0.5)
            env[off:off+m] += e

        # apply envelope
        x *= env

        # short fade in/out to avoid hard edges
        fade = min(64, n // 20)
        if fade > 0:
            w = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            x[:fade] *= w
            x[-fade:] *= w[::-1]

        # tiny dither kills denormals on some CPUs
        x += (1e-8 * np.random.randn(n)).astype(np.float32)

        # level & return
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

# =============================================================================
# Synth graph & audio callback
# =============================================================================

class LiveSynthGraph:
    """
    Manages instruments, patterns, FX, ducking, scenes, meters & recording.
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

        # Mixer gains (kick loud by default)
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

        # Active voices
        self.active: list[dict] = []

        # helpers
        self._rng = np.random.default_rng(42)
        self.acid_octave = 0
        self.acid_pattern = "Root"

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

    # --- helpers ---
    def _acid_semitone_for_step(self, s: int) -> int:
        if self.acid_pattern == "Root": return 0
        if self.acid_pattern == "UpDown":
            seq = [0,3,5,7,10,7,5,3,0,-2,-5,-7,-5,-2,0,3]
            return seq[s % 16]
        return int(self._rng.choice([0,2,3,5,7,9,10,12,-2,-5]))

    # --- audio callback ---
    def render(self, frames: int) -> np.ndarray:
        out = np.zeros((2, frames), dtype=np.float32)
        triggers = self.seq.advance(frames)
        step_len = self.seq.samples_per_step

        kick_in_block = False
        step0_in_block = False

        for track, step in triggers:
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
        if step0_in_block and self.queued_scene and self.queued_scene in self.scenes:
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
    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=12)
        master.title("PsyForge Live – Kick/Bass/Acid + Delay + RGB")
        master.minsize(1240, 820)

        # Sequencer + default patterns
        self.seq = StepSequencer(bpm=145, steps=16)
        # ensure patterns exist
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
        self._build_status()

        # layout
        self.grid(sticky="nsew")
        master.columnconfigure(0, weight=1); master.rowconfigure(0, weight=1)

    # ---------------- Transport (Play, REC, Tap, Style, Key, Scenes) ----------------
    def _build_transport(self):
        fx = ttk.Frame(self); fx.grid(row=0, column=0, sticky="ew", pady=(0,8))
        for c in range(14): fx.columnconfigure(c, weight=1)

        self.btn_play = ttk.Button(fx, text="Play", command=self._toggle_play); self.btn_play.grid(row=0, column=0, padx=(0,8))
        self.btn_rec  = ttk.Button(fx, text="REC",  command=self._toggle_record); self.btn_rec.grid(row=0, column=1, padx=(0,12))

        ttk.Label(fx, text="BPM").grid(row=0, column=2, sticky="e")
        self.var_bpm = tk.IntVar(value=145)
        s = ttk.Scale(fx, from_=120, to=160, orient="horizontal",
                      command=lambda v: self._on_bpm_change(int(float(v))))
        s.set(self.var_bpm.get()); s.grid(row=0, column=3, sticky="ew", padx=(6,12))
        ttk.Button(fx, text="Tap", command=self._tap_tempo).grid(row=0, column=4, padx=(0,12))

        ttk.Label(fx, text="Style").grid(row=0, column=5, sticky="e")
        self.var_style = tk.StringVar(value="psy")
        ttk.Radiobutton(fx, text="Psy", value="psy", variable=self.var_style, command=self._on_style_change).grid(row=0, column=6, sticky="w")
        ttk.Radiobutton(fx, text="Goa", value="goa", variable=self.var_style, command=self._on_style_change).grid(row=0, column=7, sticky="w")

        ttk.Label(fx, text="Key").grid(row=0, column=8, sticky="e")
        self.var_key = tk.StringVar(value="F#")
        cmb = ttk.Combobox(fx, textvariable=self.var_key, values=list(NOTE_FREQS.keys()), width=4, state="readonly")
        cmb.grid(row=0, column=9, sticky="w"); cmb.bind("<<ComboboxSelected>>", lambda e: self._on_key_change())

        # Scenes A..D
        for j, name in enumerate(["A", "B", "C", "D"]):
            ttk.Button(fx, text=f"Queue {name}", command=lambda n=name: self._queue_scene(n)).grid(row=0, column=10+j, padx=(6,2))

    def _tap_tempo(self):
        now = time.time()
        if not hasattr(self, "_tap_times"): self._tap_times = []
        self._tap_times = [t for t in self._tap_times if now - t < 3.0]
        self._tap_times.append(now)
        if len(self._tap_times) >= 2:
            intervals = [self._tap_times[i+1]-self._tap_times[i] for i in range(len(self._tap_times)-1)]
            avg = sum(intervals) / len(intervals)
            bpm = max(80, min(200, int(round(60.0 / max(0.05, avg)))))
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
    def _build_grid(self):
        g = ttk.LabelFrame(self, text="Sequencer (click to toggle steps)")
        g.grid(row=1, column=0, sticky="nsew")
        self.rowconfigure(1, weight=1)

        for c in range(1, 17): g.columnconfigure(c, weight=1)
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
                var = tk.IntVar(value=int(pattern[c]))
                btn = ttk.Checkbutton(g, variable=var, onvalue=1, offvalue=0,
                                      command=lambda t=track, s=c: self._toggle_step(t, s))
                btn.grid(row=r, column=c+1, padx=3, pady=6)
                self.step_vars[track].append(var); self.step_btns[track].append(btn)
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
        pat = self.seq.patterns.get(track); vars_list = self.step_vars.get(track)
        if pat is None or vars_list is None: return
        for i in range(len(pat)): pat[i] = 1; vars_list[i].set(1)

    def _clear_row(self, track: str):
        pat = self.seq.patterns.get(track); vars_list = self.step_vars.get(track)
        if pat is None or vars_list is None: return
        for i in range(len(pat)): pat[i] = 0; vars_list[i].set(0)

    def _invert_row(self, track: str):
        pat = self.seq.patterns.get(track); vars_list = self.step_vars.get(track)
        if pat is None or vars_list is None: return
        for i in range(len(pat)): pat[i] = 0 if pat[i] else 1; vars_list[i].set(int(pat[i]))

    def _pattern_every_n(self, track: str, n: int, start: int = 0):
        pat = self.seq.patterns.get(track); vars_list = self.step_vars.get(track)
        if pat is None or vars_list is None: return
        for i in range(len(pat)):
            val = 1 if ((i - start) % n == 0) else 0
            pat[i] = val; vars_list[i].set(val)

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

    # ---------------- Status + meters ----------------
    def _build_status(self):
        st = ttk.Frame(self); st.grid(row=6, column=0, sticky="ew", pady=(8,0))
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
        self.var_bpm.set(bpm); self.seq.set_bpm(bpm); self.var_status.set(f"BPM: {bpm}")

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
