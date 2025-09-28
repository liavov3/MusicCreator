"""
Tkinter (ttk) GUI for PsyForge.
Keeps a clean, modern look with stock ttk widgets only (no external GUI libs).
"""

from __future__ import annotations
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import soundfile as sf

from psyforge.render import render_loop
from psyforge.presets import NOTE_FREQS
from psyforge.constants import SR


APP_TITLE = "PsyForge – Tk UI"
DEFAULT_OUTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))


class PsyForgeApp(ttk.Frame):
    """Main application frame."""

    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master, padding=16)

        # Window setup
        master.title(APP_TITLE)
        master.minsize(900, 560)

        # ttk theme
        style = ttk.Style()
        theme = "vista" if "vista" in style.theme_names() else "clam"
        style.theme_use(theme)

        # State variables (ttk-friendly)
        self.var_style = tk.StringVar(value="psy")
        self.var_bpm = tk.IntVar(value=145)
        self.var_key = tk.StringVar(value="F#")
        self.var_bars = tk.IntVar(value=8)
        self.var_outpath = tk.StringVar(
            value=os.path.join(DEFAULT_OUTDIR, "loop_psy.wav")
        )

        self._is_rendering = False
        self._last_audio = None  # type: ignore
        self._sr = SR

        # Layout
        self._build_header()
        self._build_notebook()
        self._build_status()

        # Expand
        self.grid(sticky="nsew")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)

    # ---------- UI construction ----------

    def _build_header(self) -> None:
        frame = ttk.Frame(self)
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        frame.columnconfigure(0, weight=1)

        title = ttk.Label(frame, text="Psy/Goa Loop Generator", font=("Segoe UI", 16, "bold"))
        subtitle = ttk.Label(frame, text="Choose style, BPM, key, bars – then render a club-ready loop.",
                             foreground="#666")
        title.grid(row=0, column=0, sticky="w")
        subtitle.grid(row=1, column=0, sticky="w")

    def _build_notebook(self) -> None:
        nb = ttk.Notebook(self)
        nb.grid(row=1, column=0, sticky="nsew")
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        tab_gen = ttk.Frame(nb, padding=12)
        nb.add(tab_gen, text="Generator")
        self._build_generator_tab(tab_gen)

        tab_about = ttk.Frame(nb, padding=16)
        nb.add(tab_about, text="About")
        ttk.Label(tab_about, text="PsyForge – Kick+Bass Generator", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(tab_about, text="Built with pure Tkinter/ttk. No external GUI libs.").pack(anchor="w", pady=(6, 0))

    def _build_generator_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)

        # Style
        ttk.Label(parent, text="Style").grid(row=0, column=0, sticky="w")
        style_box = ttk.Frame(parent)
        style_box.grid(row=0, column=1, sticky="w", pady=4)
        ttk.Radiobutton(style_box, text="Psy", value="psy", variable=self.var_style, command=self._on_style_change).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(style_box, text="Goa", value="goa", variable=self.var_style, command=self._on_style_change).pack(side="left")

        # BPM
        ttk.Label(parent, text="BPM").grid(row=1, column=0, sticky="w")
        bpm_row = ttk.Frame(parent)
        bpm_row.grid(row=1, column=1, sticky="ew", pady=4)
        bpm_row.columnconfigure(1, weight=1)
        ttk.Label(bpm_row, textvariable=self.var_bpm, width=4, anchor="e").grid(row=0, column=0, padx=(0, 8))
        sld = ttk.Scale(bpm_row, from_=130, to=150, orient="horizontal",
                        command=lambda v: self.var_bpm.set(int(float(v))))
        sld.set(self.var_bpm.get())
        sld.grid(row=0, column=1, sticky="ew")

        # Key
        ttk.Label(parent, text="Key (Root)").grid(row=2, column=0, sticky="w")
        keys = list(NOTE_FREQS.keys())
        cmb = ttk.Combobox(parent, textvariable=self.var_key, values=keys, state="readonly")
        cmb.grid(row=2, column=1, sticky="w", pady=4)

        # Bars
        ttk.Label(parent, text="Bars").grid(row=3, column=0, sticky="w")
        spn = ttk.Spinbox(parent, from_=1, to=64, textvariable=self.var_bars, width=6)
        spn.grid(row=3, column=1, sticky="w", pady=4)

        # Output
        ttk.Label(parent, text="Output WAV").grid(row=4, column=0, sticky="w")
        out_row = ttk.Frame(parent)
        out_row.grid(row=4, column=1, sticky="ew", pady=4)
        out_row.columnconfigure(0, weight=1)
        ent = ttk.Entry(out_row, textvariable=self.var_outpath)
        ent.grid(row=0, column=0, sticky="ew")
        ttk.Button(out_row, text="Browse…", command=self._browse_out).grid(row=0, column=1, padx=6)

        # Buttons
        btn_row = ttk.Frame(parent)
        btn_row.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        btn_row.columnconfigure(0, weight=1)
        self.btn_render = ttk.Button(btn_row, text="Generate Loop", command=self._render_async)
        self.btn_render.grid(row=0, column=0, sticky="w")
        ttk.Button(btn_row, text="Open Output Folder", command=self._open_outdir).grid(row=0, column=1, sticky="e")

        ttk.Separator(parent, orient="horizontal").grid(row=6, column=0, columnspan=2, sticky="ew", pady=12)

        # Waveform
        box = ttk.LabelFrame(parent, text="Waveform Preview")
        box.grid(row=7, column=0, columnspan=2, sticky="nsew")
        parent.rowconfigure(7, weight=1)
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(box, background="#0f0f12", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        ttk.Label(box, text="Waveform shows after rendering. Resize the window to zoom.",
                  foreground="#888").grid(row=1, column=0, sticky="w", padx=6, pady=4)

    def _build_status(self) -> None:
        status = ttk.Frame(self)
        status.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        status.columnconfigure(0, weight=1)
        self.var_status = tk.StringVar(value="Ready.")
        ttk.Label(status, textvariable=self.var_status).grid(row=0, column=0, sticky="w")
        self.prog = ttk.Progressbar(status, mode="determinate", length=220)
        self.prog.grid(row=0, column=1, sticky="e")

    # ---------- Actions ----------

    def _browse_out(self) -> None:
        os.makedirs(DEFAULT_OUTDIR, exist_ok=True)
        f = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV file", "*.wav")],
            initialdir=DEFAULT_OUTDIR,
            initialfile=os.path.basename(self.var_outpath.get() or "loop.wav"),
        )
        if f:
            self.var_outpath.set(f)

    def _open_outdir(self) -> None:
        path = self.var_outpath.get()
        folder = os.path.dirname(path) if path else DEFAULT_OUTDIR
        os.makedirs(folder, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(folder)  # type: ignore[attr-defined]
            else:
                import subprocess, sys
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", folder])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open folder:\n{e}")

    def _on_style_change(self) -> None:
        # Keep output filename aligned with the style
        base = "loop_goa.wav" if self.var_style.get() == "goa" else "loop_psy.wav"
        out = self.var_outpath.get()
        folder = os.path.dirname(out) if out else DEFAULT_OUTDIR
        self.var_outpath.set(os.path.join(folder, base))

    def _render_async(self) -> None:
        if self._is_rendering:
            return
        self._is_rendering = True
        self.btn_render.state(["disabled"])
        self.var_status.set("Rendering…")
        self.prog.configure(value=0, maximum=100)

        t = threading.Thread(target=self._do_render, daemon=True)
        t.start()
        self._pulse()

    def _pulse(self) -> None:
        if not self._is_rendering:
            return
        self.prog.configure(value=(self.prog["value"] + 2) % 100)
        self.after(60, self._pulse)

    def _do_render(self) -> None:
        try:
            style = self.var_style.get()
            bpm = int(self.var_bpm.get())
            key = self.var_key.get()
            bars = int(self.var_bars.get())
            outpath = self.var_outpath.get()

            os.makedirs(os.path.dirname(outpath), exist_ok=True)

            # Render stereo (2, N)
            audio = render_loop(style=style, bpm=bpm, key=key, bars=bars, stereo=True)
            sf.write(outpath, audio.T, SR, subtype="PCM_16")

            self._last_audio = audio
            self._draw_waveform(audio)

            self.var_status.set(f"Done: {os.path.basename(outpath)}")
        except Exception as e:
            self.var_status.set("Error")
            messagebox.showerror("Render Error", str(e))
        finally:
            self._is_rendering = False
            self.btn_render.state(["!disabled"])
            self.prog.configure(value=100)

    # ---------- Waveform ----------

    def _draw_waveform(self, audio) -> None:
        """Draw a simple min/max column waveform of the rendered (stereo) buffer."""
        if audio is None or audio.shape[1] == 0:
            return
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            self.after(50, lambda: self._draw_waveform(audio))
            return

        import numpy as np
        mono = audio.mean(axis=0)
        n = mono.shape[0]
        step = max(1, n // w)
        mid = h // 2
        scale = 0.9 * (h / 2.0)

        for x in range(w):
            start = x * step
            end = min(start + step, n)
            seg = mono[start:end]
            if seg.size == 0:
                break
            y1 = int(mid - seg.max() * scale)
            y2 = int(mid - seg.min() * scale)
            self.canvas.create_line(x, y1, x, y2, fill="#36d1ff")

        self.canvas.create_rectangle(1, 1, w - 2, h - 2, outline="#222")


def main() -> None:
    root = tk.Tk()
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    app = PsyForgeApp(root)
    app.grid(sticky="nsew")
    root.mainloop()


if __name__ == "__main__":
    main()
