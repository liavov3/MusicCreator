# PsyForge – Kick+Bass Loop Generator (Psy/Goa)

Generate club-ready kick+bass loops using pure Python DSP, with both CLI and a clean Tkinter (ttk) GUI.

## Quickstart (Windows)
```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.cli --style psy --bpm 145 --key F# --bars 8 --out outputs/loop_psy.wav
python -m src.ui_tk
