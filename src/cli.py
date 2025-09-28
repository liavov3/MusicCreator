"""
Minimal CLI entrypoint for generating loops without opening the GUI.
"""

from __future__ import annotations
import argparse
import os
import soundfile as sf

from psyforge.render import render_loop
from psyforge.constants import SR


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a Psy/Goa kick+bass loop.")
    ap.add_argument("--style", choices=["psy", "goa"], default="psy", help="Loop style")
    ap.add_argument("--bpm", type=int, default=145, help="Beats per minute")
    ap.add_argument("--key", type=str, default="F#", help="Root key (C, C#, D, ...)")
    ap.add_argument("--bars", type=int, default=8, help="Number of 4/4 bars")
    ap.add_argument("--out", type=str, default="outputs/loop_psy.wav", help="Output WAV path")
    args = ap.parse_args()

    audio = render_loop(style=args.style, bpm=args.bpm, key=args.key, bars=args.bars, stereo=True)
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    sf.write(args.out, audio.T, SR, subtype="PCM_16")
    print(f"[OK] Wrote: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
