"""
Sound character presets for Kick and Bass.
Tweak or add your own flavors easily.
"""

KICK_PRESETS = {
    # silky, gentle top, round low
    "Soft / Silky": dict(length_ms=185.0, f_start=78.0, f_end=48.0, click_ms=1.0, body_tau_s=0.07, gain=0.9),
    # the default psy punch
    "Punchy / Classic": dict(length_ms=180.0, f_start=85.0, f_end=48.0, click_ms=2.0, body_tau_s=0.06, gain=0.95),
    # more body, lower sweep
    "Fat / Warm": dict(length_ms=195.0, f_start=82.0, f_end=45.0, click_ms=2.0, body_tau_s=0.08, gain=1.0),
    # sharp transient, shorter body, club banger
    "Hard / Aggressive": dict(length_ms=170.0, f_start=92.0, f_end=50.0, click_ms=2.5, body_tau_s=0.05, gain=1.05),
}

BASS_PRESETS = {
    # smooth, deeper low-pass, gentle drive
    "Smooth": dict(lp_cut=120.0, drive=1.2, level=0.75),
    # classic psy rolling
    "Classic": dict(lp_cut=140.0, drive=1.6, level=0.85),
    # gritty mid presence
    "Gritty": dict(lp_cut=180.0, drive=2.0, level=0.9),
    # bright & screaming (still mono bass lane)
    "Acidic": dict(lp_cut=220.0, drive=2.3, level=0.95),
    # goa offbeat friendly, more body
    "Goa Warm": dict(lp_cut=160.0, drive=1.5, level=0.85),
}
