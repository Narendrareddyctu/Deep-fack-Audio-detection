# generate_demo_data.py
# Makes a SMALL TOY dataset so the project can run without downloading anything.
#
# NOTE: this is NOT real speech. "genuine" clips imitate natural voice
# irregularities (pitch wobble, jitter, breath noise, room noise) and "synthetic"
# clips imitate vocoder-like behaviour (too smooth pitch, very clean, band-limited,
# small periodic artefact). For the real project replace the files in data/real
# and data/fake with a real dataset (ASVspoof 2019 LA, WaveFake, FoR, ...).
import os
import sys
import numpy as np
import scipy.signal as sig
import soundfile as sf
import config

SR = config.SAMPLE_RATE
DUR = config.DURATION
rng = np.random.default_rng(1)


def make_voice(fake):
    n = SR * DUR
    t = np.arange(n) / SR
    base = rng.uniform(90, 260)

    # pitch contour
    if fake:
        # smooth contour, tiny variation
        contour = base * (1 + 0.04 * np.sin(2 * np.pi * rng.uniform(0.3, 0.8) * t))
        jitter = 0.0005
    else:
        walk = np.cumsum(rng.normal(0, 1, n))
        walk = sig.lfilter(*sig.butter(2, 3 / (SR / 2)), walk)
        walk = walk / (np.std(walk) + 1e-9)
        contour = base * (1 + 0.10 * walk + 0.02 * np.sin(2 * np.pi * 5.5 * t))
        jitter = rng.uniform(0.003, 0.008)
    contour = contour * (1 + rng.normal(0, jitter, n))
    contour = np.clip(contour, 60, 400)

    phase = 2 * np.pi * np.cumsum(contour) / SR
    voice = np.zeros(n)
    for k in range(1, 40):
        if (contour.max() * k) > SR / 2 - 200:
            break
        voice += np.sin(k * phase) / k

    # syllable-like amplitude envelope
    rate = rng.uniform(2.5, 4.5)
    env = 0.55 + 0.45 * np.sin(2 * np.pi * rate * t + rng.uniform(0, 6)) ** 2
    if not fake:
        env *= 1 + 0.15 * rng.normal(0, 1, n).cumsum() / np.sqrt(n)  # shimmer-ish
    voice *= env

    # formant filters (vowel colour)
    f_sets = [(rng.uniform(300, 800), rng.uniform(900, 2200), rng.uniform(2300, 3200))]
    out = np.zeros(n)
    for f in f_sets[0]:
        b, a = sig.iirpeak(f / (SR / 2), Q=rng.uniform(4, 9))
        out += sig.lfilter(b, a, voice)
    out = out / (np.max(np.abs(out)) + 1e-9)

    if fake:
        # vocoder-like: cut high frequencies + faint periodic artefact
        cutoff = rng.uniform(5500, 7500)
        b, a = sig.butter(6, cutoff / (SR / 2))
        out = sig.lfilter(b, a, out)
        art_f = rng.uniform(1500, 3500)
        out += rng.uniform(0.005, 0.02) * np.sin(2 * np.pi * art_f * t)
        out += rng.normal(0, 0.002, n)
    else:
        # breath + room noise
        breath = sig.lfilter(*sig.butter(2, [1000 / (SR / 2), 5000 / (SR / 2)], btype="band"),
                             rng.normal(0, 1, n))
        out += rng.uniform(0.01, 0.05) * breath * (env > 0.7)
        out += rng.normal(0, rng.uniform(0.003, 0.015), n)
    return (out / (np.max(np.abs(out)) + 1e-9) * 0.8).astype(np.float32)


def main(n_per_class=150):
    for label, fake in [("real", False), ("fake", True)]:
        folder = os.path.join(config.DATA_DIR, label)
        os.makedirs(folder, exist_ok=True)
        for i in range(n_per_class):
            sf.write(os.path.join(folder, f"{label}_{i:04d}.wav"), make_voice(fake), SR)
        print(f"created {n_per_class} files in {folder}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
