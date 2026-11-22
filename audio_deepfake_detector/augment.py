# augment.py - simulate different recording conditions and compression levels
import os
import subprocess
import tempfile
import numpy as np
import scipy.signal as sig
import soundfile as sf
import librosa
import config

SR = config.SAMPLE_RATE


def add_noise(y, snr_db):
    """Add white noise at a given signal-to-noise ratio."""
    p_signal = np.mean(y ** 2) + 1e-12
    p_noise = p_signal / (10 ** (snr_db / 10))
    return y + np.random.randn(len(y)) * np.sqrt(p_noise)


def telephone(y):
    """Band-limit like a phone call (300-3400 Hz)."""
    b, a = sig.butter(4, [300 / (SR / 2), 3400 / (SR / 2)], btype="band")
    return sig.lfilter(b, a, y)


def low_sample_rate(y, new_sr=8000):
    """Down-sample then up-sample (loses high frequencies)."""
    low = librosa.resample(y, orig_sr=SR, target_sr=new_sr)
    return librosa.resample(low, orig_sr=new_sr, target_sr=SR)[:len(y)]


def reverb(y, rt=0.3):
    """Fake room echo using an exponentially decaying noise impulse response."""
    n = int(SR * rt)
    ir = np.random.randn(n) * np.exp(-np.linspace(0, 8, n))
    ir[0] = 1.0
    out = sig.fftconvolve(y, ir)[:len(y)]
    return out / (np.max(np.abs(out)) + 1e-9) * np.max(np.abs(y))


def codec(y, fmt="mp3", bitrate="32k"):
    """Real lossy compression using ffmpeg (encode then decode again)."""
    with tempfile.TemporaryDirectory() as d:
        wav_in = os.path.join(d, "in.wav")
        comp = os.path.join(d, "c." + fmt)
        wav_out = os.path.join(d, "out.wav")
        sf.write(wav_in, y, SR)
        cmd1 = ["ffmpeg", "-y", "-loglevel", "error", "-i", wav_in, "-b:a", bitrate, comp]
        cmd2 = ["ffmpeg", "-y", "-loglevel", "error", "-i", comp, "-ar", str(SR), "-ac", "1", wav_out]
        subprocess.run(cmd1, check=True)
        subprocess.run(cmd2, check=True)
        out, _ = sf.read(wav_out)
    out = out.astype(np.float32)
    if len(out) < len(y):
        out = np.pad(out, (0, len(y) - len(out)))
    return out[:len(y)]


# name -> function. "clean" means no change.
CONDITIONS = {
    "clean": lambda y: y,
    "noise_20dB": lambda y: add_noise(y, 20),
    "noise_10dB": lambda y: add_noise(y, 10),
    "noise_5dB": lambda y: add_noise(y, 5),
    "reverb": reverb,
    "telephone_band": telephone,
    "resample_8k": low_sample_rate,
    "mp3_128k": lambda y: codec(y, "mp3", "128k"),
    "mp3_64k": lambda y: codec(y, "mp3", "64k"),
    "mp3_32k": lambda y: codec(y, "mp3", "32k"),
}

# cheap ones used when training with augmentation (no ffmpeg needed)
TRAIN_AUG = ["noise_20dB", "noise_10dB", "reverb", "telephone_band", "resample_8k"]


def has_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False
