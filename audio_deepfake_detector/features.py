# features.py - load audio and extract MFCC + mel-spectrogram features
import numpy as np
import librosa
import config


def load_audio(path):
    """Read audio file, convert to mono 16 kHz, fix the length."""
    y, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
    return fix_length(y)


def fix_length(y):
    n = config.SAMPLE_RATE * config.DURATION
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))   # pad with zeros if too short
    return y[:n]


def mel_spectrogram(y):
    """Log mel-spectrogram (used for the picture in the app and for features)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=config.SAMPLE_RATE, n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH, n_mels=config.N_MELS)
    return librosa.power_to_db(mel, ref=np.max)


def extract_features(y):
    """Turn a waveform into one feature vector.

    MFCC (+ delta) mean/std, log-mel mean/std, and a few spectral features.
    """
    y = y / (np.max(np.abs(y)) + 1e-9)          # normalise volume

    mfcc = librosa.feature.mfcc(
        y=y, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCC,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, n_mels=config.N_MELS)
    delta = librosa.feature.delta(mfcc)
    mel = mel_spectrogram(y)

    flat = librosa.feature.spectral_flatness(y=y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    cent = librosa.feature.spectral_centroid(y=y, sr=config.SAMPLE_RATE, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    roll = librosa.feature.spectral_rolloff(y=y, sr=config.SAMPLE_RATE, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=config.HOP_LENGTH)

    parts = [
        mfcc.mean(axis=1), mfcc.std(axis=1),
        delta.mean(axis=1), delta.std(axis=1),
        mel.mean(axis=1), mel.std(axis=1),
        [flat.mean(), flat.std(), cent.mean(), cent.std(),
         roll.mean(), roll.std(), zcr.mean(), zcr.std()],
    ]
    return np.concatenate([np.ravel(p) for p in parts]).astype(np.float32)


def features_from_file(path):
    return extract_features(load_audio(path))
