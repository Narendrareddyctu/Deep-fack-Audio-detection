# config.py - all the settings in one place so they are easy to change

SAMPLE_RATE = 16000      # all audio is converted to 16 kHz mono
DURATION = 4             # seconds of audio used per file (padded / cut)
N_MFCC = 20              # number of MFCC coefficients
N_MELS = 40              # number of mel bands
N_FFT = 512
HOP_LENGTH = 160         # 10 ms hop

DATA_DIR = "data"        # data/real/*.wav  and  data/fake/*.wav
MODEL_PATH = "models/model.joblib"
SPLIT_PATH = "models/split.json"
OUTPUT_DIR = "outputs"

CLASS_NAMES = ["genuine", "synthetic"]   # label 0 = genuine, 1 = synthetic
RANDOM_STATE = 42
