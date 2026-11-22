# evaluate.py - robustness test: apply noise / reverb / phone / mp3 compression
# to the TEST files and see how accuracy changes.
import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

import config
import features
import augment


def main():
    saved = joblib.load(config.MODEL_PATH)
    model = saved["model"]
    split = json.load(open(config.SPLIT_PATH))
    files, labels = split["test_files"], np.array(split["test_labels"])

    waves = [features.load_audio(p) for p in files]
    ffmpeg_ok = augment.has_ffmpeg()
    if not ffmpeg_ok:
        print("ffmpeg not found -> mp3 conditions skipped")

    np.random.seed(config.RANDOM_STATE)
    rows = []
    for cond, fn in augment.CONDITIONS.items():
        if cond.startswith("mp3") and not ffmpeg_ok:
            continue
        X = np.array([features.extract_features(fn(w)) for w in waves])
        pred = model.predict(X)
        acc = accuracy_score(labels, pred)
        f1 = f1_score(labels, pred)
        rows.append({"condition": cond, "accuracy": round(acc, 4), "f1_score": round(f1, 4)})
        print(f"{cond:16s} acc={acc:.3f}  f1={f1:.3f}")

    df = pd.DataFrame(rows)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(config.OUTPUT_DIR, "robustness_results.csv"), index=False)

    plt.figure(figsize=(9, 4))
    plt.bar(df["condition"], df["accuracy"], color="steelblue")
    plt.axhline(0.5, color="red", linestyle="--", label="random guess")
    plt.xticks(rotation=45, ha="right"); plt.ylim(0, 1.05)
    plt.ylabel("accuracy"); plt.title("Robustness of the classifier")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "robustness.png"), dpi=120)
    print("Saved outputs/robustness_results.csv and outputs/robustness.png")


if __name__ == "__main__":
    main()
