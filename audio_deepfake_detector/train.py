# train.py - extract features, train classifiers, save the best one
# usage:  python train.py            (train on clean audio only)
#         python train.py --augment  (also train on noisy / compressed copies)
import os
import sys
import glob
import json
import random
import numpy as np
import joblib
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import config
import features
import augment


def list_files():
    files, labels = [], []
    for label, folder in enumerate(["real", "fake"]):
        for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
            for p in glob.glob(os.path.join(config.DATA_DIR, folder, ext)):
                files.append(p)
                labels.append(label)
    return files, labels


def main():
    use_aug = "--augment" in sys.argv
    np.random.seed(config.RANDOM_STATE)
    random.seed(config.RANDOM_STATE)

    files, labels = list_files()
    if len(files) == 0:
        print("No audio found. Put files in data/real and data/fake, or run generate_demo_data.py")
        return
    print(f"Found {len(files)} files ({labels.count(0)} genuine, {labels.count(1)} synthetic)")

    # split by FILE so the same clip never appears in train and test
    f_train, f_test, y_train, y_test = train_test_split(
        files, labels, test_size=0.25, stratify=labels, random_state=config.RANDOM_STATE)
    os.makedirs("models", exist_ok=True)
    json.dump({"test_files": f_test, "test_labels": y_test}, open(config.SPLIT_PATH, "w"), indent=1)

    print("Extracting features from training files...")
    X_train, Y_train = [], []
    for i, (p, y) in enumerate(zip(f_train, y_train)):
        wav = features.load_audio(p)
        X_train.append(features.extract_features(wav)); Y_train.append(y)
        if use_aug:
            name = random.choice(augment.TRAIN_AUG)          # one random distorted copy
            X_train.append(features.extract_features(augment.CONDITIONS[name](wav))); Y_train.append(y)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(f_train)}")
    X_train, Y_train = np.array(X_train), np.array(Y_train)

    print("Extracting features from test files...")
    X_test = np.array([features.features_from_file(p) for p in f_test])
    y_test = np.array(y_test)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_STATE),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, probability=True,
                                                   random_state=config.RANDOM_STATE)),
    }
    best_name, best_acc, best_model = None, -1, None
    for name, m in models.items():
        m.fit(X_train, Y_train)
        acc = accuracy_score(y_test, m.predict(X_test))
        print(f"{name}: test accuracy = {acc:.3f}")
        if acc > best_acc:
            best_name, best_acc, best_model = name, acc, m

    print(f"\nBest model: {best_name}")
    pred = best_model.predict(X_test)
    print(classification_report(y_test, pred, target_names=config.CLASS_NAMES))

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(cm, display_labels=config.CLASS_NAMES).plot(cmap="Blues")
    plt.title(f"Confusion matrix ({best_name})")
    plt.savefig(os.path.join(config.OUTPUT_DIR, "confusion_matrix.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # a sample mel-spectrogram picture for the report
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    for a, (lab, name) in zip(ax, [(0, "genuine"), (1, "synthetic")]):
        p = next(f for f, l in zip(files, labels) if l == lab)
        mel = features.mel_spectrogram(features.load_audio(p))
        a.imshow(mel, origin="lower", aspect="auto"); a.set_title(f"Mel-spectrogram: {name}")
        a.set_xlabel("time frames"); a.set_ylabel("mel band")
    plt.savefig(os.path.join(config.OUTPUT_DIR, "mel_examples.png"), dpi=120, bbox_inches="tight")
    plt.close()

    joblib.dump({"model": best_model, "name": best_name, "augmented": use_aug}, config.MODEL_PATH)
    print("Model saved to", config.MODEL_PATH)


if __name__ == "__main__":
    main()
