# predict.py - classify every audio file in a folder and write a CSV
# usage: python predict.py data/real            (folder of audio files)
#        python predict.py some_folder out.csv
# If the folder is called "real" or "fake" the true label is known, so the
# CSV also contains an evaluation column.
import os
import sys
import glob
import joblib
import pandas as pd
import config
import features


def predict_file(model, path):
    """Return (predicted_class, confidence) for one audio file."""
    x = features.features_from_file(path).reshape(1, -1)
    proba = model.predict_proba(x)[0]
    idx = int(proba.argmax())
    return config.CLASS_NAMES[idx], float(proba[idx]), proba


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/real"
    out_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join(config.OUTPUT_DIR, "predictions.csv")
    model = joblib.load(config.MODEL_PATH)["model"]

    parent = os.path.basename(os.path.normpath(folder))
    true_label = {"real": "genuine", "fake": "synthetic"}.get(parent)

    files = []
    for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
        files += glob.glob(os.path.join(folder, ext))
    rows = []
    for p in sorted(files):
        cls, conf, _ = predict_file(model, p)
        row = {"file_id": os.path.basename(p), "predicted_class": cls, "confidence": round(conf, 4)}
        if true_label:
            row["true_class"] = true_label
            row["evaluation"] = "correct" if cls == true_label else "wrong"
        rows.append(row)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(df.head(10))
    if true_label:
        print("Accuracy on this folder:", round((df["evaluation"] == "correct").mean(), 3))
    print("Saved", out_csv)


if __name__ == "__main__":
    main()
