# app.py - simple web interface (Streamlit)
# run with:  streamlit run app.py
import os
import tempfile
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import config
import features

st.set_page_config(page_title="Audio Deepfake Detector", layout="centered")
st.title("Audio Deepfake & Speaker Spoofing Detector")
st.write("Upload a speech recording. The model says whether it is **genuine** or **synthetic (AI-generated / spoofed)**.")


@st.cache_resource
def load_model():
    return joblib.load(config.MODEL_PATH)["model"]


if not os.path.exists(config.MODEL_PATH):
    st.error("Model not found. Run `python train.py` first.")
    st.stop()
model = load_model()

uploaded = st.file_uploader("Choose audio file(s)", type=["wav", "flac", "mp3", "ogg"],
                            accept_multiple_files=True)

results = []
for up in uploaded:
    suffix = os.path.splitext(up.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(up.read())
        path = tmp.name

    y = features.load_audio(path)
    x = features.extract_features(y).reshape(1, -1)
    proba = model.predict_proba(x)[0]
    idx = int(proba.argmax())
    label, conf = config.CLASS_NAMES[idx], float(proba[idx])
    results.append({"file_id": up.name, "predicted_class": label, "confidence": round(conf, 4)})

    st.subheader(up.name)
    st.audio(up.getvalue())
    if label == "genuine":
        st.success(f"Prediction: GENUINE  (confidence {conf * 100:.1f}%)")
    else:
        st.error(f"Prediction: SYNTHETIC / SPOOFED  (confidence {conf * 100:.1f}%)")
    st.progress(conf)
    st.write(f"P(genuine) = {proba[0]:.3f}   |   P(synthetic) = {proba[1]:.3f}")

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.imshow(features.mel_spectrogram(y), origin="lower", aspect="auto")
    ax.set_title("Mel-spectrogram"); ax.set_xlabel("time frames"); ax.set_ylabel("mel band")
    st.pyplot(fig)
    plt.close(fig)
    os.remove(path)

if results:
    st.divider()
    df = pd.DataFrame(results)
    st.dataframe(df)
    st.download_button("Download results as CSV", df.to_csv(index=False), "predictions.csv", "text/csv")

# show robustness results if evaluate.py was already run
rob = os.path.join(config.OUTPUT_DIR, "robustness_results.csv")
if os.path.exists(rob):
    with st.expander("Model robustness results (from evaluate.py)"):
        st.dataframe(pd.read_csv(rob))
        if os.path.exists(os.path.join(config.OUTPUT_DIR, "robustness.png")):
            st.image(os.path.join(config.OUTPUT_DIR, "robustness.png"))
