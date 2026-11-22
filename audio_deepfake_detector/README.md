# T245 - AI-Based Audio Deepfake & Speaker Spoofing Detection

Mini project: a classifier that decides whether a speech recording is **genuine** or
**synthetic / spoofed**, with a small web interface.

## What the project does (requirements checklist)
| Requirement | Where |
|---|---|
| Extract mel-spectrograms and MFCCs | `features.py` |
| Train classifier on genuine + synthetic speech | `train.py` (Random Forest and SVM, best one is saved) |
| Robustness across recording conditions and compression | `augment.py`, `evaluate.py` (noise, reverb, telephone band, 8 kHz, MP3 128k/64k/32k) |
| Interface with prediction confidence | `app.py` (Streamlit) |
| CSV with file ID, predicted class, confidence, evaluation | `predict.py` -> `outputs/predictions.csv`, `outputs/robustness_results.csv` |

## Folder structure
```
data/real/        genuine speech files
data/fake/        synthetic speech files
samples/          4 small files to try in the interface
models/           saved model (created by train.py)
outputs/          csv files and graphs
config.py  features.py  augment.py  train.py  evaluate.py  predict.py  app.py  generate_demo_data.py
```

## How to run
```
pip install -r requirements.txt
# ffmpeg must be installed for the MP3 tests (optional, they are skipped otherwise)

python generate_demo_data.py      # only for the toy dataset (skip if you use a real dataset)
python train.py --augment         # train (remove --augment to train on clean audio only)
python evaluate.py                # robustness test -> outputs/robustness_results.csv
python predict.py data/fake       # batch prediction -> outputs/predictions.csv
streamlit run app.py              # open the interface in the browser
```

## IMPORTANT note about the dataset
No internet dataset was available while building this, so `generate_demo_data.py` creates
a **toy dataset** (computer-generated voice-like sounds, NOT real speech). Genuine clips have
natural pitch wobble, jitter and breath/room noise; synthetic clips are smoother, cleaner and
band-limited with a small periodic artefact. The scores on this toy data are therefore much
higher than they would be on real deepfakes.

To use real data: put files in `data/real` and `data/fake`
(e.g. ASVspoof 2019 LA, WaveFake, Fake-or-Real) and run `train.py` again. Nothing else changes.

## Method
1. Audio -> mono, 16 kHz, 4 seconds.
2. Features: 20 MFCC + delta (mean, std), 40-band log-mel spectrogram (mean, std),
   spectral flatness / centroid / rolloff, zero-crossing rate -> about 200 numbers per file.
3. Classifiers: Random Forest and SVM (RBF). Train/test split is done per file (75/25).
4. Confidence = predicted probability of the winning class.
5. Robustness: the test files are distorted (noise at 20/10/5 dB SNR, reverb, 300-3400 Hz phone
   band, 8 kHz resampling, MP3 at 128/64/32 kbps) and accuracy / F1 are measured again.

## Results on the toy dataset (model trained with --augment)
| condition | accuracy |
|---|---|
| clean | 1.00 |
| noise 20 dB / 10 dB / 5 dB | 1.00 / 0.95 / 0.85 |
| reverb | 0.95 |
| telephone band | 0.99 |
| 8 kHz resample | 0.99 |
| MP3 128k / 64k / 32k | 1.00 / 1.00 / 0.59 |

Observation: the model is fine up to moderate noise and compression, but very strong MP3
compression (32 kbps) removes the artefacts the model relies on and accuracy drops a lot.

## Limitations / future work
- Toy data only; real deepfakes need a real dataset and will be harder.
- Hand-crafted features + classic ML; a CNN on mel-spectrograms or wav2vec features would be stronger.
- Unseen vocoders / unseen speakers are not tested.
- Training with more codec types as augmentation would improve the 32k MP3 result.
