# Bimodal Emotion Detection

Real-time emotion recognition system that combines **facial expression analysis** (video) and **speech prosody** (audio) to classify human emotion more accurately than either signal alone. Originally built as an undergraduate thesis project in Computer Science and Engineering.

## What It Does

The system detects five core emotions — **Happy, Sad, Angry, Fear, Neutral** — by running two models in parallel and fusing their outputs:

- **Audio model:** Extracts MFCCs (Mel-Frequency Cepstral Coefficients) from speech and classifies emotion using an LSTM network, which is well suited to picking up patterns in pitch, tone, and rhythm over time.
- **Video model:** Detects faces in webcam frames, crops and processes them, and classifies emotion using a CNN trained to recognize facial expression patterns.
- **Fusion logic:** Combines both predictions. If both models agree, that's the output. If they disagree, the system trusts whichever model has the higher confidence score. If one modality fails (e.g. face occluded, audio too noisy), it falls back to the other.

## How It Works

1. **Audio pipeline:** raw audio → resampling, mono conversion, silence trimming, normalization, noise reduction → MFCC feature extraction → LSTM classification
2. **Video pipeline:** webcam frames → face detection (Haar Cascades + OpenCV DNN) → crop/resize to 48x48 grayscale → CNN classification
3. **Fusion:** softmax outputs from both models are compared and merged into one final emotion label, displayed live on screen with a confidence score

## Results

| Modality | Model | Accuracy |
|---|---|---|
| Audio | LSTM | ~82% |
| Video | CNN | ~68% |
| Fused | Late Fusion | +5–10% over either model alone |

Best recognition was on **Happy** and **Angry**. Most common confusion was **Sad vs. Neutral**, a known difficulty in emotion recognition since both tend to show low facial and vocal intensity.

## Datasets

- **Audio:** [RAVDESS](https://zenodo.org/record/1188976) (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Video:** A custom dataset (~5,500 clips) manually sourced and labeled from movies, interviews, and short-form video, trimmed into 5-second segments using FFmpeg

## Tech Stack

Python · TensorFlow/Keras · OpenCV · librosa · scikit-learn · PyAudio · FFmpeg

## Setup

```bash
git clone https://github.com/<your-username>/bimodal-emotion-detection.git
cd bimodal-emotion-detection
pip install -r requirements.txt
jupyter notebook emotion_detection.ipynb
```

> Note: pretrained weights aren't included. You'll need to train on the datasets above, or plug in your own.

## Limitations

- Real-time predictions update every ~5–7 seconds rather than continuously
- Video faces are resized to a small 48x48 resolution, which may lose some subtle expression detail
- Minimal error handling around file/model loading
- No pretrained/transfer learning weights used, trained from scratch due to resource constraints

## Background

This project began as an undergraduate thesis, *"Towards Real-Time Emotion Analytics: Integrating Facial Landmarks and Speech Prosody,"* exploring how combining audio and visual emotion cues improves on single-modality systems, particularly in noisy or visually obstructed conditions where one modality alone tends to fail.

## License

MIT
