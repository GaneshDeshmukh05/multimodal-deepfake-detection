import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers, models
import soundfile as sf
import tempfile

# PAGE CONFIG
st.set_page_config(page_title="Audio Deepfake Detector", layout="centered")

st.title("Audio Deepfake Detection System")
st.write("Upload an audio file to detect whether it is REAL or FAKE")

# MODEL
def build_model(input_shape=(128, 94, 1)):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),

        layers.Dense(1, activation='sigmoid')
    ])
    return model

# LOAD MODEL
model_path = os.path.join("models", "audio.weights.h5")

if not os.path.exists(model_path):
    st.error(" Model weights not found.")
    st.stop()

model = build_model()
model.load_weights(model_path)

# FEATURE EXTRACTION (FIXED)
def extract_features(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        audio, sr = sf.read(temp_path)

        # Convert stereo → mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        #FIX: ensure 2 sec length (32000 samples)
        if len(audio) < 32000:
            audio = np.pad(audio, (0, 32000 - len(audio)))
        else:
            audio = audio[:32000]

        # Spectrogram
        spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        log_spec = librosa.power_to_db(spec)

        #FIX: force shape (128, 94)
        if log_spec.shape[1] < 94:
            pad_width = 94 - log_spec.shape[1]
            log_spec = np.pad(log_spec, ((0,0),(0,pad_width)))
        else:
            log_spec = log_spec[:, :94]

        return log_spec[..., np.newaxis], audio, sr

    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None, None

#FILE UPLOAD
uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3", "flac"]
)

#PREDICTION
if uploaded_file is not None:

    st.audio(uploaded_file)
    uploaded_file.seek(0)

    features, audio, sr = extract_features(uploaded_file)

    if features is not None:
        st.write("Feature shape:", features.shape)

        features = np.expand_dims(features, axis=0)

        prediction = model.predict(features)
        st.write("Raw prediction:", prediction)

        pred_value = float(prediction[0][0])

        if pred_value > 0.5:
            result = "FAKE"
            confidence = pred_value
        else:
            result = "REAL"
            confidence = 1 - pred_value

        st.subheader(f"Prediction: {result}")
        st.progress(confidence)
        st.write(f"Confidence: {confidence:.2f}")

        # WAVEFORM
        st.subheader("Waveform")
        fig, ax = plt.subplots()
        ax.plot(audio)
        st.pyplot(fig)

        #SPECTROGRAM
        st.subheader("Spectrogram")
        fig2, ax2 = plt.subplots()
        ax2.imshow(features[0].squeeze(), aspect='auto')
        st.pyplot(fig2)

#FOOTER
st.markdown("---")
st.caption("Deepfake Detection Project | Audio Model")