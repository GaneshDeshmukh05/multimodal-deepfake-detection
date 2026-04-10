import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import soundfile as sf
import tempfile
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Multimodal Deepfake Detection", layout="wide")

st.title("Multimodal Deepfake Detection System")
st.write("Upload Image, Audio, or Video to detect if it is REAL or FAKE")

# AUDIO MODEL ARCHITECTURE
def build_audio_model(input_shape=(128,94,1)):
    model = models.Sequential([
        layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64,(3,3),activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128,(3,3),activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.GlobalAveragePooling2D(),

        layers.Dense(128,activation='relu'),
        layers.Dropout(0.4),

        layers.Dense(1,activation='sigmoid')
    ])
    return model

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():

    # AUDIO MODEL
    audio_model = build_audio_model()
    audio_path = os.path.join("models","audio.weights.h5")

    if os.path.exists(audio_path):
        audio_model.load_weights(audio_path)
        st.sidebar.success("Audio model loaded")
    else:
        st.sidebar.error("Audio weights missing")

audio_model = load_models()


# =========================
# SIDEBAR OPTION
# =========================
option = st.sidebar.selectbox(
    "Select Detection Type",
    ["Audio Detection","Video Detection"]
)

# =====================================================
# AUDIO DETECTION
# =====================================================
if option == "Audio Detection":

    st.header("Audio Deepfake Detection")

    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=["wav","mp3","flac"]
    )

    if uploaded_file is not None:

        st.audio(uploaded_file)
        uploaded_file.seek(0)

        try:

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.read())
                temp_path = tmp.name

            audio, sr = sf.read(temp_path)

            # stereo → mono
            if len(audio.shape) > 1:
                audio = np.mean(audio,axis=1)

            # resample
            if sr != 16000:
                audio = librosa.resample(audio,orig_sr=sr,target_sr=16000)
                sr = 16000

            # fix length
            if len(audio) < 32000:
                audio = np.pad(audio,(0,32000-len(audio)))
            else:
                audio = audio[:32000]

            # spectrogram
            spec = librosa.feature.melspectrogram(y=audio,sr=sr,n_mels=128)
            log_spec = librosa.power_to_db(spec)

            if log_spec.shape[1] < 94:
                pad = 94 - log_spec.shape[1]
                log_spec = np.pad(log_spec,((0,0),(0,pad)))
            else:
                log_spec = log_spec[:,:94]

            features = log_spec[...,np.newaxis]
            features = np.expand_dims(features,0)

            prediction = audio_model.predict(features)

            value = float(prediction[0][0])

            if value > 0.5:
                result="FAKE"
                confidence=value
            else:
                result="REAL"
                confidence=1-value

            st.subheader(f"Prediction: {result}")
            st.progress(confidence)
            st.write(f"Confidence: {confidence:.2f}")

            # waveform
            fig,ax = plt.subplots()
            ax.plot(audio)
            st.pyplot(fig)

            # spectrogram
            fig2,ax2 = plt.subplots()
            ax2.imshow(log_spec,aspect='auto')
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"Audio processing error: {e}")



# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Deepfake Detection Project | Multimodal AI")