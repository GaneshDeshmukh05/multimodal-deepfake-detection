import streamlit as st
import cv2
import numpy as np
import librosa
import tempfile
import os
import tensorflow as tf
from moviepy.editor import VideoFileClip

# ── Constants (must match training) ──────────────────────────────────────────
FRAME_SIZE = 128
MAX_FRAMES  = 15
AUDIO_LEN   = 100

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Video Detector",
    page_icon="🕵️",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .result-real  { background: #1a3a2a; border: 2px solid #2ecc71;
                    border-radius: 12px; padding: 20px; text-align: center; }
    .result-fake  { background: #3a1a1a; border: 2px solid #e74c3c;
                    border-radius: 12px; padding: 20px; text-align: center; }
    .result-label { font-size: 2rem; font-weight: 700; }
    .result-prob  { font-size: 1.1rem; color: #aaa; margin-top: 6px; }
    .step-box     { background: #1c1f26; border-radius: 8px;
                    padding: 14px 18px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🕵️ Deepfake Video Detector")
st.caption("Multimodal detection — analyses both video frames and audio together.")

# ── Model loader (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)

# ── Feature extractors ────────────────────────────────────────────────────────
def extract_frames(video_path: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, total // MAX_FRAMES)
    fc    = 0
    while cap.isOpened() and len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if fc % step == 0:
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
            frame = frame.astype("float32") / 255.0
            frames.append(frame)
        fc += 1
    cap.release()
    return np.array(frames) if len(frames) == MAX_FRAMES else None


def extract_audio(video_path: str) -> np.ndarray | None:
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            clip.close()
            return None
        clip.audio.write_audiofile(wav_path, verbose=False, logger=None)
        clip.close()

        y, sr = librosa.load(wav_path, sr=16000)
        mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        if mfcc.shape[1] < AUDIO_LEN:
            mfcc = np.pad(mfcc, ((0, 0), (0, AUDIO_LEN - mfcc.shape[1])))
        else:
            mfcc = mfcc[:, :AUDIO_LEN]
        os.unlink(wav_path)
        return mfcc
    except Exception as e:
        st.warning(f"Audio extraction issue: {e}")
        return None

# ── Sidebar — model path ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    model_path = st.text_input(
        "Model path (.h5)",
        value="models/video_deepfake_model.h5",
        help="Path to your trained Keras model file.",
    )
    st.divider()
    st.markdown("**Model expects:**")
    st.markdown(f"- {MAX_FRAMES} frames at {FRAME_SIZE}×{FRAME_SIZE}px")
    st.markdown(f"- MFCC audio features: (40, {AUDIO_LEN})")
    st.divider()
    st.markdown("**Threshold:** 0.5  \n> 0.5 → Fake  \n≤ 0.5 → Real")

# ── Load model ────────────────────────────────────────────────────────────────
if not os.path.exists(model_path):
    st.error(f"Model file not found at `{model_path}`. Update the path in the sidebar.")
    st.stop()

with st.spinner("Loading model…"):
    model = load_model(model_path)
st.success("Model loaded.", icon="✅")

# ── Upload ────────────────────────────────────────────────────────────────────
st.divider()
uploaded = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov", "mkv"],
    help="Short clips (< 30 s) work best for speed.",
)

if not uploaded:
    st.info("Upload a video above to run detection.")
    st.stop()

# ── Save upload to temp file ──────────────────────────────────────────────────
suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

st.video(tmp_path)

# ── Run prediction ────────────────────────────────────────────────────────────
if st.button("🔍 Analyse Video", use_container_width=True, type="primary"):

    with st.status("Running analysis…", expanded=True) as status:

        st.write("📹 Extracting frames…")
        frames = extract_frames(tmp_path)
        if frames is None:
            status.update(label="Failed", state="error")
            st.error(f"Could not extract {MAX_FRAMES} frames. Try a longer video.")
            os.unlink(tmp_path)
            st.stop()

        st.write("🎙️ Extracting audio features…")
        mfcc = extract_audio(tmp_path)
        if mfcc is None:
            status.update(label="Failed", state="error")
            st.error("Audio extraction failed. Make sure the video has an audio track.")
            os.unlink(tmp_path)
            st.stop()

        st.write("🧠 Running model inference…")
        video_input = np.expand_dims(frames, axis=0)   # (1, 15, 128, 128, 3)
        audio_input = np.expand_dims(mfcc,   axis=0)   # (1, 40, 100)
        prob = float(model.predict([video_input, audio_input], verbose=0)[0][0])

        status.update(label="Done!", state="complete", expanded=False)

    # ── Results ───────────────────────────────────────────────────────────────
    st.divider()
    is_fake = prob > 0.5
    confidence = prob if is_fake else (1 - prob)

    if is_fake:
        st.markdown(f"""
        <div class="result-fake">
            <div class="result-label">⚠️ FAKE DETECTED</div>
            <div class="result-prob">Fake probability: {prob:.1%}  ·  Confidence: {confidence:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-real">
            <div class="result-label">✅ LIKELY REAL</div>
            <div class="result-prob">Fake probability: {prob:.1%}  ·  Confidence: {confidence:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Fake Probability",  f"{prob:.1%}")
    col2.metric("Real Probability",  f"{1 - prob:.1%}")
    col3.metric("Verdict", "FAKE" if is_fake else "REAL")

    # Probability bar
    st.markdown("**Fake probability scale**")
    st.progress(prob)

    # Sample frames preview
    with st.expander("Sampled Frames", expanded=False):
        cols = st.columns(5)
        for i, col in enumerate(cols):
            idx = i * (MAX_FRAMES // 5)
            col.image(frames[idx], caption=f"Frame {idx}", use_container_width=True)

os.unlink(tmp_path) if os.path.exists(tmp_path) else None
