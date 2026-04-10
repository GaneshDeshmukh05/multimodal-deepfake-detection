import os
import cv2
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split

# SETTINGS
MAX_FRAMES = 15
IMG_SIZE = 128
AUDIO_LEN = 100

FRAME_PATH = "frames"
AUDIO_PATH = "audio"

# VIDEO LOADER
def load_video_frames(folder):

    frames = []

    files = sorted(os.listdir(folder))[:MAX_FRAMES]

    for file in files:
        img = cv2.imread(os.path.join(folder, file))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        frames.append(img)

    if len(frames) != MAX_FRAMES:
        return None

    return np.array(frames)

# AUDIO LOADER
def load_audio_features(file_path):

    y, sr = librosa.load(file_path, sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    if mfcc.shape[1] < AUDIO_LEN:
        pad = AUDIO_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad)))
    else:
        mfcc = mfcc[:, :AUDIO_LEN]

    return mfcc

# DATASET CREATION
video_data = []
audio_data = []
labels = []

categories = ["real", "fake"]

for label, cat in enumerate(categories):

    frame_dir = os.path.join(FRAME_PATH, f"{cat}_frames")
    audio_dir = os.path.join(AUDIO_PATH, f"{cat}_audio")

    for video_folder in os.listdir(frame_dir):

        frame_path = os.path.join(frame_dir, video_folder)
        audio_file = os.path.join(audio_dir, video_folder + ".wav")

        if not os.path.exists(audio_file):
            continue

        frames = load_video_frames(frame_path)
        audio_feat = load_audio_features(audio_file)

        if frames is None:
            continue

        video_data.append(frames)
        audio_data.append(audio_feat)
        labels.append(label)

video_data = np.array(video_data)
audio_data = np.array(audio_data)
labels = np.array(labels)

print("Video shape:", video_data.shape)
print("Audio shape:", audio_data.shape)

# TRAIN TEST SPLIT
Xv_train, Xv_test, Xa_train, Xa_test, y_train, y_test = train_test_split(
    video_data, audio_data, labels, test_size=0.2, random_state=42
)

# VIDEO MODEL (EfficientNet)
video_input = layers.Input(shape=(MAX_FRAMES, IMG_SIZE, IMG_SIZE, 3))

base_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet',
    include_top=False,
    pooling='avg'
)

base_model.trainable = False

x = layers.TimeDistributed(base_model)(video_input)
x = layers.LSTM(128)(x)

# AUDIO MODEL
audio_input = layers.Input(shape=(40, AUDIO_LEN))

y = layers.Conv1D(64, 3, activation='relu')(audio_input)
y = layers.MaxPooling1D(2)(y)
y = layers.Flatten()(y)

# FUSION
combined = layers.concatenate([x, y])

z = layers.Dense(64, activation='relu')(combined)
z = layers.Dropout(0.5)(z)
output = layers.Dense(1, activation='sigmoid')(z)

model = Model(inputs=[video_input, audio_input], outputs=output)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# TRAIN MODEL
history = model.fit(
    [Xv_train, Xa_train],
    y_train,
    validation_data=([Xv_test, Xa_test], y_test),
    epochs=10,
    batch_size=2
)

# SAVE MODEL
os.makedirs("models", exist_ok=True)

model.save("models/multimodal_deepfake_model.h5")

print("Model saved successfully")