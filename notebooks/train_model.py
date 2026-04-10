import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from utils.data_loader import load_data
from models.resnet_lstm_deepfake import build_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

print("Loading dataset...")

# Load data
real_data = load_data("frames/real_frames", 0)
fake_data = load_data("frames/fake_frames", 1)

data = real_data + fake_data

# Check dataset
if len(data) == 0:
    raise ValueError("Dataset is empty! Check frame extraction.")

# Split X and y
X = np.array([item[0] for item in data])
y = np.array([item[1] for item in data])

print("Dataset loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Train-test split (BETTER than validation_split)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Build model
model = build_model()

print("\nModel Summary:")
model.summary()

# Callbacks (VERY IMPORTANT)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "models/best_model.h5",
    monitor='val_accuracy',
    save_best_only=True
)

print("\nTraining started...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=4,
    callbacks=[early_stop, checkpoint]
)

# Save final model
model.save("models/deepfake_model.h5")

print("\nModel saved successfully!")