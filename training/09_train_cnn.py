
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Locate project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# 2. Load the training dataset
df = pd.read_csv(DATASET_DIR / "sign_mnist_train.csv")

# 3. Separate pixels and original labels
X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
y_original = df["label"].to_numpy(dtype=np.int32)

# 4. Remap labels to consecutive numbers: 0 to 23
labels = sorted(np.unique(y_original))
label_map = {label: index for index, label in enumerate(labels)}
y = np.array([label_map[label] for label in y_original])

# Save mapping for later predictions
letter_map = {
    new_label: chr(ord("A") + original_label
                   + (1 if original_label >= 9 else 0))
    for original_label, new_label in label_map.items()
}
np.save(MODEL_DIR / "label_map.npy", letter_map)

# 5. Normalize and reshape images
X = X / 255.0
X = X.reshape(-1, 28, 28, 1)

# 6. Split training and validation data
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 7. Build the CNN
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(24, activation="softmax")
])

# 8. Configure training
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 9. Train the CNN
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_val, y_val)
)

# 10. Save the trained model
model.save(MODEL_DIR / "sign_language_cnn.keras")

print("\nModel saved successfully!")
print("Location:", MODEL_DIR / "sign_language_cnn.keras")
