
import pandas as pd
import numpy as np
from pathlib import Path
from tensorflow import keras

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_PATH = PROJECT_DIR / "models" / "sign_language_cnn.keras"

# Correct mapping from the original dataset labels
LABEL_TO_LETTER = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I",
    10: "K", 11: "L", 12: "M", 13: "N",
    14: "O", 15: "P", 16: "Q", 17: "R",
    18: "S", 19: "T", 20: "U", 21: "V",
    22: "W", 23: "X", 24: "Y"
}

df = pd.read_csv(DATASET_DIR / "sign_mnist_test.csv")

# Keep the same class ordering used during training
train_df = pd.read_csv(DATASET_DIR / "sign_mnist_train.csv")
original_labels = sorted(train_df["label"].unique())
class_to_original = {
    index: label for index, label in enumerate(original_labels)
}

X_test = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
X_test = X_test / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)

model = keras.models.load_model(MODEL_PATH)

# Inspect five sample predictions
probabilities = model.predict(X_test[:5], verbose=0)
predicted_classes = np.argmax(probabilities, axis=1)

for i, predicted_class in enumerate(predicted_classes):
    original_label = class_to_original[int(predicted_class)]
    predicted_letter = LABEL_TO_LETTER[original_label]
    confidence = float(np.max(probabilities[i])) * 100

    actual_label = int(df.iloc[i]["label"])
    actual_letter = LABEL_TO_LETTER[actual_label]

    print(f"\nImage {i + 1}")
    print(f"Actual letter:    {actual_letter}")
    print(f"Predicted letter: {predicted_letter}")
    print(f"Confidence:       {confidence:.2f}%")
