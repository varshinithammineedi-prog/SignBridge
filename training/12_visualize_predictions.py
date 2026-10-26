
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow import keras

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_PATH = PROJECT_DIR / "models" / "sign_language_cnn.keras"

LABEL_TO_LETTER = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I",
    10: "K", 11: "L", 12: "M", 13: "N",
    14: "O", 15: "P", 16: "Q", 17: "R",
    18: "S", 19: "T", 20: "U", 21: "V",
    22: "W", 23: "X", 24: "Y"
}

train_df = pd.read_csv(DATASET_DIR / "sign_mnist_train.csv")
test_df = pd.read_csv(DATASET_DIR / "sign_mnist_test.csv")

original_labels = sorted(train_df["label"].unique())
class_to_original = {
    index: label for index, label in enumerate(original_labels)
}

X_test = test_df.drop(columns=["label"]).to_numpy(
    dtype=np.float32
) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)

y_actual = test_df["label"].to_numpy(dtype=np.int32)

model = keras.models.load_model(MODEL_PATH)

# Select 12 test images
number_of_images = 12
probabilities = model.predict(
    X_test[:number_of_images], verbose=0
)
predicted_classes = np.argmax(probabilities, axis=1)

fig, axes = plt.subplots(3, 4, figsize=(12, 9))
axes = axes.flatten()

for i, ax in enumerate(axes):
    actual_letter = LABEL_TO_LETTER[int(y_actual[i])]

    original_prediction = class_to_original[
        int(predicted_classes[i])
    ]
    predicted_letter = LABEL_TO_LETTER[original_prediction]

    confidence = float(np.max(probabilities[i])) * 100
    correct = actual_letter == predicted_letter

    ax.imshow(X_test[i].squeeze(), cmap="gray")
    ax.set_title(
        f"Actual: {actual_letter} | Predicted: {predicted_letter}\n"
        f"Confidence: {confidence:.1f}% | "
        f"{'Correct' if correct else 'Incorrect'}"
    )
    ax.axis("off")

plt.tight_layout()
plt.show()
