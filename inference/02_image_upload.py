
import cv2
import numpy as np
import pandas as pd
import tkinter as tk

from pathlib import Path
from tkinter import filedialog
from tensorflow import keras

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_PATH = PROJECT_DIR / "models" / "sign_language_cnn.keras"

# Original Sign Language MNIST labels
LABEL_TO_LETTER = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I",
    10: "K", 11: "L", 12: "M", 13: "N",
    14: "O", 15: "P", 16: "Q", 17: "R",
    18: "S", 19: "T", 20: "U", 21: "V",
    22: "W", 23: "X", 24: "Y"
}

# Match the consecutive class ordering used during training
train_df = pd.read_csv(DATASET_DIR / "sign_mnist_train.csv")
original_labels = sorted(train_df["label"].unique())

class_to_original = {
    index: label for index, label in enumerate(original_labels)
}

# Open a file picker
root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Choose an ASL hand sign image",
    filetypes=[
        ("Image files", "*.png *.jpg *.jpeg *.bmp"),
        ("All files", "*.*")
    ]
)

if not image_path:
    print("No image selected.")
    raise SystemExit

# Load the trained CNN
model = keras.models.load_model(MODEL_PATH)

# Read the selected image
image = cv2.imread(image_path)

if image is None:
    print("Could not open this image.")
    raise SystemExit

# Convert to grayscale and resize
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (28, 28))

# Normalize pixel values
normalized = resized.astype(np.float32) / 255.0

# CNN input shape: (batch, height, width, channel)
model_input = normalized.reshape(1, 28, 28, 1)

# Predict
probabilities = model.predict(model_input, verbose=0)[0]
predicted_class = int(np.argmax(probabilities))
confidence = float(probabilities[predicted_class]) * 100

original_label = class_to_original[predicted_class]
predicted_letter = LABEL_TO_LETTER[original_label]

print("\n--- SignBridge Prediction ---")
print("Image:", Path(image_path).name)
print("Predicted letter:", predicted_letter)
print(f"Model confidence: {confidence:.2f}%")

# Display the uploaded image and its prediction
cv2.imshow("Uploaded Image", image)
cv2.imshow("28x28 CNN Input", cv2.resize(resized, (280, 280)))

print("\nPress any key in the image window to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()
root.destroy()
