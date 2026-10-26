
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow import keras

# Project paths
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

# Reproduce the class ordering used during training
train_df = pd.read_csv(DATASET_DIR / "sign_mnist_train.csv")
original_labels = sorted(train_df["label"].unique())
class_to_original = {
    index: label for index, label in enumerate(original_labels)
}

# Load the trained model only once
model = keras.models.load_model(MODEL_PATH)
print("CNN model loaded successfully.")

# Start webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Could not open webcam. Check camera permissions or camera index."
    )

print("Webcam started. Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read webcam frame.")
        break

    # Mirror the image like a selfie camera
    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    # Create a square region in the center
    
# Use a smaller square so your hand fills more of the image
    box_size = min(200, height, width)

    x1 = (width - box_size) // 2
    y1 = (height - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size


    roi = frame[y1:y2, x1:x2]

    # Convert webcam crop to the model's input format
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Resize the hand region to the CNN's input size
    resized = cv2.resize(gray, (28, 28))

    # Show an enlarged preview of the exact image sent to the CNN
    preview = cv2.resize(
        resized, (280, 280),
        interpolation=cv2.INTER_NEAREST
    )
    cv2.imshow("What the CNN sees", preview)

    normalized = resized.astype(np.float32) / 255.0

    # Shape: batch, height, width, channel
    model_input = normalized.reshape(1, 28, 28, 1)

    probabilities = model.predict(model_input, verbose=0)[0]
    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class]) * 100

    original_label = class_to_original[predicted_class]
    letter = LABEL_TO_LETTER[original_label]

    # Draw the region and prediction
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    text = f"Letter: {letter} | Confidence: {confidence:.1f}%"
    cv2.putText(
        frame, text, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (0, 255, 0), 2
    )

    cv2.putText(
        frame, "Place one hand inside the box | Q = quit",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (255, 255, 255), 2
    )

    cv2.imshow("SignBridge - Webcam Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
