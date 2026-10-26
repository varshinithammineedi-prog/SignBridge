
import pandas as pd
import numpy as np
from pathlib import Path
from tensorflow import keras
from sklearn.metrics import classification_report, accuracy_score

# 1. Locate files
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_PATH = PROJECT_DIR / "models" / "sign_language_cnn.keras"

# 2. Load the untouched test dataset
test_df = pd.read_csv(DATASET_DIR / "sign_mnist_test.csv")

# 3. Prepare test images
X_test = test_df.drop(columns=["label"]).to_numpy(dtype=np.float32)
y_original = test_df["label"].to_numpy(dtype=np.int32)

X_test = X_test / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)

# 4. Remap labels exactly as during training
all_labels = sorted(
    pd.read_csv(DATASET_DIR / "sign_mnist_train.csv")["label"].unique()
)
label_map = {label: index for index, label in enumerate(all_labels)}

y_test = np.array([label_map[label] for label in y_original])

# 5. Load the saved CNN and predict
model = keras.models.load_model(MODEL_PATH)
probabilities = model.predict(X_test)
predictions = np.argmax(probabilities, axis=1)

# 6. Evaluate
print("Test images:", len(X_test))
print("Test accuracy:", accuracy_score(y_test, predictions))
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))
