
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# 1. Locate and load the training CSV
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"

df = pd.read_csv(DATASET_DIR / "sign_mnist_train.csv")

# 2. Separate images (X) and correct labels (y)
X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
y = df["label"].to_numpy(dtype=np.int32)

# 3. Normalize pixel values to the range 0–1
X = X / 255.0

# 4. Reshape images for the CNN
X = X.reshape(-1, 28, 28, 1)

# 5. Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Display the results
print("Training images:", X_train.shape)
print("Training labels:", y_train.shape)

print("\nValidation images:", X_val.shape)
print("Validation labels:", y_val.shape)

print("\nTraining pixel range:", X_train.min(), "to", X_train.max())
print("Number of classes:", len(np.unique(y)))
