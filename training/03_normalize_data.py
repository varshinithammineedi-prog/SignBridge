
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Find the project and dataset folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"

# 2. Load the training dataset
train_df = pd.read_csv(DATASET_DIR / "sign_mnist_train.csv")

# 3. Separate the correct answers from the pixels
y_train = train_df["label"].to_numpy()
X_train = train_df.drop(columns=["label"]).to_numpy()

# 4. Convert pixel values to floating-point numbers
X_train = X_train.astype("float32")

# 5. Normalize pixel values from 0–255 to 0–1
X_train = X_train / 255.0

# 6. Display the results
print("Input shape:", X_train.shape)
print("Label shape:", y_train.shape)
print("Normalized minimum:", X_train.min())
print("Normalized maximum:", X_train.max())

print("\nFirst 5 original labels:", y_train[:5])
print("First 5 normalized pixel values:", X_train[0][:5])
