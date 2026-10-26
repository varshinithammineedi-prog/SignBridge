
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Locate the dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"

# 2. Load the training data
train_df = pd.read_csv(DATASET_DIR / "sign_mnist_train.csv")

# 3. Separate images and their correct labels
y_train = train_df["label"].to_numpy()
X_train = train_df.drop(columns=["label"]).to_numpy()

# 4. Normalize the pixels
X_train = X_train.astype("float32") / 255.0

# 5. Reshape each image into 28 x 28 with 1 channel
X_train = X_train.reshape(-1, 28, 28, 1)

# 6. Display the shapes
print("CNN input shape:", X_train.shape)
print("Labels shape:", y_train.shape)
print("One image shape:", X_train[0].shape)
