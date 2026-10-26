

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Locate the project and dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"

# Load the CSV
train_df = pd.read_csv(DATASET_DIR / "sign_mnist_train.csv")

# Separate the label and pixel values
labels = train_df["label"].to_numpy()
pixels = train_df.drop(columns=["label"]).to_numpy()

# Select the first image
image = pixels[0]

# Convert 784 values into a 28 x 28 image
image = image.reshape(28, 28)

# Display the image
plt.imshow(image, cmap="gray")
plt.title(f"Numeric label: {labels[0]}")
plt.axis("off")
plt.show()

print("Original pixel shape:", pixels[0].shape)
print("Image shape:", image.shape)
print("Minimum pixel value:", image.min())
print("Maximum pixel value:", image.max())
