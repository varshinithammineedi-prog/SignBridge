
import pandas as pd
from pathlib import Path

# Locate and load the training dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"

train_df = pd.read_csv(DATASET_DIR / "sign_mnist_train.csv")

# Get the unique labels present in the dataset
labels = sorted(train_df["label"].unique())

# Map each numeric label to its letter


LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"

def label_to_letter(label):
    return LETTERS[label]


print("Total training images:", len(train_df))
print("Number of classes:", len(labels))
print("\nAvailable classes:")

for label in labels:
    count = (train_df["label"] == label).sum()
    letter = label_to_letter(label)
    print(f"Label {label:2} -> Letter {letter} | Images: {count}")
