
import pandas as pd
from pathlib import Path

# Find our project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Find the dataset folder
DATASET_DIR = PROJECT_DIR / "dataset"

# Load the training CSV file
train_df = pd.read_csv(DATASET_DIR / "sign_mnist_train.csv")

# Display the dataset information
print("Dataset shape:", train_df.shape)
print("\nFirst five rows:")
print(train_df.head())

print("\nColumn names:")
print(train_df.columns[:10].tolist())
