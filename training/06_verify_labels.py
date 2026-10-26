
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(PROJECT_DIR / "dataset" / "sign_mnist_train.csv")

print("Unique labels:")
print(sorted(df["label"].unique()))

print("\nLabel range:")
print("Minimum:", df["label"].min())
print("Maximum:", df["label"].max())
