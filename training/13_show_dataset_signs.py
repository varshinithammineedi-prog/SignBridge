
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "dataset" / "sign_mnist_train.csv"

# Original dataset label-to-letter mapping
LABEL_TO_LETTER = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I",
    10: "K", 11: "L", 12: "M", 13: "N",
    14: "O", 15: "P", 16: "Q", 17: "R",
    18: "S", 19: "T", 20: "U", 21: "V",
    22: "W", 23: "X", 24: "Y"
}

df = pd.read_csv(CSV_PATH)

fig, axes = plt.subplots(4, 6, figsize=(12, 9))
axes = axes.flatten()

for i, label in enumerate(sorted(LABEL_TO_LETTER)):
    # Select an actual image from this class
    sample = df[df["label"] == label].iloc[0]

    pixels = sample.drop(labels=["label"]).to_numpy()
    image = pixels.reshape(28, 28)

    axes[i].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[i].set_title(
        f"Letter {LABEL_TO_LETTER[label]} (label {label})"
    )
    axes[i].axis("off")

plt.suptitle("Actual Sign Language MNIST Training Images", fontsize=16)
plt.tight_layout()
plt.show()
