
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Step 1: Create the CNN architecture
model = keras.Sequential([
    # Input: 28 x 28 grayscale image
    layers.Input(shape=(28, 28, 1)),

    # Step 2: Detect simple features
    layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu"
    ),

    # Step 3: Reduce image dimensions
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Step 4: Detect more complex features
    layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    ),

    # Step 5: Reduce dimensions again
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Step 6: Convert feature maps into a vector
    layers.Flatten(),

    # Step 7: Learn combinations of features
    layers.Dense(128, activation="relu"),

    # Step 8: Produce scores for 24 numeric labels
    layers.Dense(24, activation="softmax")
])

# Step 9: Display the model architecture
model.summary()
