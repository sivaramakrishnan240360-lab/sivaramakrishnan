import tensorflow as tf
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD MNIST DATASET
# ==========================================

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

print("Training images:", x_train.shape)
print("Testing images :", x_test.shape)


# ==========================================
# 2. DATA PREPROCESSING
# ==========================================

# Convert pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add grayscale channel
x_train = x_train[..., None]
x_test = x_test[..., None]


# ==========================================
# 3. BUILD CNN MODEL
# ==========================================

model = tf.keras.Sequential([

    # First convolution layer
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    # First pooling layer
    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Second convolution layer
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    # Second pooling layer
    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Convert feature maps into vector
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Output layer - 10 digits
    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])


# ==========================================
# 4. COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 5. DISPLAY MODEL ARCHITECTURE
# ==========================================

model.summary()


# ==========================================
# 6. TRAIN THE MODEL
# ==========================================

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.1,
    verbose=1
)


# ==========================================
# 7. EVALUATE MODEL
# ==========================================

loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Test Loss     :", round(loss, 4))
print(
    "Test Accuracy :",
    round(accuracy * 100, 2),
    "%"
)


# ==========================================
# 8. PREDICT A DIGIT
# ==========================================

index = 0

prediction = model.predict(
    x_test[index:index + 1],
    verbose=0
)

predicted_digit = prediction.argmax()

print("\n==============================")
print("DIGIT PREDICTION")
print("==============================")

print("Actual Digit   :", y_test[index])
print("Predicted Digit:", predicted_digit)


# ==========================================
# 9. DISPLAY TEST IMAGE
# ==========================================

plt.figure(figsize=(5, 5))

plt.imshow(
    x_test[index].squeeze(),
    cmap="gray"
)

plt.title(
    "Predicted Digit: " + str(predicted_digit)
)

plt.axis("off")
plt.show()


# ==========================================
# 10. ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(7, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Training and Validation Accuracy"
)

plt.legend()
plt.grid(True)

plt.show()


# ==========================================
# 11. LOSS GRAPH
# ==========================================

plt.figure(figsize=(7, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training and Validation Loss"
)

plt.legend()
plt.grid(True)

plt.show()