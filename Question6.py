from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

#Buildng the Structure
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train, y_train,
    epochs=15,
    validation_split=0.1,
    batch_size=32
)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print("Test Accuracy:", test_accuracy)

# CNNs are generally preferred over fully connected networks for image data
# because they preserve spatial structure. A fully connected network treats
# every pixel more independently after flattening, while a CNN can learn local
# patterns such as edges, corners, and simple textures, then build them into
# more complex visual features.

# In this task, the convolution layer is learning small visual patterns from the clothing images
# for eg. as edges, curves, outlines, texture-like region or whatever unique structures are on the image