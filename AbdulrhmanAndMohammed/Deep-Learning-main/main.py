# Deep Learning Model: Handwritten Digit Recognition (MNIST)

from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Preprocess the data
x_train = x_train.reshape((60000, 28, 28, 1)).astype("float32") / 255
x_test = x_test.reshape((10000, 28, 28, 1)).astype("float32") / 255

y_train = to_categorical(y_train, 10)  # one-hot encode labels
y_test = to_categorical(y_test, 10)

# 3. Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")  # 10 classes for digits
])

# 4. Compile the model
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# 5. Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# 6. Evaluate on test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose="auto")
print(f"Test Accuracy: {test_acc*100:.2f}%")

# 7. Make a prediction (example: first test image)
pred = model.predict(x_test[0:1])

print("Predicted digit:", np.argmax(pred))

# Reshape is necessary to remove the extra channel dimension (from shape (28,28,1) to (28,28)) for proper display with imshow
plt.imshow(x_test[0].reshape(28,28), cmap="gray")
plt.title(f"Prediction: {np.argmax(pred)}")
plt.savefig("predicted_digit.png")  # Save the image to a file
plt.show()
