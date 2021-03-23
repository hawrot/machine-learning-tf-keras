import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print(tf.__version__)
print(keras.__version__)

# Import dataset
fashion_mnist = keras.datasets.fashion_mnist

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

#print(X_train_full.shape) # Shape check
#print(X_train_full.dtype) # Type chek

# Prepare the validation set
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0 # Divided by 255 to scale the pixels for Gradient Descent
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# Set the classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Build the model
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28])) # Convert each input image into 1D array
model.add(keras.layers.Dense(300, activation="relu")) # 300 neurons | Dense layers manage weight matrix on its own
model.add(keras.layers.Dense(100, activation="relu")) # 300 neurons | Dense layers manage weight matrix on its own
model.add(keras.layers.Dense(10, activation='softmax')) # 10 neurons - one per class | softmax

# Summary the model
model.summary()
print(model.layers)

# Get the weights and biases
weights, biases = model.layers[1].get_weights()
#print(weights)
#print(biases)

# Compile the model
optimizer = keras.optimizers.SGD(lr=0.1)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Fit the model
history = model.fit(X_train, y_train, epochs = 30, validation_data=(X_valid, y_valid))

# Plot the data

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) # set the vertical range to [0,1]
plt.show()

# Evaluate the model
model.evaluate(X_test, y_test)

# Predict

X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

y_pred = model.predict_classes(X_new)
print(y_pred)
print(np.array(class_names)[y_pred])

y_new = y_test[:3]
print(y_new)
