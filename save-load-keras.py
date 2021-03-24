import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import dataset
fashion_mnist = keras.datasets.fashion_mnist

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

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

# Summarise the model
model.summary()
print(model.layers)

# Compile the model
optimizer = keras.optimizers.SGD(lr=0.2)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Create the checkpoint
checkpoint_cb = keras.callbacks.ModelCheckpoint("checkpoint-keras-model.h5", save_best_only=True) # Saves the model with best accuracy

# Fit the model
history = model.fit(X_train, y_train, epochs = 50, validation_data=(X_valid, y_valid), callbacks=[checkpoint_cb])

# Load the saved model
loadedModel = keras.models.load_model("checkpoint-keras-model.h5") #roll back to best model

# Evaluate the model
loadedModel.evaluate(X_test, y_test)

# Predict
X_new = X_test[:3]
y_proba = loadedModel.predict(X_new)
y_proba.round(2)

y_pred = loadedModel.predict_classes(X_new)
print(y_pred)
print(np.array(class_names)[y_pred])

y_new = y_test[:3]
print(y_new)
