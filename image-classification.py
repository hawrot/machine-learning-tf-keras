import tensorflow as tf
from tensorflow import keras

print(tf.__version__)
print(keras.__version__)

# Import dataset
fashion_mnist = keras.datasets.fashion_mnist

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Shape of X_train
print(X_train_full.shape)

