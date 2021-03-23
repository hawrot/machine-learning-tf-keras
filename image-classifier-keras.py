import tensorflow as tf
from tensorflow import keras

print(tf.__version__)
print(keras.__version__)

# Import dataset
fashion_mnist = keras.datasets.fashion_mnist

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Shape check
print(X_train_full.shape)

# Type
print(X_train_full.dtype)

