import tensorflow as tf
from tensorflow import keras

print(tf.__version__)
print(keras.__version__)

# Import dataset
fashion_mnist = keras.datasets.fashion_mnist

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Shape check
#print(X_train_full.shape)

# Type
#print(X_train_full.dtype)

# Prepare the validation set
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0 # Divided by 255 to scale the pixels for Gradient Descent
y_valid, y_train = y_train_full[:5000] / 255.0, y_train_full[5000:] / 255.0


# Set the classes

class_names = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandals', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

