import tensorflow as tf
from tensorflow import keras

print(tf.__version__)
print(keras.__version__)

# Import dataset
fashion_mnist = keras.datasets.fashion_mnist

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

#print(X_train_full.shape) # Shape check
#print(X_train_full.dtype) # Type chek

# Prepare the validation set
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0 # Divided by 255 to scale the pixels for Gradient Descent
y_valid, y_train = y_train_full[:5000] / 255.0, y_train_full[5000:] / 255.0

# Set the classes
class_names = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandals', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Build the model
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28])) # Convert each input image into 1D array
model.add(keras.layers.Dense(300, activation="relu")) # 300 neurons | Dense layers manage weight matrix on its own
model.add(keras.layers.Dense(100, activation="relu")) # 300 neurons | Dense layers manage weight matrix on its own
model.add(keras.layers.Dense(10, activation='softmax')) # 10 neurons - one per class | softmax

# Summary the model
model.summary()
