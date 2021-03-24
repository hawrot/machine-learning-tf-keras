from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

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


# Network
opt = keras.optimizers.SGD(learning_rate=0.01)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(10, activation="softmax"),
])

# Summarise the model
model.summary()

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Fit the model
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

# Save the model
model.save("non-sequential-regression-model.h5")

mse_test = model.evaluate(X_test, y_test)

print(history.history)

X_new = X_test[:3]
y_pred = model.predict(X_new)
