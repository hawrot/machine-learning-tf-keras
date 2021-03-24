from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

housing = fetch_california_housing() # Fetch the data

# Split the data
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


# Network
opt = keras.optimizers.Adam(learning_rate=0.01)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(10, activation="softmax"),
])



# Compile the model
model.compile(loss="mean_squared_error", optimizer=opt)

# Fit the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_valid, y_valid))

# Save the model
model.save("non-sequential-regression-model.h5")

mse_test = model.evaluate(X_test, y_test)

print(history.history)

X_new = X_test[:3]
y_pred = model.predict(X_new)
