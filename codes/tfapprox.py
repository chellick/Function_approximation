import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x ** 3 + 2 * np.sin(x) + 1


x_train = np.linspace(-2 * np.pi, 2 * np.pi, 1000).astype(np.float32)
y_train = f(x_train).astype(np.float32)

x_test = np.linspace(-2 * np.pi, 2 * np.pi, 300).astype(np.float32)
y_test = f(x_test).astype(np.float32)

print(x_train.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])


loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss=loss)
model.fit(x_train, y_train, epochs=300, verbose=0) # type: ignore

predictions = model(x_test)


plt.plot(x_test, y_test, 'b', label='True')
plt.plot(x_test, predictions, 'r', label='Predicted')
plt.show()