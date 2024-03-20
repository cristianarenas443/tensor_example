#!usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

input = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
output = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

layer = tf.keras.layers.Dense(units=1, input_shape = [1])
model = tf.keras.Sequential([layer])

model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')
print('Begining trainig...')

his = model.fit(input, output, epochs=500, verbose=False)

print('Finished training')

plt.xlabel('# epoca')
plt.ylabel('Loss')
plt.plot(his.history['loss'])
plt.show()

result = model.predict(np.array([100]))

print('prediction >> ' + str(result))