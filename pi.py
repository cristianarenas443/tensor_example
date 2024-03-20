#!usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math as mt

input_3_items = np.array([21.4, 28, 20.9], dtype=float)
output_3_items = np.array([6.2, 8.75, 6.5], dtype=float)

input_9_items = np.array([6.28, 12.57, 18.85, 25.13, 31.42, 37.7, 43.98, 50.27, 56.55], dtype=float)
output_9_items = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18], dtype=float)

input = input_9_items
output = output_9_items

layer0 = tf.keras.layers.Dense(units=5, input_shape = [1])
layer1 = tf.keras.layers.Dense(units=10)
layer2 = tf.keras.layers.Dense(units=15)
layer3 = tf.keras.layers.Dense(units=20)
layer4 = tf.keras.layers.Dense(units=15)
layer5 = tf.keras.layers.Dense(units=10)
layer6 = tf.keras.layers.Dense(units=5)
layer_output = tf.keras.layers.Dense(units=1)

model = tf.keras.Sequential([layer0, layer1, layer2, layer3, layer4, layer5, layer6, layer_output])

model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

print('Begining trainig...')

his = model.fit(input, output, epochs=500, verbose=False)

print('Finished training')

plt.xlabel('# epoca')
plt.ylabel('Loss')
plt.plot(his.history['loss'])
#plt.show()

value = 10

result = model.predict(np.array([value]))

res = result[[0]]

resultVal = np.float64(value/res)

resultPercent = 100 * resultVal / mt.pi

print("\n ****************   Resultados  *******************")

print('predicted value >> ' + str(round(resultVal, 4)))

err = abs(round(resultPercent, 4)) - 100

print('err ' + str(abs(round(err, 4))))

print('prediction at ' + str(round(100 - err, 4)) + '%')

print("\n **************************************************")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)