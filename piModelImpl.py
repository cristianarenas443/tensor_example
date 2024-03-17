#!usr/bin/env python

import tensorflow as tf
import numpy as np

new_model = tf.keras.models.load_model('modelLess1P.h5')

val = 150

result = new_model.predict(np.array([val]))

print(np.float64(val) / np.float64(result))