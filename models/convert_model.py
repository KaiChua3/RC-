import numpy as np
import tensorflow as tf
import keras

model = keras.models.load_model("recycling_model.keras")
model.summary()
"""
import tensorflowjs as tfjs
from tensorflowjs import converters
converters.save_keras_model(model, '')
"""