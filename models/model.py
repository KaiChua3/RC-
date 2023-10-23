import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import PIL
import tensorflow as tf
import pathlib

from tensorflow import keras
from keras import layers
from keras.models import Sequential
from PIL import Image

RECYCLED_TRAIN_DATA = 'C:/Users/chuaz/OneDrive/Desktop/Coding/congressionalappcompetition/data/recycled_32/recycled_32_train.npz'
RECYCLED_TEST_DATA = 'C:/Users/chuaz/OneDrive/Desktop/Coding/congressionalappcompetition/data/recycled_32/recycled_32_test.npz'

def unpickle(file):
  with open(file, 'rb') as fo:
      data = np.load(file)
  x, y = data['x'], data['y']
  return x, y
(x_train, y_train) = unpickle(RECYCLED_TRAIN_DATA)
(x_test, y_test) = unpickle(RECYCLED_TEST_DATA)
x_train, x_test = x_train / 255.0, x_test / 255.0
train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train))
test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test))
train_ds = train_ds.shuffle(10000).batch(32)
test_ds = test_ds.shuffle(1500).batch(32)
model = keras.models.Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  tf.keras.layers.Flatten(input_shape=(3, 32, 32)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])
model.fit(train_ds, validation_data= test_ds, epochs=15)
model.summary()
#Example code for training model (using mnist dataset from tensorflow)
#mnist = tf.keras.datasets.mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0
#model = tf.keras.models.Sequential([
#  tf.keras.layers.Flatten(input_shape=(28, 28)),
#  tf.keras.layers.Dense(128, activation='relu'),
#  tf.keras.layers.Dropout(0.2),
#  tf.keras.layers.Dense(10)
#])
#predictions = model(x_train[:1]).numpy()
#predictions
#loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#loss_fn(y_train[:1], predictions).numpy()
#model.compile(optimizer='adam',
#              loss=loss_fn,
#              metrics=['accuracy'])
#model.fit(x_train, y_train, epochs=5)
#model.evaluate(x_test,  y_test, verbose=2)