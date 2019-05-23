# Copyright 2019 ChangyuLiu Authors. All Rights Reserved.
#
#           Licensed under the MIT License.
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at 
#         https://opensource.org/licenses/MIT
# ==============================================================================

"""model of generate"""

import tensorflow as tf
from tensorflow.python.keras import layers

DIM = 100


def make_generator_model():
  """ implements generate.

  Returns:
    model.

  """
  model = tf.python.keras.models.Sequential()
  model.add(
    layers.Dense(256, input_dim=DIM, kernel_initializer=tf.python.keras.initializers.RandomNormal(stddev=0.02)))
  model.add(layers.LeakyReLU(0.2))

  model.add(layers.Dense(512))
  model.add(layers.LeakyReLU(0.2))

  model.add(layers.Dense(1024))
  model.add(layers.LeakyReLU(0.2))

  model.add(layers.Dense(784, activation='tanh'))

  return model
