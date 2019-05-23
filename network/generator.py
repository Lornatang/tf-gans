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


def make_generator_model(z):
  """ implements generate.

  Args:
    z: noise.

  Returns:
    model.

  """
  model = tf.keras.models.Sequential()
  model.add(layers.Dense(256, input_dim=z))
  model.add(layers.LeakyReLU())

  model.add(layers.Dense(512))
  # model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Dense(1024))
  # model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Dense(28 * 28 * 1, activation=tf.nn.tanh))
  model.add(layers.Reshape((28, 28, 1)))

  return model
