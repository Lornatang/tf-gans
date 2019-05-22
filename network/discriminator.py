# Copyright 2019 ChangyuLiu Authors. All Rights Reserved.
#
#           Licensed under the MIT License.
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at 
#         https://opensource.org/licenses/MIT
# ==============================================================================

"""model of discriminator"""

import tensorflow as tf
from tensorflow.contrib.keras import layers

DIM = 100


def generator():
  """ implements discriminate.

  Returns:
    model.

  """
  model = tf.contrib.keras.models.Sequential()
  model.add(
    layers.Dense(1024, input_dim=784, kernel_initializer=tf.contrib.keras.initializers.RandomNormal(stddev=0.02)))
  model.add(layers.LeakyReLU(0.2))
  model.add(layers.Dropout(0.3))

  model.add(layers.Dense(512))
  model.add(layers.LeakyReLU(0.2))
  model.add(layers.Dropout(0.3))

  model.add(layers.Dense(256))
  model.add(layers.LeakyReLU(0.2))
  model.add(layers.Dropout(0.3))

  model.add(layers.Dense(1, activation='sigmoid'))
