# Copyright 2019 ChangyuLiu Authors. All Rights Reserved.
#
#           Licensed under the MIT License.
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at 
#         https://opensource.org/licenses/MIT
# ==============================================================================

"""model of discriminator"""

import tensorflow as tf
from tensorflow.python.keras import layers


def make_discriminator_model():
  """ implements discriminate.

  Returns:
    model.

  """
  model = tf.keras.models.Sequential()
  model.add(layers.Flatten(input_shape=[28, 28, 1]))

  model.add(layers.Dense(1024))
  model.add(layers.LeakyReLU(alpha=0.2))

  model.add(layers.Dense(512))
  model.add(layers.LeakyReLU(alpha=0.2))

  model.add(layers.Dense(256))
  model.add(layers.LeakyReLU(alpha=0.2))

  model.add(layers.Dense(1, activation='sigmoid'))

  return model
