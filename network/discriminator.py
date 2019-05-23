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


def make_discriminator_model(dataset='mnist'):
  """ implements discriminate.

  Args:
    dataset: mnist or cifar10 dataset. (default='mnist'). choice{'mnist', 'cifar'}.

  Returns:
    model.

  """
  model = tf.keras.models.Sequential()
  if dataset == 'mnist':
    model.add(layers.Flatten(input_shape=[28, 28, 1]))
  elif dataset == 'cifar':
    model.add(layers.Flatten(input_shape=[32, 32, 3]))

  model.add(layers.Dense(1024))
  model.add(layers.LeakyReLU(alpha=0.2))

  model.add(layers.Dense(512))
  model.add(layers.LeakyReLU(alpha=0.2))

  model.add(layers.Dense(256))
  model.add(layers.LeakyReLU(alpha=0.2))

  model.add(layers.Dense(1, activation='sigmoid'))

  return model
