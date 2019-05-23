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


def make_generator_model(dataset='mnist'):
  """ implements generate.

  Args:
    dataset: mnist or cifar10 dataset. (default='mnist'). choice{'mnist', 'cifar'}.

  Returns:
    model.

  """
  model = tf.keras.models.Sequential()
  model.add(layers.Dense(256, input_dim=100))
  model.add(layers.LeakyReLU(alpha=0.2))

  model.add(layers.Dense(512))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha=0.2))

  model.add(layers.Dense(1024))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha=0.2))

  if dataset == 'mnist':
    model.add(layers.Dense(28 * 28 * 1, activation='tanh'))
    model.add(layers.Reshape((28, 28, 1)))
  elif dataset == 'cifar':
    model.add(layers.Dense(32 * 32 * 3, activation='tanh'))
    model.add(layers.Reshape((32, 32, 3)))

  return model
