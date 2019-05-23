# Copyright 2019 ChangyuLiu Authors. All Rights Reserved.
#
#           Licensed under the MIT License.
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at 
#         https://opensource.org/licenses/MIT
# ==============================================================================

"""Generate optim loss and Discriminate optim loss"""

import tensorflow as tf

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(fake_output):
  """ The generator's loss quantifies how well it was able to trick the discriminator.
      Intuitively, if the generator is performing well, the discriminator will classify the fake images as real (or 1).
      Here, we will compare the discriminators decisions on the generated images to an array of 1s.

  Args:
    fake_output: generate pic.

  Returns:
    loss

  """
  return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output):
  """ This method quantifies how well the discriminator is able to distinguish real images from fakes.
      It compares the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions
      on fake (generated) images to an array of 0s.

  Args:
    real_output: origin pic.
    fake_output: generate pic.

  Returns:
    real loss + fake loss

  """
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss

  return total_loss


def generator_optimizer():
  """ The training generator optimizes the network.

  Returns:
    optim loss.

  """
  return tf.keras.optimizers.Adam(lr=1e-4)


def discriminator_optimizer():
  """ The training discriminator optimizes the network.

  Returns:
    optim loss.

  """
  return tf.keras.optimizers.Adam(lr=1e-4)
