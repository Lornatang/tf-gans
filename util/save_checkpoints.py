# Copyright 2019 ChangyuLiu Authors. All Rights Reserved.
#
#           Licensed under the MIT License.
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at 
#         https://opensource.org/licenses/MIT
# ==============================================================================

"""save model function"""

import os
import tensorflow as tf


def save_checkpoints(generator, discriminator, generator_optimizer, discriminator_optimizer, save_path):
  """ save gan model

  Args:
    generator: generate model.
    discriminator: discriminate model.
    generator_optimizer: generate optimizer func.
    discriminator_optimizer: discriminator optimizer func.
    save_path: save gan model dir path.

  Returns:
    checkpoint path

  """
  checkpoint_dir = save_path
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                   discriminator_optimizer=discriminator_optimizer,
                                   generator=generator,
                                   discriminator=discriminator)

  return checkpoint_dir, checkpoint, checkpoint_prefix
