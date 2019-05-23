# Copyright 2019 ChangyuLiu Authors. All Rights Reserved.
#
#           Licensed under the MIT License.
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at 
#         https://opensource.org/licenses/MIT
# ==============================================================================
"""cal Generate loss and Discriminate loss"""

from dataset.load_dataset import load_dataset
from network.generator import make_generator_model
from network.discriminator import make_discriminator_model
from util.loss_and_optim import generator_loss, generator_optimizer
from util.loss_and_optim import discriminator_loss, discriminator_optimizer
from util.save_checkpoints import save_checkpoints

import tensorflow as tf


# define paras
save_path = 'training_checkpoint'

dataset = 'mnist'

# define random noise
noise = tf.random.normal([16, 100])

# load dataset
train_dataset, _ = load_dataset(60000,
                                128,
                                50000,
                                64)

# load network and optim paras
generator = make_generator_model(dataset)
generator_optimizer = generator_optimizer()

discriminator = make_discriminator_model(dataset)
discriminator_optimizer = discriminator_optimizer()

checkpoint_dir, checkpoint, checkpoint_prefix = save_checkpoints(generator,
                                                                 discriminator,
                                                                 generator_optimizer,
                                                                 discriminator_optimizer,
                                                                 save_path)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def cal_loss(images):
  """ break it down into training steps.

  Args:
    images: input images.

  """
  generated_images = generator(noise, training=False)

  real_output = discriminator(images, training=False)
  fake_output = discriminator(generated_images, training=False)

  gen_loss = generator_loss(fake_output)
  disc_loss = discriminator_loss(real_output, fake_output)

  print(f'Loss G {gen_loss:.6f} Loss D {disc_loss:.6f}.')
