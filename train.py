# Copyright 2019 ChangyuLiu Authors. All Rights Reserved.
#
#           Licensed under the MIT License.
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#         https://opensource.org/licenses/MIT
# ==============================================================================

"""The training loop begins with generator receiving a random seed as input.
   That seed is used to produce an image.
   The discriminator is then used to classify real images (drawn from the training set)
   and fakes images (produced by the generator).
   The loss is calculated for each of these models,
   and the gradients are used to update the generator and discriminator.
"""

from dataset.load_dataset import load_dataset
from network.generator import make_generator_model
from network.discriminator import make_discriminator_model
from util.loss_and_optim import generator_loss, generator_optimizer
from util.loss_and_optim import discriminator_loss, discriminator_optimizer
from util.save_checkpoints import save_checkpoints
from util.generate_and_save_images import generate_and_save_images

import tensorflow as tf
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', type=str,
                    help='use dataset {mnist or cifar}.')
parser.add_argument('--epochs', default=50, type=int,
                    help='Epochs for training.')
args = parser.parse_args()
print(args)

# define model save path
save_path = 'training_checkpoint'

# create dir
if not os.path.exists(save_path):
  os.makedirs(save_path)

# define random noise
noise = tf.random.normal([16, 100])

# load dataset
mnist_train_dataset, cifar_train_dataset = load_dataset(60000, 128, 50000, 64)

# load network and optim paras
generator = make_generator_model(args.dataset)
generator_optimizer = generator_optimizer()

discriminator = make_discriminator_model(args.dataset)
discriminator_optimizer = discriminator_optimizer()

checkpoint_dir, checkpoint, checkpoint_prefix = save_checkpoints(generator,
                                                                 discriminator,
                                                                 generator_optimizer,
                                                                 discriminator_optimizer,
                                                                 save_path)


# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
  """ break it down into training steps.

  Args:
    images: input images.

  """
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training=True)

    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)

  gradients_of_generator = gen_tape.gradient(gen_loss,
                                             generator.trainable_variables)
  gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                  discriminator.trainable_variables)

  generator_optimizer.apply_gradients(
    zip(gradients_of_generator, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(
    zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
  """ train op

  Args:
    dataset: mnist dataset or cifar10 dataset.
    epochs: number of iterative training.

  """
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as we go
    generate_and_save_images(generator,
                             epoch + 1,
                             noise,
                             save_path)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

    print(f'Time for epoch {epoch+1} is {time.time()-start:.3f} sec.')

  # Generate after the final epoch
  generate_and_save_images(generator,
                           epochs,
                           noise,
                           save_path)


if __name__ == '__main__':
  if args.dataset == 'mnist':
    train(mnist_train_dataset, args.epochs)
  else:
    train(cifar_train_dataset, args.epochs)
