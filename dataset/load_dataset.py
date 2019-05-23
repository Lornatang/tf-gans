# Copyright 2019 ChangyuLiu Authors. All Rights Reserved.
#
#           Licensed under the MIT License.
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at 
#         https://opensource.org/licenses/MIT
# ==============================================================================

"""load mnist or cifar10 dataset"""

import tensorflow as tf


def load_dataset(mnist_size, mnist_batch_size, cifar_size, cifar_batch_size,):
  """ load mnist and cifar10 dataset to shuffle.

  Args:
    mnist_size: mnist dataset size.
    mnist_batch_size: every train dataset of mnist.
    cifar_size: cifar10 dataset size.
    cifar_batch_size: every train dataset of cifar10.

  Returns:
    mnist dataset, cifar10 dataset

  """
  # load mnist data
  (mnist_train_images, mnist_train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

  # load cifar10 data
  (cifar_train_images, cifar_train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()

  mnist_train_images = mnist_train_images.reshape(mnist_train_images.shape[0], 28, 28, 1).astype('float32')
  mnist_train_images = (mnist_train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

  cifar_train_images = cifar_train_images.reshape(cifar_train_images.shape[0], 32, 32, 3).astype('float32')
  cifar_train_images = (cifar_train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

  # Batch and shuffle the data
  mnist_train_dataset = tf.data.Dataset.from_tensor_slices(mnist_train_images)
  mnist_train_dataset = mnist_train_dataset.shuffle(mnist_size).batch(mnist_batch_size)

  cifar_train_dataset = tf.data.Dataset.from_tensor_slices(cifar_train_images)
  cifar_train_dataset = cifar_train_dataset.shuffle(cifar_size).batch(cifar_batch_size)

  return mnist_train_dataset, cifar_train_dataset
