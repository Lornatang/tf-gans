# Copyright 2019 ChangyuLiu Authors. All Rights Reserved.
#
#           Licensed under the MIT License.
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at 
#         https://opensource.org/licenses/MIT
# ==============================================================================

"""implements basic function"""

from matplotlib import pyplot as plt


def generate_and_save_images(model, epoch, seed, save_path):
  """make sure the training parameter is set to False because we
     don't want to train the batch-norm layer when doing inference.

  Args:
    model: generate model.
    epoch: train epoch nums.
    seed: random seed at (16, 100).
    save_path: generate images path.

  Returns:
    none.

  """
  predictions = model(seed, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.axis('off')

  plt.savefig(save_path + '/' + f'{epoch:04d}.png')
  # plt.show()
  plt.close(fig)
