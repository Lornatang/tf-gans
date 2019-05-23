# Copyright 2019 ChangyuLiu Authors. All Rights Reserved.
#
#           Licensed under the MIT License.
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at 
#         https://opensource.org/licenses/MIT
# ==============================================================================

from dataset.load_dataset import load_dataset
from eval.cal_loss import cal_loss


train_dataset, _ = load_dataset(6000,
                                128,
                                50000,
                                128)

for image_batch in train_dataset:
  cal_loss(image_batch)
