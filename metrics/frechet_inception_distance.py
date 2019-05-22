# Copyright 2019 ChangyuLiu Authors. All Rights Reserved.
#
# Licensed under the MIT License,
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at 
#        https://opensource.org/licenses/MIT
# ==============================================================================

""" Calculates the Frechet Inception Distance (FID) to evalulate GANs.
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectivly.
See --help to see further details.
"""

# TODO: implements code
