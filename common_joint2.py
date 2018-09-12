# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Common ulitites for joint2 model."""

from collections import namedtuple
import os
from os.path import join

import tensorflow as tf
import numpy as np

import common

DirsBlob = namedtuple('DirsBlob', ['save_dir', 'sample_dir'])


def get_dirs(global_sig, sig):
  """Get needed directories for training.

  This method would create dir(s) if not existing. Also would
    remove content in dir(s) if anything exists.

  Args:
    global_sig: (string) global singature.
    sig: (string) local, or run-wise, signature.

  Returns:
    An DirsBlob instance, storing needed directories as strings.
  """

  local_base_path = join(common.get_default_scratch(), global_sig)
  save_dir = join(local_base_path, 'save', sig)
  os.system('rm -rf "%s"' % save_dir)
  tf.gfile.MakeDirs(save_dir)
  sample_dir = join(local_base_path, 'sample', sig)
  os.system('rm -rf "%s"' % sample_dir)
  tf.gfile.MakeDirs(sample_dir)
  return DirsBlob(save_dir=save_dir, sample_dir=sample_dir)


def get_domain_A(batch_size):
  """Get the one-hot domain vector for domain A."""
  arr = np.zeros(shape=(batch_size, 2), dtype=np.float32)
  arr[:, 0] = 1.0
  return arr


def get_domain_B(batch_size):
  """Get the one-hot domain vector for domain B."""
  arr = np.zeros(shape=(batch_size, 2), dtype=np.float32)
  arr[:, 1] = 1.0
  return arr
