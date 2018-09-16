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
"""Traing joint2 model for transfer between MNIST families."""

# pylint:disable=C0103

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join

import tensorflow as tf
from tqdm import tqdm

import common_joint2
import common_joint2_mnist_family

ds = tf.contrib.distributions

FLAGS = tf.flags.FLAGS


def main(unused_argv):
  """Main function."""
  del unused_argv

  dataset_A = common_joint2.load_dataset(FLAGS.config_A, FLAGS.exp_uid_A)
  dataset_B = common_joint2.load_dataset(FLAGS.config_B, FLAGS.exp_uid_B)

  sig = common_joint2_mnist_family.get_sig()
  dirs = common_joint2.get_dirs('joint2_mnist_family', sig, clear_dir=True)
  save_dir, sample_dir = dirs

  vae_config = common_joint2_mnist_family.get_vae_config()

  # Build the joint model.
  helper_joint = common_joint2.JointVAEHelper(vae_config, save_dir)

  # Build pre-trained models
  helper_A = common_joint2.OneSideHelper(
      FLAGS.config_A,
      FLAGS.exp_uid_A,
      FLAGS.config_classifier_A,
      FLAGS.exp_uid_classifier_A,
  )
  helper_B = common_joint2.OneSideHelper(
      FLAGS.config_B,
      FLAGS.exp_uid_B,
      FLAGS.config_classifier_B,
      FLAGS.exp_uid_classifier_B,
  )

  # Initialize and restore pre-trained models
  helper_A.restore(dataset_A)
  helper_B.restore(dataset_B)

  # Prepare data iterators.
  batch_size = vae_config['batch_size']
  n_sup = vae_config['n_sup']
  eval_batch_size = 1000  # bettert be an multiple of 10.

  unsup_iterator_A = common_joint2.DataIterator(
      dataset_A, max_n=-1, batch_size=batch_size)
  unsup_iterator_B = common_joint2.DataIterator(
      dataset_B, max_n=-1, batch_size=batch_size)
  sup_iterator_A = common_joint2.DataIterator(
      dataset_A, max_n=n_sup, batch_size=batch_size)
  sup_iterator_B = common_joint2.DataIterator(
      dataset_B, max_n=n_sup, batch_size=batch_size)
  eval_iterator_A = common_joint2.DataIterator(
      dataset_A, max_n=-1, batch_size=eval_batch_size)
  eval_iterator_B = common_joint2.DataIterator(
      dataset_B, max_n=-1, batch_size=eval_batch_size)

  # Training loop
  for i in tqdm(range(FLAGS.n_iters), desc='training', unit=' batch'):
    x_A, _ = next(unsup_iterator_A)
    x_B, _ = next(unsup_iterator_B)
    x_sup_A, label_sup_A = next(sup_iterator_A)
    x_sup_B, label_sup_B = next(sup_iterator_B)

    helper_joint.train_one_batch(i, x_A, x_B, x_sup_A, x_sup_B, label_sup_A,
                                 label_sup_B)

    # Evalution part
    is_last_batch = (i == FLAGS.n_iters - 1)
    should_save = is_last_batch or (i % FLAGS.n_iters_per_save == 0)
    should_evaluate = is_last_batch or (i % FLAGS.n_iters_per_eval == 0)

    # Save the model if instructed
    if should_save:
      helper_joint.save(i)

    # Evaluate if instructed
    if should_evaluate:
      eval_x_A, _ = next(eval_iterator_A)
      eval_x_B, _ = next(eval_iterator_B)

      eval_dir = join(sample_dir, 'transfer_eval_sample', '%010d' % i)
      tf.gfile.MakeDirs(eval_dir)

      sig = 'recons_A'
      x_A = eval_x_A
      x_prime_A = helper_joint.get_x_prime_A(x_A)
      helper_joint.compare(x_A, x_prime_A, helper_A, helper_A, eval_dir, i, sig)
      helper_A.save_data(x_A, sig + '_x_A', eval_dir)
      helper_A.save_data(x_prime_A, sig + '_x_prime_A', eval_dir)

      sig = 'recons_B'
      x_B = eval_x_B
      x_prime_B = helper_joint.get_x_prime_B(x_B)
      helper_joint.compare(x_B, x_prime_B, helper_B, helper_B, eval_dir, i, sig)
      helper_B.save_data(x_B, sig + '_x_B', eval_dir)
      helper_B.save_data(x_prime_B, sig + '_x_prime_B', eval_dir)

      sig = 'sample_joint'
      x_A, x_B = helper_joint.sample_prior(eval_batch_size)
      helper_joint.compare(x_A, x_B, helper_A, helper_B, eval_dir, i, sig)
      helper_A.save_data(x_A, sig + '_x_A', eval_dir)
      helper_B.save_data(x_B, sig + '_x_B', eval_dir)

      sig = 'transfer_A_to_B'
      x_A = eval_x_A
      x_prime_B = helper_joint.get_x_prime_B_from_x_A(x_A)
      helper_joint.compare(x_A, x_prime_B, helper_A, helper_B, eval_dir, i, sig)
      helper_A.save_data(x_A, sig + '_x_A', eval_dir)
      helper_B.save_data(x_prime_B, sig + '_x_prime_B', eval_dir)

      sig = 'transfer_B_to_A'
      x_B = eval_x_B
      x_prime_A = helper_joint.get_x_prime_A_from_x_B(x_B)
      helper_joint.compare(x_B, x_prime_A, helper_B, helper_A, eval_dir, i, sig)
      helper_B.save_data(x_B, sig + '_x_B', eval_dir)
      helper_A.save_data(x_prime_A, sig + '_x_prime_A', eval_dir)

      sig = 'sample_transfer_A_to_B'
      x_A, _ = helper_joint.sample_prior(eval_batch_size)
      x_prime_B = helper_joint.get_x_prime_B_from_x_A(x_A)
      helper_joint.compare(x_A, x_prime_B, helper_A, helper_B, eval_dir, i, sig)
      helper_A.save_data(x_A, sig + '_x_A', eval_dir)
      helper_B.save_data(x_prime_B, sig + '_x_prime_B', eval_dir)

      sig = 'sample_transfer_B_to_A'
      _, x_B = helper_joint.sample_prior(eval_batch_size)
      x_prime_A = helper_joint.get_x_prime_A_from_x_B(x_B)
      helper_joint.compare(x_B, x_prime_A, helper_B, helper_A, eval_dir, i, sig)
      helper_B.save_data(x_B, sig + '_x_B', eval_dir)
      helper_A.save_data(x_prime_A, sig + '_x_prime_A', eval_dir)


import pdb, traceback, sys, code  # pylint:disable=W0611,C0413,C0411,C0410
if __name__ == '__main__':
  try:
    tf.app.run(main)
  except Exception:  # pylint:disable=W0703
    post_mortem = True
    if post_mortem:
      type_, value_, tb = sys.exc_info()
      traceback.print_exc()
      pdb.post_mortem(tb)
    else:
      raise
