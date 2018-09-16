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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
from os.path import join

import numpy as np
import tensorflow as tf

import common
import common_joint2
import common_joint2_mnist_family

ds = tf.contrib.distributions

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('load_ckpt_iter', -1, '')  # -1 for last ckpt
tf.flags.DEFINE_string('interpolate_labels', '',
                       'a `,` separated list of 0-indexed labels.')
tf.flags.DEFINE_integer('nb_images_between_labels', 1, '')
tf.flags.DEFINE_integer('random_seed', 19260817, '')


def main(unused_argv):
  """Main function."""
  del unused_argv

  dataset_A = common_joint2.load_dataset(FLAGS.config_A, FLAGS.exp_uid_A)
  dataset_B = common_joint2.load_dataset(FLAGS.config_B, FLAGS.exp_uid_B)

  sig = common_joint2_mnist_family.get_sig()
  dirs = common_joint2.get_dirs('joint2_mnist_family', sig, clear_dir=False)
  save_dir, sample_dir = dirs

  vae_config = common_joint2_mnist_family.get_vae_config()

  load_ckpt_iter = FLAGS.load_ckpt_iter
  if load_ckpt_iter == -1:
    load_ckpt_iter = FLAGS.n_iters - 1

  # Build and restore models
  helper_joint = common_joint2.JointVAEHelper(vae_config, save_dir)
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

  helper_joint.restore(load_ckpt_iter)
  helper_A.restore(dataset_A)
  helper_B.restore(dataset_B)

  # prepare intepolate dir
  evaluate_dir = join(sample_dir, 'evaluate', '%010d' % load_ckpt_iter)
  tf.gfile.MakeDirs(evaluate_dir)

  ############################################################################
  # Interploation
  ############################################################################

  np.random.seed(FLAGS.random_seed)
  interpolate_labels = [int(_) for _ in FLAGS.interpolate_labels.split(',')]
  nb_images_between_labels = FLAGS.nb_images_between_labels
  labels = interpolate_labels

  def get_x(dataset):
    index_list = []
    for label in labels:
      index_candidate = dataset.index_grouped_by_label[label]
      index_list.append(np.random.choice(index_candidate))

    x = []
    emphasize = []
    x.append(dataset.train_mu[index_list[0]])
    emphasize.append(len(x) - 1)
    for i_label in range(1, len(labels)):
      last_x = x[-1]
      this_x = dataset.train_mu[index_list[i_label]]
      for j in range(1, nb_images_between_labels + 1):
        x.append(last_x +
                 (this_x - last_x) * (float(j) / nb_images_between_labels))
      emphasize.append(len(x) - 1)
    x = np.array(x, dtype=np.float32)
    return x, emphasize

  x_A, emphasize = get_x(dataset_A)
  x_B, _ = get_x(dataset_B)
  x_A_prime = helper_joint.get_x_prime_A(x_A)
  x_B_prime = helper_joint.get_x_prime_B(x_B)
  x_A_tr = helper_joint.get_x_prime_A_from_x_B(x_B)
  x_B_tr = helper_joint.get_x_prime_B_from_x_A(x_A)

  batch_image_fn = partial(
      common.batch_image,
      max_images=len(x_A),
      rows=len(x_A),
      cols=1,
  )
  interpolate_sig = 'interpolate_il:%s:_nibl:%d:_' % (
      FLAGS.interpolate_labels, FLAGS.nb_images_between_labels)

  def save(helper, var, var_name):
    helper.save_data(
        var,
        interpolate_sig + var_name,
        evaluate_dir,
        batch_image_fn=batch_image_fn,
        emphasize=emphasize)

  save(helper_A, x_A, 'x_A')
  save(helper_A, x_A_prime, 'x_A_prime')
  save(helper_A, x_A_tr, 'x_A_tr')
  save(helper_B, x_B, 'x_B')
  save(helper_B, x_B_prime, 'x_B_prime')
  save(helper_B, x_B_tr, 'x_B_tr')


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
