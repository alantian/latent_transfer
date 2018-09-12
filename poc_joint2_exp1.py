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
"""POC (Proof-of-Concept) of joint2 model (=shared VAEs), experiment 1.

This scripts contains experiments merging two disjoint circles.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from functools import partial
import importlib
import os
from os.path import join

import numpy as np

import tensorflow as tf
from tqdm import tqdm
from sklearn.manifold import TSNE

import common
import common_joint2
import model_joint2

ds = tf.contrib.distributions


class SyntheticData(object):

  @classmethod
  def sample(cls, batch_size):
    radius = np.random.uniform(size=(batch_size)) * (2 * np.pi)
    x = np.stack([np.cos(radius), np.sin(radius)], axis=-1)
    return x, radius

  @classmethod
  def sample_A(cls, batch_size):
    x, radius = cls.sample(batch_size)
    x[:, 1] *= +0.5
    x[:, 0] += -0.5
    x += np.random.normal(size=(batch_size, 2)) * 0.025
    radius = radius / (2 * np.pi)  # range: [0.0, 1.0]
    attr = 0.0 + radius  # range: [0.0, 1.0]
    label = np.zeros(batch_size, dtype=np.int32)
    label[attr >= 0.5] = 1
    return x, attr, label

  @classmethod
  def sample_B(cls, batch_size):
    x, radius = cls.sample(batch_size)
    x[:, 0] *= +0.5
    x[:, 0] += +0.5
    x += np.random.normal(size=(batch_size, 2)) * 0.025

    radius = radius / (2 * np.pi)  # range: [0.0, 1.0]
    radius = radius + 0.25  # range [0.25, 1.25]
    radius = (
        (radius >= 1.0).astype(np.float32) * (radius - 1.0) +
        (radius < 1.0).astype(np.float32) * (radius))  # wrap to [0.0, 1.0]
    attr = 2.0 + radius  # range: 2.0 + [0.0, 1.0]

    label = np.zeros(batch_size, dtype=np.int32)
    label[attr < 2.0 + 0.5] = 1
    return x, attr, label

  @classmethod
  def get_domain_A(cls, batch_size):
    arr = np.zeros(shape=(batch_size, 2), dtype=np.float32)
    arr[:, 0] = 1.0
    return arr

  @classmethod
  def get_domain_B(cls, batch_size):
    arr = np.zeros(shape=(batch_size, 2), dtype=np.float32)
    arr[:, 1] = 1.0
    return arr


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('n_latent', 2, '')
tf.flags.DEFINE_integer('n_latent_shared', 2, '')
tf.flags.DEFINE_integer('n_label', 2, '')
tf.flags.DEFINE_float('lr', 0.001, '')
tf.flags.DEFINE_float('prior_loss_beta', 0.025, '')
tf.flags.DEFINE_float('unsup_align_loss_beta', 0.0, '')
tf.flags.DEFINE_float('cls_loss_beta', 0.0, '')
tf.flags.DEFINE_integer('random_sampling_count', 128, '')
tf.flags.DEFINE_integer('batch_size', 128, '')
tf.flags.DEFINE_boolean('use_domain', False, '')
tf.flags.DEFINE_string('sig_extra', '', '')


def get_sig():
  return 'nl{nl}_nls{nls}_lr{lr}_plb{plb}_ualb{ualb}_clb{clb}_rsc{rsc}_bs{bs}_ud{ud}'.format(
      nl=FLAGS.n_latent,
      nls=FLAGS.n_latent_shared,
      lr=FLAGS.lr,
      plb=FLAGS.prior_loss_beta,
      ualb=FLAGS.unsup_align_loss_beta,
      clb=FLAGS.cls_loss_beta,
      rsc=FLAGS.random_sampling_count,
      bs=FLAGS.batch_size,
      ud=FLAGS.use_domain,
  ) + FLAGS.sig_extra


PLOT_VMIN = 0.0
PLOT_VMAX = 3.0
PLOT_X_0_MIN = -1.7
PLOT_X_0_MAX = +1.2
PLOT_X_1_MIN = -1.2
PLOT_X_1_MAX = +1.2


def draw_plot(xs, attrs, labels, fpath, plot_is_x):
  import matplotlib
  # avoid tkinter. This should happen before `import matplotlib.pyplot`.
  matplotlib.use('agg')
  import matplotlib.pyplot as plt

  assert len(xs) == len(attrs)
  assert len(xs) <= 2

  marker_list = ['.', 'x']  # ['^', 'v', '+', 'x']
  # see https://matplotlib.org/api/markers_api.html#module-matplotlib.markers

  #notes = []
  #last_pos = 0
  #for i, x in enumerate(xs):
  #  notes.append((last_pos, last_pos + len(x), marker_list[i]))
  #  last_pos += len(x)

  new_notes = []
  marker_i = 0
  last_pos = 0
  for x, attr, label in zip(xs, attrs, labels):
    for target_label in [0, 1]:
      index = [i for i in range(len(label)) if label[i] == target_label]
      index = np.array(index, dtype=np.int32)
      new_notes.append((index + last_pos,
                        marker_list[marker_i % len(marker_list)]))
      marker_i += 1
    last_pos += len(x)

  x = np.concatenate(xs)
  attr = np.concatenate(attrs)
  if x.shape[-1] > 2:
    tsne = TSNE(n_components=2, random_state=0)
    x = tsne.fit_transform(x)
  if x.shape[-1] == 1:
    x = np.stack([x, np.zeros_like(x)], axis=-1)

  fig, ax = plt.subplots()
  if plot_is_x:
    ax.set_xlim((PLOT_X_0_MIN, PLOT_X_0_MAX))
    ax.set_ylim((PLOT_X_1_MIN, PLOT_X_1_MAX))

  # for note in notes:
  for note in new_notes:
    # start, end, marker = note
    index, marker = note
    ax.scatter(
        x=x[index, 0],
        y=x[index, 1],
        c=attr[index],
        cmap='RdYlGn',
        marker=marker,
        norm=matplotlib.colors.Normalize(vmin=0., vmax=3.),
        alpha=0.6,
    )
  fig.savefig(fpath)
  plt.close(fig)


def main(unused_argv):
  del unused_argv

  sig = get_sig()
  dirs = common_joint2.get_dirs('poc_joint2_exp1', sig)
  save_dir, sample_dir = dirs.save_dir, dirs.sample_dir

  # Plot true distribution
  x_A, attr_A, label_A = SyntheticData.sample_A(100)
  x_B, attr_B, label_B = SyntheticData.sample_B(100)
  draw_plot(
      [x_A, x_B], [attr_A, attr_B], [label_A, label_B],
      join(sample_dir, 'true_dist.png'),
      plot_is_x=True)

  # make model
  layers = [8, 8, 8]
  Encoder = partial(
      model_joint2.EncoderLatentFull,
      input_size=FLAGS.n_latent,
      output_size=FLAGS.n_latent_shared,
      layers=layers,
  )
  Decoder = partial(
      model_joint2.DecoderLatentFull,
      input_size=FLAGS.n_latent_shared,
      output_size=FLAGS.n_latent,
      layers=layers,
  )
  cls_layers = [8]
  Classifier = partial(
      model_joint2.ClassifierLatentFull,
      input_size=FLAGS.n_latent_shared,
      output_size=FLAGS.n_label,
      layers=cls_layers,
  )
  vae_config = {
      'Encoder': Encoder,
      'Decoder': Decoder,
      'Classifier': Classifier,
      'prior_loss_beta': FLAGS.prior_loss_beta,
      'random_sampling_count': FLAGS.random_sampling_count,
      'unsup_align_loss_beta': FLAGS.unsup_align_loss_beta,
      'cls_loss_beta': FLAGS.cls_loss_beta,
      'batch_size': FLAGS.batch_size,
      'n_latent': FLAGS.n_latent,
      'n_latent_shared': FLAGS.n_latent_shared,
      'n_label': FLAGS.n_label,
      'lr': FLAGS.lr,
      'use_domain': FLAGS.use_domain
  }

  tf.reset_default_graph()
  sess = tf.Session()
  m = model_joint2.VAE(vae_config, name='vae')
  m()

  train_writer = tf.summary.FileWriter(save_dir + '/transfer_train', sess.graph)
  scalar_summaries = tf.summary.merge([
      tf.summary.scalar(key, value)
      for key, value in m.get_summary_kv_dict().items()
  ])

  sess.run(tf.global_variables_initializer())

  # training loop
  n_iters = 5000
  batch_size = vae_config['batch_size']
  for i in tqdm(range(n_iters), desc='training', unit=' batch'):
    x_A, _, label_A = SyntheticData.sample_A(batch_size)
    x_B, _, label_B = SyntheticData.sample_B(batch_size)
    x_A_domain = SyntheticData.get_domain_A(batch_size)
    x_B_domain = SyntheticData.get_domain_B(batch_size)
    res = sess.run(
        [m.train_full, scalar_summaries], {
            m.x: np.concatenate([x_A, x_B]),
            m.x_domain: np.concatenate([x_A_domain, x_B_domain]),
            m.x_A: x_A,
            m.x_B: x_B,
            m.x_A_domain: x_A_domain,
            m.x_B_domain: x_B_domain,
            m.x_cls: np.concatenate([x_A, x_B]),
            m.x_cls_domain: np.concatenate([x_A_domain, x_B_domain]),
            m.labels_cls: np.concatenate([label_A, label_B]),
        })
    train_writer.add_summary(res[-1], i)

    if i % 100 == 0:
      sample_batch_size = 128
      x_A, attr_A, label_A = SyntheticData.sample_A(sample_batch_size)
      x_B, attr_B, label_B = SyntheticData.sample_B(sample_batch_size)
      x_A_domain = SyntheticData.get_domain_A(batch_size)
      x_B_domain = SyntheticData.get_domain_B(batch_size)
      z_A = sess.run(m.q_z_sample, {m.x: x_A, m.x_domain: x_A_domain})
      z_B = sess.run(m.q_z_sample, {m.x: x_B, m.x_domain: x_B_domain})
      x_prime_A = sess.run(m.x_prime, {m.x: x_A, m.x_domain: x_A_domain})
      x_prime_B = sess.run(m.x_prime, {m.x: x_B, m.x_domain: x_B_domain})
      x_prime_A_to_B = sess.run(
          m.x_prime_transfer, {
              m.x_transfer: x_A,
              m.x_transfer_encode_domain: x_A_domain,
              m.x_transfer_decode_domain: x_B_domain,
          })
      x_prime_B_to_A = sess.run(
          m.x_prime_transfer, {
              m.x_transfer: x_B,
              m.x_transfer_encode_domain: x_B_domain,
              m.x_transfer_decode_domain: x_A_domain,
          })

      this_iter_sample_dir = join(sample_dir, '%010d' % i)
      tf.gfile.MakeDirs(this_iter_sample_dir)
      draw_plot(
          [x_A, x_B], [attr_A, attr_B], [label_A, label_B],
          join(this_iter_sample_dir, 'x.png'),
          plot_is_x=True)
      draw_plot(
          [x_prime_A, x_prime_B], [attr_A, attr_B], [label_A, label_B],
          join(this_iter_sample_dir, 'x_prime.png'),
          plot_is_x=True)
      draw_plot(
          [x_prime_A_to_B, x_prime_B], [attr_A, attr_B], [label_A, label_B],
          join(this_iter_sample_dir, 'x_A_to_B_and_x_B.png'),
          plot_is_x=True)
      draw_plot(
          [x_prime_A, x_prime_B_to_A], [attr_A, attr_B], [label_A, label_B],
          join(this_iter_sample_dir, 'x_B_to_A_and_x_A.png'),
          plot_is_x=True)
      draw_plot(
          [z_A, z_B], [attr_A, attr_B], [label_A, label_B],
          join(this_iter_sample_dir, 'z.png'),
          plot_is_x=False)


# pylint:disable=all
import pdb, traceback, sys, code

if __name__ == '__main__':
  try:
    tf.app.run(main)
  except Exception:
    post_mortem = True
    if post_mortem:
      type, value, tb = sys.exc_info()
      traceback.print_exc()
      pdb.post_mortem(tb)
    else:
      raise
