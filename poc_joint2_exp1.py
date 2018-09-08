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
from six import iteritems

from functools import partial
import importlib
import os
from os.path import join

import numpy as np
import matplotlib
matplotlib.use('agg')  # avoid tkinter
import matplotlib.pyplot as plt
import tensorflow as tf
import sonnet as snt
from tqdm import tqdm
from sklearn.manifold import TSNE

import common
import common_joint
import nn
import model_joint

ds = tf.contrib.distributions


class VAE(snt.AbstractModule):

  def __init__(self, config, name=''):
    super(VAE, self).__init__(name=name)
    self.config = config

  def _build(self, unused_input=None):

    config = self.config

    # Constants
    batch_size = config['batch_size']
    n_latent = config['n_latent']
    n_latent_shared = config['n_latent_shared']

    # ---------------------------------------------------------------------
    # ## Placeholders
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # ## Modules with parameters
    # ---------------------------------------------------------------------
    Encoder = config['Encoder']
    Decoder = config['Decoder']
    encoder = Encoder(name='encoder')
    decoder = Decoder(name='decoder')

    # ---------------------------------------------------------------------
    # ## One side VAE (training)
    # ---------------------------------------------------------------------
    # Reconstruction
    x = tf.placeholder(tf.float32, shape=(None, n_latent))
    mu, sigma = encoder(x)
    q_z = ds.Normal(loc=mu, scale=sigma)
    q_z_sample = q_z.sample()
    x_prime = decoder(q_z_sample)
    recons = tf.reduce_mean(tf.square(x_prime - x))
    mean_recons = tf.reduce_mean(recons)

    # Prior
    p_z = ds.Normal(loc=0., scale=1.)
    #p_z_sample = p_z.sample(sample_shape=[batch_size, n_latent_shared])
    #x_from_prior = decoder(p_z_sample)
    beta = config['prior_loss_beta']
    KL_qp = ds.kl_divergence(ds.Normal(loc=mu, scale=sigma), p_z)
    KL = tf.reduce_sum(KL_qp, axis=-1)
    mean_KL = tf.reduce_mean(KL)
    prior_loss = mean_KL

    # ---------------------------------------------------------------------
    # ## Unsupervised alignment
    # ---------------------------------------------------------------------
    x_A = tf.placeholder(tf.float32, shape=(None, n_latent))
    x_B = tf.placeholder(tf.float32, shape=(None, n_latent))
    mu_A, sigma_A = encoder(x_A)
    mu_B, sigma_B = encoder(x_B)
    q_z_sample_A = ds.Normal(loc=mu_A, scale=sigma_A).sample()
    q_z_sample_B = ds.Normal(loc=mu_B, scale=sigma_B).sample()
    random_sampling_count = config['random_sampling_count']
    random_projection_dim = n_latent_shared
    unsup_align_loss = nn.sliced_wasserstein_tfgan(
        q_z_sample_A,
        q_z_sample_B,
        random_sampling_count,
        random_projection_dim,
    )

    # ---------------------------------------------------------------------
    # ## All looses
    # ---------------------------------------------------------------------

    prior_loss_beta = tf.constant(config['prior_loss_beta'])
    scaled_prior_loss = prior_loss * prior_loss_beta
    vae_loss = mean_recons + scaled_prior_loss
    unsup_align_loss_beta = tf.constant(config['unsup_align_loss_beta'])
    scaled_unsup_align_loss = unsup_align_loss * unsup_align_loss_beta
    full_loss = vae_loss + scaled_unsup_align_loss

    # ---------------------------------------------------------------------
    # ## Training
    # ---------------------------------------------------------------------
    # Learning rates
    lr = tf.constant(config['lr'])
    vae_vars = list(encoder.get_variables())
    vae_vars.extend(decoder.get_variables())
    vae_saver = tf.train.Saver(vae_vars, max_to_keep=100)
    train_full = tf.train.AdamOptimizer(learning_rate=lr).minimize(
        full_loss, var_list=vae_vars)

    # Add all endpoints as object attributes
    for k, v in iteritems(locals()):
      self.__dict__[k] = v

  def get_summary_kv_dict(self):
    m = self
    return {
        'm.mean_recons': m.mean_recons,
        'm.prior_loss': m.prior_loss,
        'm.scaled_prior_loss': m.scaled_prior_loss,
        'm.unsup_align_loss': m.unsup_align_loss,
        'm.scaled_unsup_align_loss': m.scaled_unsup_align_loss,
        'm.full_loss': m.full_loss,
    }


class SyntheticData(object):

  @classmethod
  def sample(cls, batch_size):
    radius = np.random.uniform(size=(batch_size)) * (2 * np.pi)
    x = np.stack([np.cos(radius), np.sin(radius)], axis=-1)
    return x

  @classmethod
  def sample_A(cls, batch_size):
    x = cls.sample(batch_size)
    x[:, 1] *= +0.5
    x[:, 0] += -0.5
    x += np.random.normal(size=(batch_size, 2)) * 0.025
    attr = np.zeros(batch_size)
    return x, attr

  @classmethod
  def sample_B(cls, batch_size):
    x = cls.sample(batch_size)
    x[:, 0] *= +0.5
    x[:, 0] += +0.5
    x += np.random.normal(size=(batch_size, 2)) * 0.025
    attr = np.ones(batch_size)
    return x, attr


FLAGS = tf.flags.FLAGS


def get_dirs(sig):
  local_base_path = join(common.get_default_scratch(), 'poc_joint2_exp1')
  save_dir = join(local_base_path, 'save', sig)
  os.system('rm -rf "%s"' % save_dir)
  tf.gfile.MakeDirs(save_dir)
  sample_dir = join(local_base_path, 'sample', sig)
  os.system('rm -rf "%s"' % sample_dir)
  tf.gfile.MakeDirs(sample_dir)
  return save_dir, sample_dir


def draw_plot(xs, attrs, fpath):
  x = np.concatenate(xs)
  attr = np.concatenate(attrs)
  if x.shape[-1] > 2:
    tsne = TSNE(n_components=2, random_state=0)
    x = tsne.fit_transform(x)
  if x.shape[-1] == 1:
    x = np.stack([x, np.zeros_like(x)], axis=-1)
  fig, ax = plt.subplots()
  sctr = ax.scatter(x=x[:, 0], y=x[:, 1], c=attr, cmap='RdYlGn', marker='.')
  fig.savefig(fpath)
  plt.close(fig)


def main(unused_argv):
  del unused_argv

  save_dir, sample_dir = get_dirs('vaeonly')

  # Plot true distribution
  x_A, attr_A = SyntheticData.sample_A(100)
  x_B, attr_B = SyntheticData.sample_B(100)
  draw_plot([x_A, x_B], [attr_A, attr_B], join(sample_dir, 'true_dist.png'))

  # make model
  n_latent = 2
  n_latent_shared = 2
  layers = [8, 8, 8]
  vae_config = {
      'Encoder':
      partial(
          model_joint.EncoderLatentFull,
          input_size=n_latent,
          output_size=n_latent_shared,
          layers=layers,
      ),
      'Decoder':
      partial(
          model_joint.DecoderLatentFull,
          input_size=n_latent_shared,
          output_size=n_latent,
          layers=layers,
      ),
      'prior_loss_beta':
      0.025,
      'random_sampling_count':
      128,
      'unsup_align_loss_beta':
      0.0,
      'batch_size':
      128,
      'n_latent':
      n_latent,
      'n_latent_shared':
      n_latent_shared,
      'lr':
      0.001,
  }

  tf.reset_default_graph()
  sess = tf.Session()
  m = VAE(vae_config, name='vae')
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
    x_A, _ = SyntheticData.sample_A(batch_size)
    x_B, _ = SyntheticData.sample_B(batch_size)
    res = sess.run([m.train_full, scalar_summaries], {
        m.x: np.concatenate([x_A, x_B]),
        m.x_A: x_A,
        m.x_B: x_B,
    })
    train_writer.add_summary(res[-1], i)

    if i % 100 == 0:
      sample_batch_size = 128
      x_A, attr_A = SyntheticData.sample_A(sample_batch_size)
      x_B, attr_B = SyntheticData.sample_B(sample_batch_size)
      z_A = sess.run(m.q_z_sample, {m.x: x_A})
      z_B = sess.run(m.q_z_sample, {m.x: x_B})
      x_prime_A = sess.run(m.x_prime, {m.x: x_A})
      x_prime_B = sess.run(m.x_prime, {m.x: x_B})

      this_iter_sample_dir = join(sample_dir, '%010d' % i)
      tf.gfile.MakeDirs(this_iter_sample_dir)
      draw_plot([x_A, x_B], [attr_A, attr_B], join(this_iter_sample_dir,
                                                   'x.png'))
      draw_plot([x_prime_A, x_prime_B], [attr_A, attr_B],
                join(this_iter_sample_dir, 'x_prime.png'))
      draw_plot([z_A, z_B], [attr_A, attr_B], join(this_iter_sample_dir,
                                                   'z.png'))


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
