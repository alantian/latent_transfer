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
"""joint2 model (=shared VAEs).

This model model two latent space with the same shared latent space
and the shared encoder/decoder (VAE).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six import iteritems

import sonnet as snt
import tensorflow as tf

import nn

ds = tf.contrib.distributions


def affine(x, output_size, z=None, residual=False, softplus=False):
  """Make an affine layer with optional residual link and softplus activation.

  Args:
    x: An TF tensor which is the input.
    output_size: The size of output, e.g. the dimension of this affine layer.
    z: An TF tensor which is added when residual link is enabled.
    residual: A boolean indicating whether to enable residual link.
    softplus: Whether to apply softplus activation at the end.

  Returns:
    The output tensor.
  """
  if residual:
    x = snt.Linear(2 * output_size)(x)
    z = snt.Linear(output_size)(z)
    dz = x[:, :output_size]
    gates = tf.nn.sigmoid(x[:, output_size:])
    output = (1 - gates) * z + gates * dz
  else:
    output = snt.Linear(output_size)(x)

  if softplus:
    output = tf.nn.softplus(output)

  return output


class EncoderLatentFull(snt.AbstractModule):
  """An MLP (Full layers) encoder for modeling latent space."""

  def __init__(self,
               input_size,
               output_size,
               layers=(2048,) * 4,
               name='EncoderLatentFull',
               residual=True):
    super(EncoderLatentFull, self).__init__(name=name)
    self.layers = layers
    self.input_size = input_size
    self.output_size = output_size
    self.residual = residual

  def _build(self, z):  # pylint:disable=W0221
    assert isinstance(z, tuple)
    z = tf.concat(z, axis=-1)

    x = z
    for l in self.layers:
      x = tf.nn.relu(snt.Linear(l)(x))

    mu = affine(x, self.output_size, z, residual=self.residual, softplus=False)
    sigma = affine(
        x, self.output_size, z, residual=self.residual, softplus=True)
    return mu, sigma


class DecoderLatentFull(snt.AbstractModule):
  """An MLP (Full layers) decoder for modeling latent space."""

  def __init__(self,
               input_size,
               output_size,
               layers=(2048,) * 4,
               name='DecoderLatentFull',
               residual=True):
    super(DecoderLatentFull, self).__init__(name=name)
    self.layers = layers
    self.input_size = input_size
    self.output_size = output_size
    self.residual = residual

  def _build(self, z):  # pylint:disable=W0221
    assert isinstance(z, tuple)
    z = tf.concat(z, axis=-1)

    x = z
    for l in self.layers:
      x = tf.nn.relu(snt.Linear(l)(x))

    mu = affine(x, self.output_size, z, residual=self.residual, softplus=False)
    return mu


ClassifierLatentFull = DecoderLatentFull  # It can also be a classifier.


class VAE(snt.AbstractModule):
  """A shared VAE for modeling latent spaces of two domains."""

  def __init__(self, config, name=''):
    super(VAE, self).__init__(name=name)
    self.config = config

  def encode(self, encoder, x, domain):
    """Encode `x` using `encoder` with optional `domain`."""

    if self.config['use_domain']:
      # Sonnet expects tuple rather than list since tuple is not mutable.
      input_ = (x, domain)
    else:
      input_ = (x,)
    return encoder(input_)

  def decode(self, decoder, z, domain):
    """Decode `z` using `decoder` with optional `domain`."""
    if self.config['use_domain']:
      # Sonnet expects tuple rather than list since tuple is not mutable.
      input_ = (z, domain)
    else:
      input_ = (z,)
    return decoder(input_)

  def classify(self, classifier, z):
    """Classify `z` using `classifier`."""
    input_ = (z,)
    return classifier(input_)

  def _build(self, unused_input=None):  # pylint:disable=W0221

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
    x_domain = tf.placeholder(tf.float32, shape=(None, 2))
    mu, sigma = self.encode(encoder, x, x_domain)
    q_z = ds.Normal(loc=mu, scale=sigma)
    q_z_sample = q_z.sample()
    x_prime = self.decode(decoder, q_z_sample, x_domain)
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
    # ## Unsupervised alignment (training)
    # ---------------------------------------------------------------------
    x_A = tf.placeholder(tf.float32, shape=(None, n_latent))
    x_B = tf.placeholder(tf.float32, shape=(None, n_latent))
    x_A_domain = tf.placeholder(tf.float32, shape=(None, 2))
    x_B_domain = tf.placeholder(tf.float32, shape=(None, 2))
    mu_A, sigma_A = self.encode(encoder, x_A, x_A_domain)
    mu_B, sigma_B = self.encode(encoder, x_B, x_B_domain)
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
    # ## Classifier on shared latent space (training)
    # ---------------------------------------------------------------------
    x_cls = tf.placeholder(tf.float32, shape=(None, n_latent))
    x_cls_domain = tf.placeholder(tf.float32, shape=(None, 2))
    labels_cls = tf.placeholder(tf.int32, shape=(None))
    mu_cls, sigma_cls = self.encode(encoder, x_cls, x_cls_domain)
    q_z_sample_cls = ds.Normal(loc=mu_cls, scale=sigma_cls).sample()
    n_label = config['n_label']
    Classifier = config['Classifier']
    classifier = Classifier(name='classifier')
    logits_cls = self.classify(classifier, q_z_sample_cls)
    cls_loss = tf.losses.sparse_softmax_cross_entropy(labels_cls, logits_cls)
    cls_accuarcy = nn.on_the_fly_accuarcy(labels_cls, logits_cls)

    # ---------------------------------------------------------------------
    #  Domain Transfer (inferring)
    # ---------------------------------------------------------------------
    x_transfer = tf.placeholder(tf.float32, shape=(None, n_latent))
    x_transfer_encode_domain = tf.placeholder(tf.float32, shape=(None, 2))
    x_transfer_decode_domain = tf.placeholder(tf.float32, shape=(None, 2))

    x_transfer_mu, x_transfer_sigma = self.encode(encoder, x_transfer,
                                                  x_transfer_encode_domain)
    q_z_transfer = ds.Normal(loc=x_transfer_mu, scale=x_transfer_sigma)
    q_z_sample_transfer = q_z_transfer.sample()
    x_prime_transfer = self.decode(decoder, q_z_sample_transfer,
                                   x_transfer_decode_domain)

    # ---------------------------------------------------------------------
    # ## All looses
    # ---------------------------------------------------------------------

    prior_loss_beta = tf.constant(config['prior_loss_beta'])
    scaled_prior_loss = prior_loss * prior_loss_beta
    vae_loss = mean_recons + scaled_prior_loss
    unsup_align_loss_beta = tf.constant(config['unsup_align_loss_beta'])
    scaled_unsup_align_loss = unsup_align_loss * unsup_align_loss_beta
    cls_loss_beta = tf.constant(config['cls_loss_beta'])
    scaled_cls_loss = cls_loss * cls_loss_beta
    full_loss = vae_loss + scaled_unsup_align_loss + scaled_cls_loss

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
    """Get summary as a dict with name as key and endpoint as value."""
    m = self
    return {
        'm.mean_recons': m.mean_recons,
        'm.prior_loss': m.prior_loss,
        'm.scaled_prior_loss': m.scaled_prior_loss,
        'm.unsup_align_loss': m.unsup_align_loss,
        'm.scaled_unsup_align_loss': m.scaled_unsup_align_loss,
        'm.cls_loss': m.cls_loss,
        'm.scaled_cls_loss': m.scaled_cls_loss,
        'm.cls_accuarcy': m.cls_accuarcy,
        'm.full_loss': m.full_loss,
    }
