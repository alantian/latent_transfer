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

import tensorflow as tf

import model_joint2

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('config_A', '', '')
tf.flags.DEFINE_string('exp_uid_A', '_exp_0', '')
tf.flags.DEFINE_string('config_B', '', '')
tf.flags.DEFINE_string('exp_uid_B', '_exp_1', '')
tf.flags.DEFINE_string('config_classifier_A', '', '')
tf.flags.DEFINE_string('exp_uid_classifier_A', '_exp_0', '')
tf.flags.DEFINE_string('config_classifier_B', '', '')
tf.flags.DEFINE_string('exp_uid_classifier_B', '_exp_0', '')

tf.flags.DEFINE_string('layers', '256,256,256,256', '')
tf.flags.DEFINE_string('cls_layers', '256', '')
tf.flags.DEFINE_integer('n_latent', 100, '')
tf.flags.DEFINE_integer('n_latent_shared', 2, '')
tf.flags.DEFINE_integer('n_label', 10, '')
tf.flags.DEFINE_integer('n_sup', -1, '')
tf.flags.DEFINE_float('lr', 0.001, '')
tf.flags.DEFINE_float('prior_loss_beta', 0.0, '')  # good value was 0.025
tf.flags.DEFINE_float('unsup_align_loss_beta', 0.0, '')
tf.flags.DEFINE_float('cls_loss_beta', 0.0, '')
tf.flags.DEFINE_integer('random_sampling_count', 128, '')
tf.flags.DEFINE_integer('batch_size', 512, '')
tf.flags.DEFINE_boolean('use_domain', True, '')
tf.flags.DEFINE_string('sig_extra', '', '')

tf.flags.DEFINE_integer('n_iters', 20000, '')
tf.flags.DEFINE_integer('n_iters_per_eval', 1000, '')
tf.flags.DEFINE_integer('n_iters_per_save', 1000, '')


def get_sig():
  """Get signature of this run, with parameterization resolved."""
  return 'sigv1_cA:{cA}:_cb:{cB}:_l:{l}:_cl:{cl}:-_nl{nl}_nls{nls}_ns{ns}_lr{lr}_plb{plb}_ualb{ualb}_clb{clb}_rsc{rsc}_bs{bs}_ud{ud}'.format(  # pylint:disable=C0301
      cA=FLAGS.config_A,
      cB=FLAGS.config_B,
      l=FLAGS.layers,
      cl=FLAGS.cls_layers,
      nl=FLAGS.n_latent,
      nls=FLAGS.n_latent_shared,
      ns=FLAGS.n_sup,
      lr=FLAGS.lr,
      plb=FLAGS.prior_loss_beta,
      ualb=FLAGS.unsup_align_loss_beta,
      clb=FLAGS.cls_loss_beta,
      rsc=FLAGS.random_sampling_count,
      bs=FLAGS.batch_size,
      ud=FLAGS.use_domain,
  ) + FLAGS.sig_extra


def get_vae_config():

  layers = [int(_) for _ in FLAGS.layers.strip().split(',') if _]
  cls_layers = [int(_) for _ in FLAGS.cls_layers.strip().split(',') if _]
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
      'n_sup': FLAGS.n_sup,
      'lr': FLAGS.lr,
      'use_domain': FLAGS.use_domain
  }
  return vae_config
