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

from collections import namedtuple
from functools import partial
from os.path import join

import numpy as np

import tensorflow as tf
from tqdm import tqdm

import common
import common_joint2
import model_dataspace
import model_joint2

ds = tf.contrib.distributions

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
tf.flags.DEFINE_float('prior_loss_beta', 0.025, '')
tf.flags.DEFINE_float('unsup_align_loss_beta', 0.0, '')
tf.flags.DEFINE_float('cls_loss_beta', 0.0, '')
tf.flags.DEFINE_integer('random_sampling_count', 128, '')
tf.flags.DEFINE_integer('batch_size', 512, '')
tf.flags.DEFINE_boolean('use_domain', True, '')
tf.flags.DEFINE_string('sig_extra', '', '')

tf.flags.DEFINE_integer('n_iters', 10000, '')


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


DatasetBlob = namedtuple('DatasetBlob', [
    'train_data', 'train_label', 'train_mu', 'train_sigma',
    'index_grouped_by_label', 'save_path'
])


def load_dataset(config_name, exp_uid):
  """Load a dataset from a config's name.

  The loaded dataset consists of:
    - original data (dataset, train_data, train_label),
    - encoded data from a pretrained model (train_mu, train_sigma), and
    - index grouped by label (index_grouped_by_label).
    - path of saving (save_path) for restoring pre-trained models.

  Args:
    config_name: A string indicating the name of config to parameterize the
        model that associates with the dataset.
    exp_uid: A string representing the unique id of experiment to be used in
        model that associates with the dataset.

  Returns:
    A DatasetBlob of abovementioned components in the dataset.
  """

  config = common.load_config(config_name)

  model_uid = common.get_model_uid(config_name, exp_uid)

  dataset = common.load_dataset(config)
  train_data = dataset.train_data
  attr_train = dataset.attr_train
  save_path = dataset.save_path
  path_train = join(dataset.basepath, 'encoded', model_uid,
                    'encoded_train_data.npz')
  train = np.load(path_train)
  train_mu = train['mu']
  train_sigma = train['sigma']
  train_label = np.argmax(attr_train, axis=-1)  # from one-hot to label
  index_grouped_by_label = common.get_index_grouped_by_label(train_label)

  tf.logging.info('index_grouped_by_label size: %s',
                  [len(_) for _ in index_grouped_by_label])

  tf.logging.info('train loaded from %s', path_train)
  tf.logging.info('train shapes: mu = %s, sigma = %s', train_mu.shape,
                  train_sigma.shape)
  return DatasetBlob(
      train_data=train_data,
      train_label=train_label,
      train_mu=train_mu,
      train_sigma=train_sigma,
      index_grouped_by_label=index_grouped_by_label,
      save_path=save_path,
  )


def run_with_batch(sess, op_target, op_feed, arr_feed, batch_size=None):
  """Run a tensorflow op with batchting."""
  if batch_size is None:
    batch_size = len(arr_feed)
  return np.concatenate([
      sess.run(op_target, {op_feed: arr_feed[i:i + batch_size]})
      for i in range(0, len(arr_feed), batch_size)
  ])


def restore_model(saver, config_name, exp_uid, sess, save_path,
                  ckpt_filename_template):
  """Restore a pre-trained model."""
  model_uid = common.get_model_uid(config_name, exp_uid)
  saver.restore(
      sess,
      join(save_path, model_uid, 'best', ckpt_filename_template % model_uid))


class ModelHelper(object):
  """A Helper that provides sampling and classification for pre-trained model.

  This generic helper is for VAE model we trained as dataspace model.
  """
  DEFAULT_BATCH_SIZE = 100

  def __init__(self, config_name, exp_uid):
    self.config_name = config_name
    self.exp_uid = exp_uid

    self.build()

  def build(self):
    """Build the TF graph and heads for dataspace model.

    It also prepares different graph, session and heads for sampling and
    classification respectively.
    """

    config_name = self.config_name
    config = common.load_config(config_name)
    exp_uid = self.exp_uid

    graph = tf.Graph()
    with graph.as_default():
      sess = tf.Session(graph=graph)

      config = common.load_config(config_name)
      model_uid = common.get_model_uid(config_name, exp_uid)
      m = model_dataspace.Model(config, name=model_uid)
      m()

    self.config = config
    self.graph = graph
    self.sess = sess
    self.m = m

  def restore_best(self, saver_name, save_path, ckpt_filename_template):
    """Restore the weights of best pre-trained models."""
    config_name = self.config_name
    exp_uid = self.exp_uid
    sess = self.sess
    saver = getattr(self.m, saver_name)
    model_uid = common.get_model_uid(config_name, exp_uid)
    ckpt_path = join(save_path, model_uid, 'best',
                     ckpt_filename_template % model_uid)
    saver.restore(sess, ckpt_path)

  def decode(self, z, batch_size=None):
    """Decode from given latant space vectors `z`.

    Args:
      z: A numpy array of latent space vectors.
      batch_size: (Optional) a integer to indication batch size for computation
          which is useful if the sampling requires lots of GPU memory.

    Returns:
      A numpy array, the dataspace points from decoding.
    """
    m = self.m
    batch_size = batch_size or self.DEFAULT_BATCH_SIZE
    return run_with_batch(self.sess, m.x_mean, m.z, z, batch_size)

  def classify(self, real_x, batch_size=None):
    """Classify given dataspace points `real_x`.

    Args:
      real_x: A numpy array of dataspace points.
      batch_size: (Optional) a integer to indication batch size for computation
          which is useful if the classification requires lots of GPU memory.

    Returns:
      A numpy array, the prediction from classifier.
    """
    m = self.m
    op_target = m.pred_classifier
    op_feed = m.x
    arr_feed = real_x
    batch_size = batch_size or self.DEFAULT_BATCH_SIZE
    pred = run_with_batch(self.sess, op_target, op_feed, arr_feed, batch_size)
    pred = np.argmax(pred, axis=-1)
    return pred

  def save_data(self, x, name, save_dir, x_is_real_x=False):
    """Save dataspace instances.

    Args:
      x: A numpy array of dataspace points.
      name: A string indicating the name in the saved file.
      save_dir: A string indicating the directory to put the saved file.
      x_is_real_x: An boolean indicating whether `x` is already in dataspace. If
          not, `x` is converted to dataspace before saving
    """
    if not x_is_real_x:
      np.savetxt(join(save_dir, '%s.x_array.txt' % name), x)
    real_x = x if x_is_real_x else self.decode(x)
    real_x = common.post_proc(real_x, self.config)
    batched_real_x = common.batch_image(real_x)
    sample_file = join(save_dir, '%s.png' % name)
    common.save_image(batched_real_x, sample_file)


class OneSideHelper(object):
  """The helper that manages model and classifier in dataspace for joint model.

  Attributes:
    config_name: A string representing the name of config for model in
        dataspace.
    exp_uid: A string representing the unique id of experiment used in
        the model in dataspace.
    config_name_classifier: A string representing the name of config for
        clasisifer in dataspace.
    exp_uid_classifier: A string representing the unique id of experiment used
        in the clasisifer in dataspace.
  """

  def __init__(
      self,
      config_name,
      exp_uid,
      config_name_classifier,
      exp_uid_classifier,
  ):
    config = common.load_config(config_name)
    # In this case two diffent objects serve two purpose.
    m_helper = ModelHelper(config_name, exp_uid)
    m_classifier_helper = ModelHelper(config_name_classifier,
                                      exp_uid_classifier)

    self.config_name = config_name
    self.config = config
    self.m_helper = m_helper
    self.m_classifier_helper = m_classifier_helper

  def restore(self, dataset):
    """Restore the pretrained model and classifier.

    Args:
      dataset: The object containts `save_path` used for restoring.
    """
    m_helper = self.m_helper
    m_classifier_helper = self.m_classifier_helper

    m_helper.restore_best('vae_saver', dataset.save_path, 'vae_best_%s.ckpt')
    m_classifier_helper.restore_best('classifier_saver', dataset.save_path,
                                     'classifier_best_%s.ckpt')


class GuassianDataHelper(object):
  """A helper to hold data where each instance is a sampled point.

  Args:
    mu: Mean of data points.
    sigma: Variance of data points. If it is None, it is treated as zeros.
    batch_size: An integer indictating size of batch.
  """

  def __init__(self, mu, sigma=None):
    if sigma is None:
      sigma = np.zeros_like(mu)
    assert mu.shape == sigma.shape
    self.mu, self.sigma = mu, sigma

  def pick_batch(self, batch_index):
    """Pick a batch where instances are sampled from Guassian distributions."""
    mu, sigma = self.mu, self.sigma
    batch_mu, batch_sigma = self._np_index_arrs(batch_index, mu, sigma)
    batch = self._np_sample_from_gaussian(batch_mu, batch_sigma)
    return batch

  def __len__(self):
    return len(self.mu)

  @staticmethod
  def _np_sample_from_gaussian(mu, sigma):
    """Sampling from Guassian distribtuion specified by `mu` and `sigma`."""
    assert mu.shape == sigma.shape
    return mu + sigma * np.random.randn(*sigma.shape)

  @staticmethod
  def _np_index_arrs(index, *args):
    """Index arrays with the same given `index`."""
    return (arr[index] for arr in args)


class DataIterator(object):
  """Iterator for data."""

  def __init__(self, dataset, max_n, batch_size):
    mu = dataset.train_mu
    sigma = dataset.train_sigma
    self.guassian_data_helper = GuassianDataHelper(mu, sigma)

    self.batch_size = batch_size

    index_grouped_by_label = dataset.index_grouped_by_label
    n_label = self.n_label = len(index_grouped_by_label)
    group_by_label = self.group_by_label = [[] for _ in range(n_label)]
    self.sub_pos = [0] * n_label
    self.pos = 0

    for i in range(n_label):
      if max_n >= 0:
        n_use = max_n // n_label
        if max_n % n_label != 0:
          n_use += int(i < max_n % n_label)
      else:
        n_use = len(index_grouped_by_label[i])
      group_by_label[i] = np.array(index_grouped_by_label[i])[:n_use]

  def __iter__(self):
    return self

  def next(self):
    """Python 2 compatible interface."""
    return self.__next__()

  def __next__(self):
    batch_index, batch_label = [], []
    for i in range(self.pos, self.pos + self.batch_size):
      label = i % self.n_label
      index = self.pick_index(label)
      batch_index.append(index)
      batch_label.append(label)

    self.pos += self.batch_size
    batch_x = self.guassian_data_helper.pick_batch(np.array(batch_index))
    return batch_x, batch_label

  def pick_index(self, label):
    """Pick an index of certain label."""
    sub_pos = self.sub_pos
    group_by_label = self.group_by_label

    if sub_pos[label] == 0:
      np.random.shuffle(group_by_label[label])

    result = group_by_label[label][sub_pos[label]]
    sub_pos[label] = (sub_pos[label] + 1) % len(group_by_label[label])
    return result


def main(unused_argv):
  """Main function."""
  del unused_argv

  dataset_A = load_dataset(FLAGS.config_A, FLAGS.exp_uid_A)
  dataset_B = load_dataset(FLAGS.config_B, FLAGS.exp_uid_B)

  sig = get_sig()
  dirs = common_joint2.get_dirs('joint2_mnist_family', sig)
  save_dir, sample_dir = dirs

  layers = [int(_) for _ in FLAGS.layers.split(',')]
  cls_layers = [int(_) for _ in FLAGS.cls_layers.split(',')]
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

  # Build the joint model.
  tf.reset_default_graph()
  sess = tf.Session()
  m = model_joint2.VAE(vae_config, name='vae')
  m()

  # Load model's architecture (= build)
  one_side_helper_A = OneSideHelper(FLAGS.config_A, FLAGS.exp_uid_A,
                                    FLAGS.config_classifier_A,
                                    FLAGS.exp_uid_classifier_A)
  one_side_helper_B = OneSideHelper(FLAGS.config_B, FLAGS.exp_uid_B,
                                    FLAGS.config_classifier_B,
                                    FLAGS.exp_uid_classifier_B)

  train_writer = tf.summary.FileWriter(save_dir + '/transfer_train', sess.graph)
  scalar_summaries = m.get_scalar_summaries()

  # Initialize and restore
  sess.run(tf.global_variables_initializer())

  one_side_helper_A.restore(dataset_A)
  one_side_helper_B.restore(dataset_B)

  # Miscs from config
  batch_size = vae_config['batch_size']
  n_sup = vae_config['n_sup']

  unsup_iterator_A = DataIterator(dataset_A, max_n=-1, batch_size=batch_size)
  unsup_iterator_B = DataIterator(dataset_B, max_n=-1, batch_size=batch_size)
  sup_iterator_A = DataIterator(dataset_A, max_n=n_sup, batch_size=batch_size)
  sup_iterator_B = DataIterator(dataset_B, max_n=n_sup, batch_size=batch_size)
  domain_A = common_joint2.get_domain_A(batch_size)
  domain_B = common_joint2.get_domain_B(batch_size)

  # Training loop
  n_iters = FLAGS.n_iters
  for i in tqdm(range(n_iters), desc='training', unit=' batch'):
    x_A, _ = next(unsup_iterator_A)
    x_B, _ = next(unsup_iterator_B)
    x_sup_A, label_sup_A = next(sup_iterator_A)
    x_sup_B, label_sup_B = next(sup_iterator_B)

    res = sess.run(
        [m.train_full, scalar_summaries], {
            m.x: np.concatenate([x_A, x_B]),
            m.x_domain: np.concatenate([domain_A, domain_B]),
            m.x_A: x_A,
            m.x_B: x_B,
            m.x_A_domain: domain_A,
            m.x_B_domain: domain_B,
            m.x_cls: np.concatenate([x_sup_A, x_sup_B]),
            m.x_cls_domain: np.concatenate([domain_A, domain_B]),
            m.labels_cls: np.concatenate([label_sup_A, label_sup_B]),
        })
    train_writer.add_summary(res[-1], i)


import pdb, traceback, sys, code  # pylint:disable=W0611,C0413,C0411,C0410
if __name__ == '__main__':
  try:
    tf.app.run(main)
  except Exception:  # pylint:disable=W0703
    post_mortem = True
    if post_mortem:
      type_, value, tb = sys.exc_info()
      traceback.print_exc()
      pdb.post_mortem(tb)
    else:
      raise
