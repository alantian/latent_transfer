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
import itertools
import os
from os.path import join

import numpy as np
from scipy.io import wavfile
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix
import tensorflow as tf

import common
import model_dataspace
import model_joint2

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string(
    'wavegan_gen_ckpt_dir', '', 'The directory to WaveGAN generator\'s ckpt. '
    'If WaveGAN is involved, this argument must be set.')
tf.flags.DEFINE_string(
    'wavegan_inception_ckpt_dir', '',
    'The directory to WaveGAN inception (classifier)\'s ckpt. '
    'If WaveGAN is involved, this argument must be set.')
tf.flags.DEFINE_string(
    'wavegan_latent_dir', '', 'The directory to WaveGAN\'s latent space.'
    'If WaveGAN is involved, this argument must be set.')

DirsBlob = namedtuple('DirsBlob', ['save_dir', 'sample_dir'])


def get_dirs(global_sig, sig, clear_dir=False):
  """Get needed directories for training.

  This method would create dir(s) if not existing. Also would
    remove content in dir(s) if anything exists AND `clear_dir` is set.

  Args:
    global_sig: (string) global singature.
    sig: (string) local, or run-wise, signature.
    clear_dir: (boolean) wehther to clear the content if there is 
      something inside.

  Returns:
    An DirsBlob instance, storing needed directories as strings.
  """

  local_base_path = join(common.get_default_scratch(), global_sig)

  save_dir = join(local_base_path, 'save', sig)
  if clear_dir:
    os.system('rm -rf "%s"' % save_dir)
  tf.gfile.MakeDirs(save_dir)

  sample_dir = join(local_base_path, 'sample', sig)
  if clear_dir:
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
  this_config_is_wavegan = common.config_is_wavegan(config)
  if this_config_is_wavegan:
    return load_dataset_wavegan()

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


def load_dataset_wavegan():
  """Load WaveGAN's dataset.

  The loaded dataset consists of:
    - original data (dataset_blob, train_data, train_label),
    - encoded data from a pretrained model (train_mu, train_sigma), and
    - index grouped by label (index_grouped_by_label).

  Some of these attributes are not avaiable (set as None) but are left here
  to keep everything aligned with returned value of `load_dataset`.

  Returns:
    An tuple of abovementioned components in the dataset.
  """

  latent_dir = os.path.expanduser(FLAGS.wavegan_latent_dir)
  path_train = os.path.join(latent_dir, 'data_train.npz')
  train = np.load(path_train)
  train_z = train['z']
  train_mu = train_z
  train_label = train['label']
  index_grouped_by_label = common.get_index_grouped_by_label(train_label)

  return DatasetBlob(
      train_data=None,
      train_label=train_label,
      train_mu=train_mu,
      train_sigma=None,
      index_grouped_by_label=index_grouped_by_label,
      save_path=None,
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

  def save_data(self,
                x,
                name,
                save_dir,
                x_is_real_x=False,
                batch_image_fn=None,
                emphasize=None):
    """Save dataspace instances.

    Args:
      x: A numpy array of dataspace points.
      name: A string indicating the name in the saved file.
      save_dir: A string indicating the directory to put the saved file.
      x_is_real_x: An boolean indicating whether `x` is already in dataspace. If
          not, `x` is converted to dataspace before saving
      batch_image_fn: An function that batches images. If not specified,
          default function `common.batch_image` will be used.
      emphasize: An optional list of indecies of x to emphasize. It not specied,
          nothing would be empasized.
    """
    batch_image_fn = batch_image_fn or common.batch_image

    if not x_is_real_x:
      np.savetxt(join(save_dir, '%s.array.txt' % name), x)
    real_x = x if x_is_real_x else self.decode(x)
    real_x = common.post_proc(real_x, self.config, emphasize=emphasize)
    batched_real_x = batch_image_fn(real_x)
    sample_file = join(save_dir, '%s.png' % name)
    common.save_image(batched_real_x, sample_file)


class ModelHelperWaveGAN(object):
  """A Helper that provides sampling and classification for pre-trained WaveGAN.
  """
  DEFAULT_BATCH_SIZE = 100

  def __init__(self):
    self.build()

  def build(self):
    """Build the TF graph and heads from pre-trained WaveGAN ckpts.

    It also prepares different graph, session and heads for sampling and
    classification respectively.
    """

    # pylint:disable=unused-variable
    # Reason:
    #   All endpoints are stored as attribute at the end of `_build`.
    #   Pylint cannot infer this case so it emits false alarm of
    #   unused-variable if we do not disable this warning.

    # pylint:disable=invalid-name
    # Reason:
    #   Variable useing 'G' in is name to be consistent with WaveGAN's author
    #   has name consider to be invalid by pylint so we disable the warning.

    # Dataset (SC09, WaveGAN)'s generator
    graph_sc09_gan = tf.Graph()
    with graph_sc09_gan.as_default():
      # Use the retrained, Gaussian priored model
      gen_ckpt_dir = os.path.expanduser(FLAGS.wavegan_gen_ckpt_dir)
      sess_sc09_gan = tf.Session(graph=graph_sc09_gan)
      saver_gan = tf.train.import_meta_graph(
          join(gen_ckpt_dir, 'infer', 'infer.meta'))

    # Dataset (SC09, WaveGAN)'s  classifier (inception)
    graph_sc09_class = tf.Graph()
    with graph_sc09_class.as_default():
      inception_ckpt_dir = os.path.expanduser(FLAGS.wavegan_inception_ckpt_dir)
      sess_sc09_class = tf.Session(graph=graph_sc09_class)
      saver_class = tf.train.import_meta_graph(
          join(inception_ckpt_dir, 'infer.meta'))

    # Dataset B (SC09, WaveGAN)'s Tensor symbols
    sc09_gan_z = graph_sc09_gan.get_tensor_by_name('z:0')
    sc09_gan_G_z = graph_sc09_gan.get_tensor_by_name('G_z:0')[:, :, 0]

    # Classification: Tensor symbols
    sc09_class_x = graph_sc09_class.get_tensor_by_name('x:0')
    sc09_class_scores = graph_sc09_class.get_tensor_by_name('scores:0')

    # Add all endpoints as object attributes
    for k, v in locals().items():
      self.__dict__[k] = v

  def restore(self):
    """Restore the weights of models."""
    gen_ckpt_dir = self.gen_ckpt_dir
    graph_sc09_gan = self.graph_sc09_gan
    saver_gan = self.saver_gan
    sess_sc09_gan = self.sess_sc09_gan

    inception_ckpt_dir = self.inception_ckpt_dir
    graph_sc09_class = self.graph_sc09_class
    saver_class = self.saver_class
    sess_sc09_class = self.sess_sc09_class

    with graph_sc09_gan.as_default():
      saver_gan.restore(sess_sc09_gan, join(gen_ckpt_dir, 'bridge',
                                            'model.ckpt'))

    with graph_sc09_class.as_default():
      saver_class.restore(sess_sc09_class,
                          join(inception_ckpt_dir, 'best_acc-103005'))

    # pylint:enable=unused-variable
    # pylint:enable=invalid-name

  def decode(self, z, batch_size=None):
    """Decode from given latant space vectors `z`.

    Args:
      z: A numpy array of latent space vectors.
      batch_size: (Optional) a integer to indication batch size for computation
          which is useful if the sampling requires lots of GPU memory.

    Returns:
      A numpy array, the dataspace points from decoding.
    """
    batch_size = batch_size or self.DEFAULT_BATCH_SIZE
    return run_with_batch(self.sess_sc09_gan, self.sc09_gan_G_z,
                          self.sc09_gan_z, z, batch_size)

  def classify(self, real_x, batch_size=None):
    """Classify given dataspace points `real_x`.

    Args:
      real_x: A numpy array of dataspace points.
      batch_size: (Optional) a integer to indication batch size for computation
          which is useful if the classification requires lots of GPU memory.

    Returns:
      A numpy array, the prediction from classifier.
    """
    batch_size = batch_size or self.DEFAULT_BATCH_SIZE
    pred = run_with_batch(self.sess_sc09_class, self.sc09_class_scores,
                          self.sc09_class_x, real_x, batch_size)
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
    batched_real_x = common.batch_audio(real_x)
    sample_file = join(save_dir, '%s.wav' % name)
    wavfile.write(sample_file, rate=16000, data=batched_real_x)


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
    this_config_is_wavegan = common.config_is_wavegan(config)

    if this_config_is_wavegan:
      # The sample object servers both purpose.
      m_helper = ModelHelperWaveGAN()
      m_classifier_helper = m_helper
    else:  # MNIST
      # In this case two diffent objects serve two purpose.
      m_helper = ModelHelper(config_name, exp_uid)
      m_classifier_helper = ModelHelper(config_name_classifier,
                                        exp_uid_classifier)

    self.config_name = config_name
    self.config = config
    self.this_config_is_wavegan = this_config_is_wavegan
    self.m_helper = m_helper
    self.m_classifier_helper = m_classifier_helper

  def restore(self, dataset):
    """Restore the pretrained model and classifier.

    Args:
      dataset: The object containts `save_path` used for restoring.
    """
    m_helper = self.m_helper
    m_classifier_helper = self.m_classifier_helper

    if self.this_config_is_wavegan:
      m_helper.restore()
      # We don't need restore the `m_classifier_helper` again since `m_helper`
      # and `m_classifier_helper` are two identicial objects.
    else:
      m_helper.restore_best('vae_saver', dataset.save_path, 'vae_best_%s.ckpt')
      m_classifier_helper.restore_best('classifier_saver', dataset.save_path,
                                       'classifier_best_%s.ckpt')

  def decode_and_classify(self, x):
    real_x = self.m_helper.decode(x)
    pred = self.m_classifier_helper.classify(real_x)
    return pred

  def save_data(self, *args, **kwargs):
    self.m_helper.save_data(*args, **kwargs)


class ManualSummaryHelper(object):
  """A helper making manual TF summary easier."""

  def __init__(self):
    self._key_to_ph_summary_tuple = {}

  def get_summary(self, sess, key, value):
    """Get TF (scalar) summary.

    Args:
      sess: A TF Session to be used in making summary.
      key: A string indicating the name of summary.
      value: A string indicating the value of summary.

    Returns:
      A TF summary.
    """
    with sess.graph.as_default():
      self._add_key_if_not_exists(key)
    placeholder, summary = self._key_to_ph_summary_tuple[key]
    return sess.run(summary, {placeholder: value})

  def _add_key_if_not_exists(self, key):
    """Add related TF heads for a key if it is not used before."""
    if key in self._key_to_ph_summary_tuple:
      return
    placeholder = tf.placeholder(tf.float32, shape=(), name=key + '_ph')
    summary = tf.summary.scalar(key, placeholder)
    self._key_to_ph_summary_tuple[key] = (placeholder, summary)


# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
# With modification
def plot_confusion_matrix(cm,
                          classes,
                          fpath,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=None):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """

  import matplotlib
  # avoid tkinter. This should happen before `import matplotlib.pyplot`.
  matplotlib.use('agg')
  import matplotlib.pyplot as plt

  plt.tight_layout()
  plt.figure()

  cmap = cmap or plt.cm.Blues
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  vmin = 0.0
  vmax = np.max(cm.sum(axis=1))

  plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j], fmt),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('Source label')
  plt.xlabel('Target label')

  plt.savefig(fpath)
  plt.close()


class JointVAEHelper(object):
  """The helper that managers the joint VAE model in the latent space.

  """

  def __init__(self, config, save_dir):
    self.config = config
    batch_size = config['batch_size']

    graph = self.graph = tf.Graph()
    with graph.as_default():
      # these variables need to be in thes graph.
      sess = self.sess = tf.Session(graph=graph)
      m = self.m = model_joint2.VAE(config, name='vae')
      m()
      sess.run(tf.global_variables_initializer())
      self.scalar_summaries = m.get_scalar_summaries()

    self.ckpt_dir = save_dir + '/ckpt'
    self.train_writer = tf.summary.FileWriter(save_dir + '/transfer_train',
                                              sess.graph)
    self.eval_writer = self.train_writer

    self.domain_A = get_domain_A(batch_size)
    self.domain_B = get_domain_B(batch_size)

    self.manual_summary_helper = ManualSummaryHelper()

  def train_one_batch(self, i, x_A, x_B, x_sup_A, x_sup_B, label_sup_A,
                      label_sup_B):
    sess = self.sess
    m = self.m
    scalar_summaries = self.scalar_summaries
    domain_A, domain_B = self.domain_A, self.domain_B
    train_writer = self.train_writer

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

  def save(self, i):
    ckpt_dir = self.ckpt_dir
    sess = self.sess
    m = self.m
    save_name = join(ckpt_dir, 'transfer_%d.ckpt' % i)
    m.vae_saver.save(sess, save_name)
    with tf.gfile.Open(join(ckpt_dir, 'ckpt_iters.txt'), 'w') as f:
      f.write('%d' % i)

  def restore(self, i):
    ckpt_dir = self.ckpt_dir
    sess = self.sess
    m = self.m
    save_name = join(ckpt_dir, 'transfer_%d.ckpt' % i)
    m.vae_saver.restore(sess, save_name)

  def eval_summary(self, key, value, i):
    manual_summary_helper = self.manual_summary_helper
    sess = self.sess
    eval_writer = self.eval_writer

    summary = manual_summary_helper.get_summary(sess, key, value)
    eval_writer.add_summary(summary, i)

  def sample_prior(self, sample_size):
    n_latent_shared = self.config['n_latent_shared']
    sess = self.sess
    m = self.m
    domain_A = get_domain_A(sample_size)
    domain_B = get_domain_B(sample_size)

    z_hat = np.random.randn(sample_size, n_latent_shared)
    x_A = sess.run(m.x_prime_decode, {
        m.q_z_sample_decode: z_hat,
        m.q_z_sample_domain_decode: domain_A
    })
    x_B = sess.run(m.x_prime_decode, {
        m.q_z_sample_decode: z_hat,
        m.q_z_sample_domain_decode: domain_B
    })
    return x_A, x_B

  def get_x_prime(self, x, encode_domain_sig, decode_domain_sig):
    assert encode_domain_sig in ('A', 'B')
    assert decode_domain_sig in ('A', 'B')

    sess = self.sess
    m = self.m

    domain_A = get_domain_A(x.shape[0])
    domain_B = get_domain_B(x.shape[0])

    encode_domain = domain_A if encode_domain_sig == 'A' else domain_B
    decode_domain = domain_A if decode_domain_sig == 'A' else domain_B

    return sess.run(
        m.x_prime_transfer,
        {
            m.x_transfer: x,
            m.x_transfer_encode_domain: encode_domain,
            m.x_transfer_decode_domain: decode_domain,
        },
    )

  def get_x_prime_A(self, x_A):
    return self.get_x_prime(x_A, 'A', 'A')

  def get_x_prime_B(self, x_B):
    return self.get_x_prime(x_B, 'B', 'B')

  def get_x_prime_B_from_x_A(self, x_A):
    return self.get_x_prime(x_A, 'A', 'B')

  def get_x_prime_A_from_x_B(self, x_B):
    return self.get_x_prime(x_B, 'B', 'A')

  def compare(self,
              x_1,
              x_2,
              one_side_helper_1,
              one_side_helper_2,
              eval_dir,
              i,
              sig,
              n_label=10):
    """Get stats of prediciton. We assume x_1 is true and x1 is pred if needed."""

    def get_accuarcy(pred_1, pred_2):
      return np.mean(np.equal(pred_1, pred_2).astype('f'))

    def normalize(v):
      return v / max(1e-10, np.absolute(v).sum())

    labels = list(range(n_label))
    pred_1 = one_side_helper_1.decode_and_classify(x_1)
    pred_2 = one_side_helper_2.decode_and_classify(x_2)

    accuarcy_ = get_accuarcy(pred_1, pred_2)

    confusion_matrix_ = confusion_matrix(
        y_true=pred_1,
        y_pred=pred_2,
        labels=labels,
    )

    entropy_ = np.array(
        [entropy(normalize(row)) for row in confusion_matrix_]).sum()

    self.eval_summary('accuracy_' + sig, accuarcy_, i)
    self.eval_summary('entropy_' + sig, entropy_, i)

    np.savetxt(
        join(eval_dir, 'confusion_matrix_' + sig + '.txt'), confusion_matrix_)

    plot_confusion_matrix(
        confusion_matrix_,
        labels,
        join(eval_dir, 'confusion_matrix_' + sig + '.png'),
    )

    plot_confusion_matrix(
        confusion_matrix_,
        labels,
        join(eval_dir, 'confusion_matrix_normalized_' + sig + '.png'),
        normalize=True,
    )


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

  VALID_USE_INTERPOLATED_CANDIDATE = {
      'none', 'linear', 'linear-random', 'spherical', 'spherical-random'
  }

  def __init__(self, dataset, max_n, batch_size, use_interpolated='none'):
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

    assert use_interpolated in self.VALID_USE_INTERPOLATED_CANDIDATE
    self.use_interpolated = use_interpolated

    self.debug_first_next_is_done = False

  def __iter__(self):
    return self

  def next(self):
    """Python 2 compatible interface."""
    return self.__next__()

  def __next__(self):
    if self.use_interpolated != 'none':
      real_size = max(1, self.batch_size // 2)
      int_size = self.batch_size - real_size  # int means interpolated in this method
    else:
      real_size = self.batch_size
      int_size = 0

    batch_x, batch_label = self.pick_batch(real_size)

    if int_size > 0:
      int_x, int_label = self.get_interpolated(batch_x, int_size)
      batch_x = np.concatenate((batch_x, int_x), axis=0)
      batch_label = batch_label + int_label

    if not self.debug_first_next_is_done:
      tf.logging.info(
          'iterator: real_size = %d, int_size = %d, batch_x.shape = %s, len(batch_label) = %d',
          real_size, int_size, batch_x.shape, len(batch_label))
      self.debug_first_next_is_done = True

    return batch_x, batch_label

  def get_interpolated(self, x, size):
    index_a = np.remainder(np.arange(size), len(x))
    index_b = np.random.randint(len(x), size=(size,))

    x_a = x[index_a, :]
    x_b = x[index_b, :]
    if self.use_interpolated == 'linear':
      x_out = x_a * 0.5 + x_b * 0.5
    if self.use_interpolated == 'linear-random':
      p = np.random.uniform(0.0, 1.0)
      x_out = x_a * 0.5 + x_b * 0.5
    elif self.use_interpolated == 'spherical':
      sqrt_half = 0.70710678118
      x_out = x_a * sqrt_half + x_b * sqrt_half
    elif self.use_interpolated == 'spherical-random':
      p = np.random.uniform(0.0, 1.0)
      x_out = x_a * np.sqrt(p) + x_b * np.sqrt(1.0 - p)

    assert x_out.shape == (size, x.shape[1])

    res_x = x_out
    res_label = [-1] * size

    return res_x, res_label

  def pick_batch(self, batch_size):
    batch_index, batch_label = [], []
    for i in range(self.pos, self.pos + batch_size):
      label = i % self.n_label
      index = self.pick_index(label)
      batch_index.append(index)
      batch_label.append(label)
    self.pos += batch_size
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
