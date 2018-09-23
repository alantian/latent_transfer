#!/usr/bin/env python3
"""Run multiple jobs, potentially in parallel to leverage multiple GPUs on host."""

import hashlib
from os import environ
from os.path import expanduser, exists
import subprocess
import shlex
import time

from absl import app
from absl import logging
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('cmds_file', '~/.run_batch_jobs.cmds_file', '')
flags.DEFINE_integer('nb_gpu', 4, '')
flags.DEFINE_string('finished_hash_file', '~/.run_batch_jobs.finished_hash', '')
flags.DEFINE_integer('loop_interval', 2, '')


def load_lines(f):
  if exists(f):
    res = [_.strip('\r\n') for _ in open(f).readlines()]
    res = [_ for _ in res if _]  # remove empy lines
  else:
    res = []

  return res


def add_lines(lines, f):
  with open(f, 'a') as fout:
    for line in lines:
      print(line, file=fout)


def load_cmds():
  return load_lines(expanduser(FLAGS.cmds_file))


def get_finished_hashs_set():
  return set(load_lines(expanduser(FLAGS.finished_hash_file)))


def in_finished_hash(hash_):
  return hash_ in get_finished_hashs_set()


def add_to_finished_hash(hash_):
  add_lines([hash_], expanduser(FLAGS.finished_hash_file))


def hash_cmd(s):
  return hashlib.sha224(s.encode()).hexdigest()


def main(unused_argv):
  """Main function."""
  del unused_argv

  logging.set_verbosity(logging.INFO)

  # Prepare
  proc_in_run = {}  # hash: str -> (id_gpu: int, popen: Popen}
  proc_to_run = {}  # hash: str -> cmd: str
  free_gpus_set = set(range(FLAGS.nb_gpu))

  last_state_changed = True

  # Main loop
  while True:
    # logging.info('Loop tick')
    # logging.info('proc_in_run %s', proc_in_run)
    # logging.info('proc_to_run %s', proc_to_run)
    # logging.info('free_gpus_set %s', free_gpus_set)

    state_changed = False

    # print and eject finished popen object
    hash_keys = list(proc_in_run.keys())
    for hash_ in hash_keys:
      id_gpu, popen = proc_in_run[hash_]
      poll_result = popen.poll()
      if poll_result is not None:
        logging.info(
            'command done (return code = %d): "%s"',
            popen.returncode,
            popen.args,
        )
        free_gpus_set.add(id_gpu)
        add_to_finished_hash(hash_)
        del proc_in_run[hash_]
        state_changed = True

    # Add job to queue if is not seen before
    finished_hashs_set = get_finished_hashs_set()
    cmds = load_cmds()

    for cmd in cmds:
      hash_ = hash_cmd(cmd)
      if (hash_ not in finished_hashs_set) and (hash_ not in proc_in_run) and (
          hash_ not in proc_to_run):
        proc_to_run[hash_] = cmd
        logging.info('Add to queue: "%s"', shlex.split(cmd))
        state_changed = True

    # Launch if there is slot
    hash_keys = list(proc_to_run.keys())
    for hash_ in hash_keys:
      cmd = proc_to_run[hash_]
      if free_gpus_set:
        logging.info('Launch: "%s"', shlex.split(cmd))
        id_gpu = free_gpus_set.pop()
        env = {k: v for k, v in environ.items()}
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # match the order from nvidia-smi
        env['CUDA_VISIBLE_DEVICES'] = '%d' % id_gpu
        popen = subprocess.Popen(
            args=shlex.split(cmd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        proc_in_run[hash_] = (id_gpu, popen)
        del proc_to_run[hash_]
        state_changed = True

    if not state_changed:
      if last_state_changed:  # only print once when entering the state-not-changed
        logging.info('State not changed.')
      last_state_changed = state_changed

    # Sleep
    time.sleep(FLAGS.loop_interval)


import pdb, traceback, sys, code  # pylint:disable=W0611,C0413,C0411,C0410
if __name__ == '__main__':
  try:
    app.run(main)
  except Exception:  # pylint:disable=W0703
    post_mortem = True
    if post_mortem:
      type_, value_, tb = sys.exc_info()
      traceback.print_exc()
      pdb.post_mortem(tb)
    else:
      raise
