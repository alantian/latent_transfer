#!/usr/bin/env python3
"""Grid search 1.

For MNIST <-> MNIST
"""

import re

shared_pattern = """\
run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --wavegan_gen_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan" \
  --wavegan_inception_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan/incept" \
  --wavegan_latent_dir "~/workspace/scratch/latent_transfer/wavegan/wavegan_gaussian" \
  --config_A "mnist_0_nlatent100" --config_B "wavegan" \
  --config_classifier_A "mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 16 \
  --layers "1024,1024,1024,1024,1024,1024" \
  --cls_layers "," \
  --prior_loss_beta {plb} \
  --unsup_align_loss_beta {ualb} \
  --cls_loss_beta {clb} \
  --cls_layers "," \
  --n_sup {ns} \
  --sig_extra "grid_3" \
  --post_mortem=false \
"""

train_pattern = """\
run_ml_docker --no-it python3 ./train_joint2_mnist_family.py \
""" + shared_pattern

n_latent_shared = 8
plb_base = 0.001
ualb_base = 3.0
clb_base = 0.03

train_cmds = []
eval_cmds = []

for plb in [0.0, plb_base * 1.]:
  for ualb in [0.0, ualb_base * 1.]:
    for clb in [0.0, clb_base * 1.]:
      for ns in [-1, 0, 10, 100, 1000, 10000]:
        if (clb == 0 and ns != -1) or (clb > 0.0 and ns == 0):
          continue  # no need to waste
        cmd = train_pattern.format(plb=plb, ualb=ualb, clb=clb, ns=ns)
        cmd = re.sub(' +', ' ', cmd)
        train_cmds.append(cmd)

for _ in train_cmds + eval_cmds:
  print(_)