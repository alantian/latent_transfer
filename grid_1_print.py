#!/usr/bin/env python3
"""Grid search 1.

For MNIST <-> MNIST
"""

import re

shared_pattern = """\
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 8 \
  --layers "512,512,512,512" \
  --cls_layers "," \
  --prior_loss_beta {plb} \
  --unsup_align_loss_beta {ualb} \
  --cls_loss_beta {clb}  \
  --n_sup {ns} \
  --sig_extra "grid_1{sig_addon}" \
  --post_mortem=false {cmd_addon}\
"""

train_pattern = """\
run_ml_docker --no-it python3 ./train_joint2_mnist_family.py \
""" + shared_pattern

eval_pattern = """\
run_ml_docker --no-it python3 ./evaluate_joint2_mnist_family.py \
""" + " " + shared_pattern + " " + """\
  --load_ckpt_iter -1 \
  --interpolate_labels "0,0,1,1,7,7,8,8,3,3" \
  --nb_images_between_labels 10 \
  --random_seed 114514 \
"""

n_latent_shared = 8
plb_base = 0.005
ualb_base = 1.0
clb_base = 0.05

train_cmds = []
eval_cmds = []

for plb in [0.0, plb_base * 1.]:
  for ualb in [0.0, ualb_base * 1.]:
    for clb in [0.0, clb_base * 1.]:
      for ns in [-1, 0, 10, 100, 1000, 10000]:
        if (clb == 0 and ns != -1) or (clb > 0.0 and ns == 0):
          continue  # no need to waste

        change_n_iter_candidates = [None]
        if plb > 0.0 and ualb > 0.0 and clb > 0.0:
          change_n_iter_candidates.append(50000)

        for change_n_iter in change_n_iter_candidates:
          if change_n_iter is None:
            sig_addon = ''
            cmd_addon = ''
          else:
            sig_addon = '_ni%d' % change_n_iter
            cmd_addon = ' --n_iters %d' % change_n_iter

          cmd = train_pattern.format(
              plb=plb,
              ualb=ualb,
              clb=clb,
              ns=ns,
              sig_addon=sig_addon,
              cmd_addon=cmd_addon,
          )
          cmd = re.sub(' +', ' ', cmd)
          train_cmds.append(cmd)

          cmd = eval_pattern.format(
              plb=plb,
              ualb=ualb,
              clb=clb,
              ns=ns,
              sig_addon=sig_addon,
              cmd_addon=cmd_addon,
          )
          cmd = re.sub(' +', ' ', cmd)
          eval_cmds.append(cmd)

for _ in train_cmds + eval_cmds:
  print(_)