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
  --sig_extra "grid_1" \
  --n_iters {ni} \
  --use_interpolated {ui} \
  --post_mortem=false \
"""

train_pattern = """\
run_ml_docker --no-it python3 ./train_joint2_mnist_family.py \
""" + shared_pattern

eval_pattern = """\
run_ml_docker --no-it python3 ./evaluate_joint2_mnist_family.py \
""" + " " + shared_pattern + " " + """\
  --load_ckpt_iter 8000 \
  --interpolate_labels "0,0,1,1,7,7,8,8,3,3" \
  --nb_images_between_labels 10 \
  --random_seed 1145141923 \
"""

n_latent_shared = 8
plb_base = 0.005
ualb_base = 1.0
clb_base = 0.05

train_cmds = []
eval_cmds = []


def add(plb, ualb, clb, ns, ni, ui):
  cmd = train_pattern.format(plb=plb, ualb=ualb, clb=clb, ns=ns, ni=ni, ui=ui)
  cmd = re.sub(' +', ' ', cmd)
  # train_cmds.append(cmd)

  cmd = eval_pattern.format(plb=plb, ualb=ualb, clb=clb, ns=ns, ni=ni, ui=ui)
  cmd = re.sub(' +', ' ', cmd)
  eval_cmds.append(cmd)


def main():
  for plb in [plb_base]:
    for ualb in [ualb_base]:
      for clb in [clb_base]:
        # for ns in [-1, 0, 10, 100, 1000, 10000]:
        for ns in [-1]:
          for ni in [20000]:
            for ui in [
                'none', 'linear', 'linear-random', 'spherical',
                'spherical-random'
            ]:

              if (clb == 0 and ns != -1) or (clb > 0.0 and ns == 0):
                continue  # no need to waste

              add(plb=plb, ualb=ualb, clb=clb, ns=ns, ni=ni, ui=ui)

  for _ in train_cmds + eval_cmds:
    print(_)


if __name__ == '__main__':
  main()