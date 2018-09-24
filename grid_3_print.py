#!/usr/bin/env python3
"""Grid search 3.

For MNIST <-> WaveGAN
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
  --layers "1024,1024,1024,1024,1024,1024,1024,1024" \
  --cls_layers "," \
  --prior_loss_beta {plb} \
  --unsup_align_loss_beta {ualb} \
  --cls_loss_beta {clb}  \
  --n_sup {ns} \
  --sig_extra "grid_2" \
  --n_iters {ni} \
  --use_interpolated {ui} \
  --post_mortem=false \
"""

train_pattern = """\
run_ml_docker --no-it python3 ./train_joint2_mnist_family.py \
""" + shared_pattern

train_cmds = []
eval_cmds = []


def add(plb, ualb, clb, ns, ni, ui):
  cmd = train_pattern.format(plb=plb, ualb=ualb, clb=clb, ns=ns, ni=ni, ui=ui)
  cmd = re.sub(' +', ' ', cmd)
  train_cmds.append(cmd)

  #cmd = eval_pattern.format(plb=plb, ualb=ualb, clb=clb, ns=ns, ni=ni, ui=ui)
  #cmd = re.sub(' +', ' ', cmd)
  #eval_cmds.append(cmd)


def main():
  for plb in [0.01]:
    for ualb in [3.0]:
      for clb in [0.03, 0.3, 0.5]:
        # for ns in [-1, 0, 10, 100, 1000, 10000]:
        for ns in [-1]:
          for ni in [20000, 50000]:
            for ui in ['none']:
              if (clb == 0 and ns != -1) or (clb > 0.0 and ns == 0):
                continue  # no need to waste

              add(plb=plb, ualb=ualb, clb=clb, ns=ns, ni=ni, ui=ui)

  for _ in train_cmds + eval_cmds:
    print(_)


if __name__ == '__main__':
  main()