#!/usr/bin/env python3
"""Grid search 3.

For MNIST <-> WaveGAN
"""

import re

shared_pattern = """\
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

eval_pattern_list = [
    """\
run_ml_docker --no-it python3 ./evaluate_joint2_mnist_family.py \
""" + " " + shared_pattern + " " + """\
  --load_ckpt_iter 50000 \
  --interpolate_labels "8,8,8,8,8,8,8" \
  --nb_images_between_labels 4 \
  --random_seed 1145141925 \
""", """\
run_ml_docker --no-it python3 ./evaluate_joint2_mnist_family.py \
""" + " " + shared_pattern + " " + """\
  --load_ckpt_iter 50000 \
  --interpolate_labels "6,6,6,6,6,6,6" \
  --nb_images_between_labels 4 \
  --random_seed 1145141925 \
""", """\
run_ml_docker --no-it python3 ./evaluate_joint2_mnist_family.py \
""" + " " + shared_pattern + " " + """\
  --load_ckpt_iter 50000 \
  --interpolate_labels "7,7,7,7,7,7,7" \
  --nb_images_between_labels 4 \
  --random_seed 1145141925 \
""", """\
run_ml_docker --no-it python3 ./evaluate_joint2_mnist_family.py \
""" + " " + shared_pattern + " " + """\
  --load_ckpt_iter 50000 \
  --interpolate_labels "0,1,7,8,9,3,2" \
  --nb_images_between_labels 4 \
  --random_seed 1145141925 \
"""
]

train_cmds = []
eval_cmds = []


def add(plb, ualb, clb, ns, ni, ui):
  cmd = train_pattern.format(plb=plb, ualb=ualb, clb=clb, ns=ns, ni=ni, ui=ui)
  cmd = re.sub(' +', ' ', cmd)
  # train_cmds.append(cmd)

  for eval_pattern in eval_pattern_list:
    cmd = eval_pattern.format(plb=plb, ualb=ualb, clb=clb, ns=ns, ni=ni, ui=ui)
    cmd = re.sub(' +', ' ', cmd)
    eval_cmds.append(cmd)


def main():
  for plb in [0.01]:
    for ualb in [3.0]:
      for clb in [0.3]:
        # for ns in [-1, 0, 10, 100, 1000, 10000]:
        for ns in [-1]:
          for ni in [50001]:
            for ui in ['none']:
              if (clb == 0 and ns != -1) or (clb > 0.0 and ns == 0):
                continue  # no need to waste

              add(plb=plb, ualb=ualb, clb=clb, ns=ns, ni=ni, ui=ui)

  for _ in train_cmds + eval_cmds:
    print(_)


if __name__ == '__main__':
  main()