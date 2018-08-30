
# A few Notes

`run_ml_docker` is just Yingtao's thin wrapper to run code on docker and can
be safely removed if it's not needed.

# Training model.

```bash

## sample WaveGAN using classifier.

run_ml_docker python3 ./sample_wavegan.py \
  --gen_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan/bridge" \
  --inception_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan/incept" \
  --latent_dir "~/workspace/scratch/latent_transfer/wavegan/wavegan_gaussian/" \
  ;




## Transfer model


################################################################################
# mnist on 64, shared latent dim = 2
################################################################################

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_2mnist_parameterized \
  --exp_uid "_exp_2mnist_run_1_pnall_soofpd_b0.02_0.5_0.5_0.5_0.5" \
  --shuffle_only_once_for_paired_data=true \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.5 --mean_recons_B_to_A_align_beta 0.5 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_2fashion_parameterized \
  --exp_uid "_exp_2fashion_run_1_pnall_soofpd_b0.02_0.5_0.5_0.5_0.5" \
  --shuffle_only_once_for_paired_data=true \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.5 --mean_recons_B_to_A_align_beta 0.5 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_mnist2fashion_parameterized \
  --exp_uid "_exp_mnist2fashion_run_1_pnall_soofpd_b0.02_0.5_0.5_0.5_0.5" \
  --shuffle_only_once_for_paired_data=true \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.5 --mean_recons_B_to_A_align_beta 0.5 \
  ;


################################################################################
# mnist on 64, shared latent dim = 4
################################################################################

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_2mnist_parameterized \
  --exp_uid "_exp_2mnist_run_1_nls4_pnall_soofpd_b0.02_0.5_0.5_0.5_0.5" \
  --shuffle_only_once_for_paired_data=true \
  --n_latent_shared 4 \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.5 --mean_recons_B_to_A_align_beta 0.5 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_2fashion_parameterized \
  --exp_uid "_exp_2fashion_run_1_nls4_pnall_soofpd_b0.02_0.5_0.5_0.5_0.5" \
  --shuffle_only_once_for_paired_data=true \
  --n_latent_shared 4 \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.5 --mean_recons_B_to_A_align_beta 0.5 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_mnist2fashion_parameterized \
  --exp_uid "_exp_mnist2fashion_run_1_nls4_pnall_soofpd_b0.02_0.5_0.5_0.5_0.5" \
  --shuffle_only_once_for_paired_data=true \
  --n_latent_shared 4 \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.5 --mean_recons_B_to_A_align_beta 0.5 \
  ;


################################################################################
# experiment "mc": mnist on 64, shared latent dim = 2
################################################################################

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_2mnist_parameterized \
  --exp_uid "_exp_mc_2mnist_run_1_pnall_soofpd_b0.02_0.5_0.5_0.5_0.5" \
  --shuffle_only_once_for_paired_data=true \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.5 --mean_recons_B_to_A_align_beta 0.5 \
  ;



```
