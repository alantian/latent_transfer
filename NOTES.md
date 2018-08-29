
# A few Notes

`run_ml_docker` is just Yingtao's thin wrapper to run code on docker and can
be safely removed if it's not needed.

# Training model.

```bash
## Transfer model

################################################################################
# mnist on 64
################################################################################

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_2mnist_parameterized \
  --exp_uid "_exp_2mnist_run_1_pnall_b0.02_0.5_0.5_0.5_0.5" \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.5 --mean_recons_B_to_A_align_beta 0.5 \
  ;

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






```
