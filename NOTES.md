
# A few Notes

`run_ml_docker` is just Yingtao's thin wrapper to run code on docker and can
be safely removed if it's not needed.

# Training model.

```bash

## VAE
run_ml_docker python3 ./train_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent64 --exp_uid "_exp_0"  # DONE
run_ml_docker python3 ./train_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent64 --exp_uid "_exp_1"  # DONE
run_ml_docker python3 ./train_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent64 --exp_uid "_exp_0"  #GPU0
run_ml_docker python3 ./train_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent64 --exp_uid "_exp_1"  #GPU1
run_ml_docker python3 ./train_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent100 --exp_uid "_exp_0"  # DONE
run_ml_docker python3 ./train_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent100 --exp_uid "_exp_1"  # DONE
run_ml_docker python3 ./train_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent100 --exp_uid "_exp_0"  #GPU0
run_ml_docker python3 ./train_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent100 --exp_uid "_exp_1"  #GPU1

## Classifier
run_ml_docker python3 ./train_dataspace_classifier.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_classifier_0 --exp_uid "_exp_0"  #DONE
run_ml_docker python3 ./train_dataspace_classifier.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_classifier_0 --exp_uid "_exp_1"  #DONE
run_ml_docker python3 ./train_dataspace_classifier.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_classifier_0 --exp_uid "_exp_0"  #DONE

## VAE (Unconditional) Sample
run_ml_docker python3 ./sample_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent64 --exp_uid "_exp_0"  # DONE
run_ml_docker python3 ./sample_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent64 --exp_uid "_exp_1"  # DONE
run_ml_docker python3 ./sample_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent64 --exp_uid "_exp_0" # DONE
run_ml_docker python3 ./sample_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent64 --exp_uid "_exp_1" # DONE
run_ml_docker python3 ./sample_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent100 --exp_uid "_exp_0"  # DONE
run_ml_docker python3 ./sample_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent100 --exp_uid "_exp_1"  # DONE
run_ml_docker python3 ./sample_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent100 --exp_uid "_exp_0" # DONE
run_ml_docker python3 ./sample_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent100 --exp_uid "_exp_1" # DONE


## MNIST VAE Encode Data
run_ml_docker python3 ./encode_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent64 --exp_uid "_exp_0"  # DONE
run_ml_docker python3 ./encode_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent64 --exp_uid "_exp_1"  # DONE
run_ml_docker python3 ./encode_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent64 --exp_uid "_exp_0" # DONE
run_ml_docker python3 ./encode_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent64 --exp_uid "_exp_1" # DONE
run_ml_docker python3 ./encode_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent100 --exp_uid "_exp_0"  # DONE
run_ml_docker python3 ./encode_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent100 --exp_uid "_exp_1"  # DONE
run_ml_docker python3 ./encode_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent100 --exp_uid "_exp_0" # DONE
run_ml_docker python3 ./encode_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent100 --exp_uid "_exp_1" # DONE


## sample WaveGAN using classifier.

run_ml_docker python3 ./sample_wavegan.py \
  --gen_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan/bridge" \
  --inception_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan/incept" \
  --latent_dir "~/workspace/scratch/latent_transfer/wavegan/wavegan_gaussian/" \
  ;

## Tensorboard
run_ml_docker --docker-extra-args "-p 6006:6006" tensorboard --logdir ~/workspace/scratch/latent_transfer/joint/


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

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_2mnist_parameterized \
  --exp_uid "_exp_mc_2mnist_run_1_pnall_soofpd_b0.02_0.5_0.5_0.0_0.0" \
  --shuffle_only_once_for_paired_data=true \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.0 --mean_recons_B_to_A_align_beta 0.0 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_2mnist_parameterized \
  --exp_uid "_exp_mc_2mnist_run_1_pnall_soofpd_b0.02_0.5_0.5_0.05_0.05" \
  --shuffle_only_once_for_paired_data=true \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.05 --mean_recons_B_to_A_align_beta 0.05 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_2mnist_parameterized \
  --exp_uid "_exp_mc_2mnist_run_1_pnall_soofpd_b0.02_0.5_0.5_0.025_0.025" \
  --shuffle_only_once_for_paired_data=true \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.025 --mean_recons_B_to_A_align_beta 0.025 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_2mnist_parameterized \
  --exp_uid "_exp_mc_2mnist_run_1_pnall_soofpd_b0.0_0.0_0.0_0.0_0.0" \
  --shuffle_only_once_for_paired_data=true \
  --pairing_number -1 \
  --prior_loss_align_beta 0.0 \
  --mean_recons_A_align_beta 0.0 --mean_recons_B_align_beta 0.0 \
  --mean_recons_A_to_B_align_beta 0.0 --mean_recons_B_to_A_align_beta 0.0 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_2mnist_parameterized \
  --exp_uid "_exp_mc_2mnist_run_1_pnall_soofpd_b0.02_0.5_0.5_0.5_0.5_fb_2.0_2.0" \
  --shuffle_only_once_for_paired_data=true \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.5 --mean_recons_B_to_A_align_beta 0.5 \
  --mean_recons_A_to_B_align_free_budget 2.0 --mean_recons_B_to_A_align_free_budget 2.0 \
  ;


run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_2mnist_parameterized \
  --exp_uid "_exp_mc_2mnist_run_1_pnall_soofpd_b0.02_0.5_0.5_0.5_0.5_fb_1.4_1.4" \
  --shuffle_only_once_for_paired_data=true \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.5 --mean_recons_B_to_A_align_beta 0.5 \
  --mean_recons_A_to_B_align_free_budget 1.4 --mean_recons_B_to_A_align_free_budget 1.4 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_2mnist_parameterized \
  --exp_uid "_exp_mc_2mnist_run_1_pnall_soofpd_b0.02_0.5_0.5_0.5_0.5_fb_1.3_1.3" \
  --shuffle_only_once_for_paired_data=true \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.5 --mean_recons_B_to_A_align_beta 0.5 \
  --mean_recons_A_to_B_align_free_budget 1.3 --mean_recons_B_to_A_align_free_budget 1.3 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_2mnist_parameterized \
  --exp_uid "_exp_mc_2mnist_run_1_pnall_soofpd_b0.02_0.5_0.5_0.5_0.5_fb_1.2_1.2" \
  --shuffle_only_once_for_paired_data=true \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.5 --mean_recons_B_to_A_align_beta 0.5 \
  --mean_recons_A_to_B_align_free_budget 1.2 --mean_recons_B_to_A_align_free_budget 1.2 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_2mnist_parameterized \
  --exp_uid "_exp_mc_2mnist_run_1_pnall_soofpd_b0.02_0.5_0.5_0.5_0.5_fb_1.1_1.1" \
  --shuffle_only_once_for_paired_data=true \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.5 --mean_recons_B_to_A_align_beta 0.5 \
  --mean_recons_A_to_B_align_free_budget 1.1 --mean_recons_B_to_A_align_free_budget 1.1 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_2mnist_parameterized \
  --exp_uid "_exp_mc_2mnist_run_1_pnall_soofpd_b0.02_0.5_0.5_0.5_0.5_fb_1.0_1.0" \
  --shuffle_only_once_for_paired_data=true \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.5 --mean_recons_B_to_A_align_beta 0.5 \
  --mean_recons_A_to_B_align_free_budget 1.0 --mean_recons_B_to_A_align_free_budget 1.0 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_2mnist_parameterized \
  --exp_uid "_exp_mc_2mnist_run_1_pnall_soofpd_b0.02_0.5_0.5_0.5_0.5_fb_0.75_0.75" \
  --shuffle_only_once_for_paired_data=true \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.5 --mean_recons_B_to_A_align_beta 0.5 \
  --mean_recons_A_to_B_align_free_budget 0.75 --mean_recons_B_to_A_align_free_budget 0.75 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_2mnist_parameterized \
  --exp_uid "_exp_mc_2mnist_run_1_pnall_soofpd_b0.02_0.5_0.5_0.5_0.5_fb_0.5_0.5" \
  --shuffle_only_once_for_paired_data=true \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.5 --mean_recons_B_to_A_align_beta 0.5 \
  --mean_recons_A_to_B_align_free_budget 0.5 --mean_recons_B_to_A_align_free_budget 0.5 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_2mnist_parameterized \
  --exp_uid "_exp_mc_2mnist_run_1_pnall_soofpd_b0.02_0.5_0.5_0.5_0.5_fb_0.25_0.25" \
  --shuffle_only_once_for_paired_data=true \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.5 --mean_recons_B_to_A_align_beta 0.5 \
  --mean_recons_A_to_B_align_free_budget 0.25 --mean_recons_B_to_A_align_free_budget 0.25 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_2mnist_parameterized \
  --exp_uid "_exp_mc_2mnist_run_1_pnall_soofpd_b0.02_0.5_0.5_0.05_0.05_fb_1.0_1.0" \
  --shuffle_only_once_for_paired_data=true \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.05 --mean_recons_B_to_A_align_beta 0.05 \
  --mean_recons_A_to_B_align_free_budget 1.0 --mean_recons_B_to_A_align_free_budget 1.0 \
  ;


run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_2mnist_parameterized \
  --exp_uid "_exp_mc_2mnist_run_1_pnall_soofpd_b0.02_0.5_0.5_0.05_0.05_fb_0.75_0.75" \
  --shuffle_only_once_for_paired_data=true \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.05 --mean_recons_B_to_A_align_beta 0.05 \
  --mean_recons_A_to_B_align_free_budget 0.75 --mean_recons_B_to_A_align_free_budget 0.75 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_2mnist_parameterized \
  --exp_uid "_exp_mc_2mnist_run_1_pnall_soofpd_b0.02_0.5_0.5_0.1_0.1_fb_1.0_1.0" \
  --shuffle_only_once_for_paired_data=true \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.1 --mean_recons_B_to_A_align_beta 0.1 \
  --mean_recons_A_to_B_align_free_budget 1.0 --mean_recons_B_to_A_align_free_budget 1.0 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_mnist2fashion_parameterized \
  --exp_uid "_exp_mc_mnist2fashion_run_1_pnall_soofpd_b0.02_0.5_0.5_0.1_0.1_fb_1.0_1.0" \
  --shuffle_only_once_for_paired_data=true \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.1 --mean_recons_B_to_A_align_beta 0.1 \
  --mean_recons_A_to_B_align_free_budget 1.0 --mean_recons_B_to_A_align_free_budget 1.0 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_mnist2fashion_parameterized \
  --exp_uid "_exp_mc_mnist2fashion_run_1_pnall_soofpd_b0.0_0.0_0.0_0.0_0.0_fb_0.0_0.0" \
  --shuffle_only_once_for_paired_data=true \
  --pairing_number -1 \
  --prior_loss_align_beta 0.0 \
  --mean_recons_A_align_beta 0.0 --mean_recons_B_align_beta 0.0 \
  --mean_recons_A_to_B_align_beta 0.0 --mean_recons_B_to_A_align_beta 0.0 \
  --mean_recons_A_to_B_align_free_budget 0.0 --mean_recons_B_to_A_align_free_budget 0.0 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_mnist2fashion_parameterized \
  --exp_uid "_exp_mc_mnist2fashion_run_1_pnall_soofpd_b0.02_0.5_0.5_0.0_0.0_fb_0.0_0.0" \
  --shuffle_only_once_for_paired_data=true \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.0 --mean_recons_B_to_A_align_beta 0.0 \
  --mean_recons_A_to_B_align_free_budget 0.0 --mean_recons_B_to_A_align_free_budget 0.0 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_mnist2fashion_parameterized \
  --exp_uid "_exp_mc_mnist2fashion_run_1_nls8_pnall_soofpd_b0.00_0.0_0.0_0.0_0.0_fb_0.0_0.0" \
  --shuffle_only_once_for_paired_data=true \
  --n_latent_shared 8 \
  --pairing_number -1 \
  --prior_loss_align_beta 0.00 \
  --mean_recons_A_align_beta 0.0 --mean_recons_B_align_beta 0.0 \
  --mean_recons_A_to_B_align_beta 0.0 --mean_recons_B_to_A_align_beta 0.0 \
  --mean_recons_A_to_B_align_free_budget 0.0 --mean_recons_B_to_A_align_free_budget 0.0 \
  ;


run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_mnist2fashion_parameterized \
  --exp_uid "_exp_mc_mnist2fashion_run_1_nls8_pnall_soofpd_b0.02_0.5_0.5_0.0_0.0_fb_0.0_0.0" \
  --shuffle_only_once_for_paired_data=true \
  --n_latent_shared 8 \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.0 --mean_recons_B_to_A_align_beta 0.0 \
  --mean_recons_A_to_B_align_free_budget 0.0 --mean_recons_B_to_A_align_free_budget 0.0 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_mnist2fashion_parameterized \
  --exp_uid "_exp_mc_mnist2fashion_run_1_nls8_pnall_soofpd_b0.02_0.5_0.5_0.1_0.1_fb_1.0_1.0" \
  --shuffle_only_once_for_paired_data=true \
  --n_latent_shared 8 \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.1 --mean_recons_B_to_A_align_beta 0.1 \
  --mean_recons_A_to_B_align_free_budget 1.0 --mean_recons_B_to_A_align_free_budget 1.0 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_mnist2fashion_parameterized \
  --exp_uid "_exp_mc_mnist2fashion_run_1_nls8_pnall_soofpd_b0.02_0.5_0.5_0.05_0.05_fb_0.0_0.0" \
  --shuffle_only_once_for_paired_data=true \
  --n_latent_shared 8 \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.05 --mean_recons_B_to_A_align_beta 0.05 \
  --mean_recons_A_to_B_align_free_budget 0.0 --mean_recons_B_to_A_align_free_budget 0.0 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_mnist2fashion_parameterized \
  --exp_uid "_exp_mc_mnist2fashion_run_1_nls8_pnall_soofpd_b0.02_0.5_0.5_0.05_0.05_fb_1.0_1.0" \
  --shuffle_only_once_for_paired_data=true \
  --n_latent_shared 8 \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.05 --mean_recons_B_to_A_align_beta 0.05 \
  --mean_recons_A_to_B_align_free_budget 1.0 --mean_recons_B_to_A_align_free_budget 1.0 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --wavegan_gen_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan" \
  --wavegan_inception_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan/incept" \
  --wavegan_latent_dir "~/workspace/scratch/latent_transfer/wavegan/wavegan_gaussian" \
  --config joint_exp_mnist100_2wavegan_parameterized \
  --exp_uid "_exp_mc_mnist100_2wavegann_run_1_nls8_pnall_b0.02_0.5_0.5_0.0_0.0_fb_0.0_0.0" \
  --n_latent_shared 8 \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.0 --mean_recons_B_to_A_align_beta 0.0 \
  --mean_recons_A_to_B_align_free_budget 0.0 --mean_recons_B_to_A_align_free_budget 0.0 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --wavegan_gen_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan" \
  --wavegan_inception_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan/incept" \
  --wavegan_latent_dir "~/workspace/scratch/latent_transfer/wavegan/wavegan_gaussian" \
  --config joint_exp_mnist100_2wavegan_parameterized \
  --exp_uid "_exp_mc_mnist100_2wavegann_run_1_nls8_pnall_b0.0_0.0_0.0_0.0_0.0_fb_0.0_0.0" \
  --n_latent_shared 8 \
  --pairing_number -1 \
  --prior_loss_align_beta 0.0 \
  --mean_recons_A_align_beta 0.0 --mean_recons_B_align_beta 0.0 \
  --mean_recons_A_to_B_align_beta 0.0 --mean_recons_B_to_A_align_beta 0.0 \
  --mean_recons_A_to_B_align_free_budget 0.0 --mean_recons_B_to_A_align_free_budget 0.0 \
  ;

run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --wavegan_gen_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan" \
  --wavegan_inception_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan/incept" \
  --wavegan_latent_dir "~/workspace/scratch/latent_transfer/wavegan/wavegan_gaussian" \
  --config joint_exp_mnist100_2wavegan_parameterized \
  --exp_uid "_exp_mc_mnist100_2wavegann_run_1_nls8_pnall_b0.0_0.0_0.0_0.0_0.0_fb_0.0_0.0" \
  --n_latent_shared 8 \
  --pairing_number -1 \
  --prior_loss_align_beta 0.0 \
  --mean_recons_A_align_beta 0.0 --mean_recons_B_align_beta 0.0 \
  --mean_recons_A_to_B_align_beta 0.0 --mean_recons_B_to_A_align_beta 0.0 \
  --mean_recons_A_to_B_align_free_budget 0.0 --mean_recons_B_to_A_align_free_budget 0.0 \
  ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --wavegan_gen_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan" \
  --wavegan_inception_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan/incept" \
  --wavegan_latent_dir "~/workspace/scratch/latent_transfer/wavegan/wavegan_gaussian" \
  --config joint_exp_mnist100_2wavegan_parameterized \
  --exp_uid "_exp_mc_mnist100_2wavegann_run_1_nls8_pnall_b0.02_0.5_0.5_0.1_0.1_fb_0.0_0.0" \
  --n_latent_shared 8 \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.1 --mean_recons_B_to_A_align_beta 0.1 \
  --mean_recons_A_to_B_align_free_budget 0.0 --mean_recons_B_to_A_align_free_budget 0.0 \
  ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --wavegan_gen_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan" \
  --wavegan_inception_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan/incept" \
  --wavegan_latent_dir "~/workspace/scratch/latent_transfer/wavegan/wavegan_gaussian" \
  --config joint_exp_mnist100_2wavegan_v2_parameterized \
  --exp_uid "_exp_mc_mnist100_2wavegann_v2_run_1_nls8_pnall_b0.02_0.5_0.5_0.1_0.1_fb_1.0_1.0" \
  --n_latent_shared 8 \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.1 --mean_recons_B_to_A_align_beta 0.1 \
  --mean_recons_A_to_B_align_free_budget 1.0 --mean_recons_B_to_A_align_free_budget 1.0 \
  ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --wavegan_gen_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan" \
  --wavegan_inception_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan/incept" \
  --wavegan_latent_dir "~/workspace/scratch/latent_transfer/wavegan/wavegan_gaussian" \
  --config joint_exp_mnist100_2wavegan_v2_parameterized \
  --exp_uid "_exp_mc_mnist100_2wavegann_v2_run_1_nls8_pnall_b0.02_0.5_0.5_0.1_3.0_fb_1.0_0.0" \
  --n_latent_shared 8 \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.1 --mean_recons_B_to_A_align_beta 3.0 \
  --mean_recons_A_to_B_align_free_budget 1.0 --mean_recons_B_to_A_align_free_budget 0.0 \
  ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --wavegan_gen_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan" \
  --wavegan_inception_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan/incept" \
  --wavegan_latent_dir "~/workspace/scratch/latent_transfer/wavegan/wavegan_gaussian" \
  --config joint_exp_mnist100_2wavegan_v2_parameterized \
  --exp_uid "_exp_mc_mnist100_2wavegann_v2_run_1_nls8_pnall_b0.02_0.5_0.5_3.0_0.1_fb_0.0_1.0" \
  --n_latent_shared 8 \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 3.0 --mean_recons_B_to_A_align_beta 0.1 \
  --mean_recons_A_to_B_align_free_budget 0.0 --mean_recons_B_to_A_align_free_budget 1.0 \
  ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --wavegan_gen_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan" \
  --wavegan_inception_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan/incept" \
  --wavegan_latent_dir "~/workspace/scratch/latent_transfer/wavegan/wavegan_gaussian" \
  --config joint_exp_mnist100_2wavegan_v2_parameterized \
  --exp_uid "_exp_mc_mnist100_2wavegann_v2_run_1_nls8_pnall_b0.02_0.5_0.5_10.0_0.1_fb_0.0_1.0" \
  --n_latent_shared 8 \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 10.0 --mean_recons_B_to_A_align_beta 0.1 \
  --mean_recons_A_to_B_align_free_budget 0.0 --mean_recons_B_to_A_align_free_budget 1.0 \
  ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint.py  --n_iters_per_eval 1000 --n_iters 40000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --wavegan_gen_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan" \
  --wavegan_inception_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan/incept" \
  --wavegan_latent_dir "~/workspace/scratch/latent_transfer/wavegan/wavegan_gaussian" \
  --config joint_exp_mnist100_2wavegan_v2_parameterized \
  --exp_uid "_exp_mc_mnist100_2wavegann_v2_run_1_nls8_pnall_b0.02_0.5_0.5_0.5_0.1_fb_0.0_1.0" \
  --n_latent_shared 8 \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.5 --mean_recons_B_to_A_align_beta 0.1 \
  --mean_recons_A_to_B_align_free_budget 0.0 --mean_recons_B_to_A_align_free_budget 1.0 \
  ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint.py  --n_iters_per_eval 1000 --n_iters 40000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --wavegan_gen_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan" \
  --wavegan_inception_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan/incept" \
  --wavegan_latent_dir "~/workspace/scratch/latent_transfer/wavegan/wavegan_gaussian" \
  --config joint_exp_mnist100_2wavegan_v2_parameterized \
  --exp_uid "_exp_mc_mnist100_2wavegann_v2_run_1_nls8_pnall_b0.02_0.5_0.5_0.5_0.5_fb_0.0_0.0" \
  --n_latent_shared 8 \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.5 --mean_recons_B_to_A_align_beta 0.5 \
  --mean_recons_A_to_B_align_free_budget 0.0 --mean_recons_B_to_A_align_free_budget 0.0 \
  ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint.py  --n_iters_per_eval 1000 --n_iters 40000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --wavegan_gen_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan" \
  --wavegan_inception_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan/incept" \
  --wavegan_latent_dir "~/workspace/scratch/latent_transfer/wavegan/wavegan_gaussian" \
  --config joint_exp_mnist100_2wavegan_v2_parameterized \
  --exp_uid "_exp_mc_mnist100_2wavegann_v2_run_1_nls8_pnall_b0.0_0.0_0.0_0.0_0.0_fb_0.0_0.0" \
  --n_latent_shared 8 \
  --pairing_number -1 \
  --prior_loss_align_beta 0.00 \
  --mean_recons_A_align_beta 0.0 --mean_recons_B_align_beta 0.0 \
  --mean_recons_A_to_B_align_beta 0.0 --mean_recons_B_to_A_align_beta 0.0 \
  --mean_recons_A_to_B_align_free_budget 0.0 --mean_recons_B_to_A_align_free_budget 0.0 \
  ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --wavegan_gen_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan" \
  --wavegan_inception_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan/incept" \
  --wavegan_latent_dir "~/workspace/scratch/latent_transfer/wavegan/wavegan_gaussian" \
  --config joint_exp_mnist100_2wavegan_v2_parameterized \
  --exp_uid "_exp_mc_mnist100_2wavegann_v2_run_1_nls8_pnall_b0.02_0.5_0.5_0.5_0.5_fb_0.0_0.0_run2" \
  --n_latent_shared 8 \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 0.5 \
  --mean_recons_A_to_B_align_beta 0.5 --mean_recons_B_to_A_align_beta 0.5 \
  --mean_recons_A_to_B_align_free_budget 0.0 --mean_recons_B_to_A_align_free_budget 0.0 \
  ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --wavegan_gen_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan" \
  --wavegan_inception_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan/incept" \
  --wavegan_latent_dir "~/workspace/scratch/latent_transfer/wavegan/wavegan_gaussian" \
  --config joint_exp_mnist100_2wavegan_v2_parameterized \
  --exp_uid "_exp_mc_mnist100_2wavegann_v2_run_1_nls8_pnall_b0.02_0.5_1.5_0.5_0.5_fb_0.0_0.0" \
  --n_latent_shared 8 \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 1.5 \
  --mean_recons_A_to_B_align_beta 0.5 --mean_recons_B_to_A_align_beta 0.5 \
  --mean_recons_A_to_B_align_free_budget 0.0 --mean_recons_B_to_A_align_free_budget 0.0 \
  ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint.py  --n_iters_per_eval 1000 \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --wavegan_gen_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan" \
  --wavegan_inception_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan/incept" \
  --wavegan_latent_dir "~/workspace/scratch/latent_transfer/wavegan/wavegan_gaussian" \
  --config joint_exp_mnist100_2wavegan_v2_parameterized \
  --exp_uid "_exp_mc_mnist100_2wavegann_v2_run_1_nls8_pnall_b0.02_0.5_1.5_1.5_0.5_fb_0.0_0.0" \
  --n_latent_shared 8 \
  --pairing_number -1 \
  --prior_loss_align_beta 0.02 \
  --mean_recons_A_align_beta 0.5 --mean_recons_B_align_beta 1.5 \
  --mean_recons_A_to_B_align_beta 1.5 --mean_recons_B_to_A_align_beta 0.5 \
  --mean_recons_A_to_B_align_free_budget 0.0 --mean_recons_B_to_A_align_free_budget 0.0 \
  ;




## Interpolation of Transfer model

run_ml_docker ./run_with_available_gpu python3 ./interpolate_joint.py  \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_2mnist_parameterized \
  --exp_uid "_exp_mc_2mnist_run_1_pnall_soofpd_b0.02_0.5_0.5_0.1_0.1_fb_1.0_1.0" \
  --load_ckpt_iter 90000  \
  --interpolate_labels "0,1,7,8,3" \
  --nb_images_between_labels 10 \
  ;

run_ml_docker ./run_with_available_gpu python3 ./interpolate_joint.py  \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config joint_exp_mnist2fashion_parameterized \
  --exp_uid "_exp_mc_mnist2fashion_run_1_nls8_pnall_soofpd_b0.02_0.5_0.5_0.05_0.05_fb_1.0_1.0" \
  --n_latent_shared 8 \
  --load_ckpt_iter 90000  \
  --interpolate_labels "0,1,7,8,3" \
  --nb_images_between_labels 10 \
  ;



```
