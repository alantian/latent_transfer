```bash

# TF:
run_ml_docker --docker-extra-args "-p 6007:6007" tensorboard --port 6007 --logdir ~/workspace/scratch/latent_transfer/joint2_mnist_family/save

# train uses ./run_with_available_gpu while evalue use ./run_with_no_gpu




# MNIST <> MNIST, AE only

run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 2 \
  --layers "512,512,512,512" \
  --sig_extra "_exp0_ae_run0" ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 4 \
  --layers "512,512,512,512" \
  --sig_extra "_exp0_ae_run0" ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 8 \
  --layers "512,512,512,512" \
  --sig_extra "_exp0_ae_run0" ;


run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100_xsigma1" --config_B "mnist_0_nlatent100_xsigma1" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 32 \
  --layers "512,512,512,512" \
  --cls_layers "," \
  --prior_loss_beta 0.005
  --unsup_align_loss_beta 0.0 \
  --cls_loss_beta 0.0 \
  --sig_extra "_exp0_ae_mnist_xsigma1_run0" ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100_xsigma1" --config_B "mnist_0_nlatent100_xsigma1" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 64 \
  --layers "512,512,512,512" \
  --cls_layers "," \
  --prior_loss_beta 0.005
  --unsup_align_loss_beta 0.0 \
  --cls_loss_beta 0.0 \
  --sig_extra "_exp0_ae_mnist_xsigma1_run0" ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100_xsigma1" --config_B "mnist_0_nlatent100_xsigma1" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 100 \
  --layers "512,512,512,512" \
  --cls_layers "," \
  --prior_loss_beta 0.005
  --unsup_align_loss_beta 0.0 \
  --cls_loss_beta 0.0 \
  --sig_extra "_exp0_ae_mnist_xsigma1_run0" ;



# MNIST <> MNIST, VAE. Note that beta is highly related to shared dim...1 dim ~ 10.

run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 8 \
  --layers "512,512,512,512" \
  --prior_loss_beta 0.005 \
  --sig_extra "_exp1_vae_run0" ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 8 \
  --layers "512,512,512,512" \
  --prior_loss_beta 0.01 \
  --sig_extra "_exp1_vae_run0" ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 8 \
  --layers "512,512,512,512" \
  --prior_loss_beta 0.02 \
  --sig_extra "_exp1_vae_run0" ;


# MNIST <> MNIST, VAE + Unsup align

run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 8 \
  --layers "512,512,512,512" \
  --prior_loss_beta 0.005 \
  --unsup_align_loss_beta 1.0 \
  --sig_extra "_exp2_vae+unsup_run0" ;

# ========================================================

# MNIST <> MNIST, VAE + Unsup align + cls (sup) align

run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 8 \
  --layers "512,512,512,512" \
  --prior_loss_beta 0.005 \
  --unsup_align_loss_beta 1.0 \
  --cls_loss_beta 0.05 \
  --sig_extra "_exp3_vae+unsup+cls_run0" ;


# --------------------------------------------------------

#  DONE
run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 8 \
  --layers "512,512,512,512" \
  --cls_layers "," \
  --prior_loss_beta 0.005 \
  --unsup_align_loss_beta 1.0 \
  --cls_loss_beta 0.05 \
  --sig_extra "_exp3_vae+unsup+cls_run0" ;

run_ml_docker ./run_with_no_gpu python3 ./evaluate_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 8 \
  --layers "512,512,512,512" \
  --cls_layers "," \
  --prior_loss_beta 0.005 \
  --unsup_align_loss_beta 1.0 \
  --cls_loss_beta 0.05 \
  --sig_extra "_exp3_vae+unsup+cls_run0" \
  --load_ckpt_iter -1 \
  --interpolate_labels "0,0,1,1,7,7,8,8,3,3" \
  --nb_images_between_labels 10 \
  --random_seed 114514 \
  ;

# --------------------------------------------------------

run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100_xsigma1" --config_B "mnist_0_nlatent100_xsigma1" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 8 \
  --layers "512,512,512,512" \
  --cls_layers "," \
  --prior_loss_beta 0.005 \
  --unsup_align_loss_beta 1.0 \
  --cls_loss_beta 0.05 \
  --sig_extra "_exp3_vae+unsup+cls_run0" ;

# --------------------------------------------------------

# ========================================================


# MNIST <> Fashion MNIST, AE only

run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "fashion_mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "fashion_mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 8 \
  --layers "512,512,512,512" \
  --sig_extra "_exp0_ae_run0" ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "fashion_mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "fashion_mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 16 \
  --layers "512,512,512,512" \
  --sig_extra "_exp0_ae_run0" ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "fashion_mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "fashion_mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 8 \
  --layers "512,512,512,512,512,512" \
  --sig_extra "_exp0_ae_run0" ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "fashion_mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "fashion_mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 16 \
  --layers "512,512,512,512,512,512" \
  --sig_extra "_exp0_ae_run0" ;


run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "fashion_mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "fashion_mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 16 \
  --layers "1024,1024,1024,1024,1024,1024,1024,1024" \
  --sig_extra "_exp0_ae_run0" ;


# MNIST <> Fashion MNIST, VAE.

run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "fashion_mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "fashion_mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 8 \
  --layers "512,512,512,512" \
  --prior_loss_beta 0.005 \
  --sig_extra "_exp1_vae_run0" ;


# MNIST <> Fashion MNIST, VAE + Unsup align

run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "fashion_mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "fashion_mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 8 \
  --layers "512,512,512,512" \
  --prior_loss_beta 0.005 \
  --unsup_align_loss_beta 1.0 \
  --sig_extra "_exp2_vae+unsup_run0" ;


# MNIST <> Fashion MNIST, VAE + Unsup align + cls (sup) align


run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "fashion_mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "fashion_mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 8 \
  --layers "512,512,512,512" \
  --cls_layers "8" \
  --prior_loss_beta 0.005 \
  --unsup_align_loss_beta 1.0 \
  --cls_loss_beta 0.05 \
  --sig_extra "_exp3_vae+unsup+cls_run0" ;


run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "fashion_mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "fashion_mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 16 \
  --layers "512,512,512,512,512,512" \
  --cls_layers "8" \
  --prior_loss_beta 0.0025 \
  --unsup_align_loss_beta 1.0 \
  --cls_loss_beta 0.05 \
  --sig_extra "_exp3_vae+unsup+cls_run0" ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "fashion_mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "fashion_mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 16 \
  --layers "512,512,512,512,512,512" \
  --cls_layers "8" \
  --prior_loss_beta 0.005 \
  --unsup_align_loss_beta 1.0 \
  --cls_loss_beta 0.05 \
  --sig_extra "_exp3_vae+unsup+cls_run0" ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "fashion_mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "fashion_mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 16 \
  --layers "512,512,512,512,512,512" \
  --cls_layers "," \
  --prior_loss_beta 0.0025 \
  --unsup_align_loss_beta 1.0 \
  --cls_loss_beta 0.05 \
  --sig_extra "_exp3_vae+unsup+cls_run0" ;

run_ml_docker ./run_with_no_gpu python3 ./evaluate_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "fashion_mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "fashion_mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 16 \
  --layers "512,512,512,512,512,512" \
  --cls_layers "," \
  --prior_loss_beta 0.0025 \
  --unsup_align_loss_beta 1.0 \
  --cls_loss_beta 0.05 \
  --sig_extra "_exp3_vae+unsup+cls_run0" \
  --load_ckpt_iter -1 \
  --interpolate_labels "0,0,1,1,7,7,8,8,3,3" \
  --nb_images_between_labels 10 \
  ;

```