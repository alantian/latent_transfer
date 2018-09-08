# Tensorboard

```bash

run_ml_docker --docker-extra-args "-p 6006:6006" tensorboard --port 6006 --logdir ~/workspace/scratch/latent_transfer/mnist/ckpts/
run_ml_docker --docker-extra-args "-p 6007:6007" tensorboard --port 6007 --logdir ~/workspace/scratch/latent_transfer/fashion-mnist/ckpts/
run_ml_docker --docker-extra-args "-p 6008:6008" tensorboard --port 6008 --logdir ~/workspace/scratch/latent_transfer/joint/ckpts/


```

# Training model.
```bash


## VAE
run_ml_docker ./run_with_available_gpu python3 ./train_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent64_xsigma1 --exp_uid "_exp_0"
run_ml_docker ./run_with_available_gpu python3 ./train_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent64_xsigma1 --exp_uid "_exp_1" 
run_ml_docker ./run_with_available_gpu python3 ./train_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent64_xsigma1 --exp_uid "_exp_0"
run_ml_docker ./run_with_available_gpu python3 ./train_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent64_xsigma1 --exp_uid "_exp_1" 
run_ml_docker ./run_with_available_gpu python3 ./train_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent100_xsigma1 --exp_uid "_exp_0"
run_ml_docker ./run_with_available_gpu python3 ./train_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent100_xsigma1 --exp_uid "_exp_1" 
run_ml_docker ./run_with_available_gpu python3 ./train_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent100_xsigma1 --exp_uid "_exp_0"
run_ml_docker ./run_with_available_gpu python3 ./train_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent100_xsigma1 --exp_uid "_exp_1"

## Classifier
run_ml_docker python3 ./train_dataspace_classifier.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_classifier_0 --exp_uid "_exp_0"  
run_ml_docker python3 ./train_dataspace_classifier.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_classifier_0 --exp_uid "_exp_1"  
run_ml_docker python3 ./train_dataspace_classifier.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_classifier_0 --exp_uid "_exp_0"  


## VAE (Unconditional) Sample
run_ml_docker ./run_with_available_gpu python3 ./sample_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent64_xsigma1 --exp_uid "_exp_0"  
run_ml_docker ./run_with_available_gpu python3 ./sample_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent64_xsigma1 --exp_uid "_exp_1"  
run_ml_docker ./run_with_available_gpu python3 ./sample_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent64_xsigma1 --exp_uid "_exp_0" 
run_ml_docker ./run_with_available_gpu python3 ./sample_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent64_xsigma1 --exp_uid "_exp_1" 
run_ml_docker ./run_with_available_gpu python3 ./sample_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent100_xsigma1 --exp_uid "_exp_0"  
run_ml_docker ./run_with_available_gpu python3 ./sample_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent100_xsigma1 --exp_uid "_exp_1"  
run_ml_docker ./run_with_available_gpu python3 ./sample_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent100_xsigma1 --exp_uid "_exp_0" 
run_ml_docker ./run_with_available_gpu python3 ./sample_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent100_xsigma1 --exp_uid "_exp_1" 

## MNIST VAE Encode Data

## MNIST VAE Encode Data
run_ml_docker ./run_with_available_gpu python3 ./encode_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent64_xsigma1 --exp_uid "_exp_0"  
run_ml_docker ./run_with_available_gpu python3 ./encode_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent64_xsigma1 --exp_uid "_exp_1"  
run_ml_docker ./run_with_available_gpu python3 ./encode_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent64_xsigma1 --exp_uid "_exp_0" 
run_ml_docker ./run_with_available_gpu python3 ./encode_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent64_xsigma1 --exp_uid "_exp_1" 
run_ml_docker ./run_with_available_gpu python3 ./encode_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent100_xsigma1 --exp_uid "_exp_0"  
run_ml_docker ./run_with_available_gpu python3 ./encode_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config mnist_0_nlatent100_xsigma1 --exp_uid "_exp_1"  
run_ml_docker ./run_with_available_gpu python3 ./encode_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent100_xsigma1 --exp_uid "_exp_0" 
run_ml_docker ./run_with_available_gpu python3 ./encode_dataspace.py --default_scratch "~/workspace/scratch/latent_transfer/" --config fashion_mnist_0_nlatent100_xsigma1 --exp_uid "_exp_1" 


## sample WaveGAN.

run_ml_docker ./run_with_available_gpu python3 ./sample_wavegan.py \
  --gen_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan/bridge" \
  --inception_ckpt_dir "~/workspace/scratch/latent_transfer/wavegan/incept" \
  --latent_dir "~/workspace/scratch/latent_transfer/wavegan/wavegan_gaussian_non_selective/" \
  --selective=false \
  --total_per_label 7000 \
  --top_per_label 6000 \
  ;


# Joint


## for mnist2mnist_exp1, args are:
##    n_latent_shared, pairing_number, prior_loss_beta_A, prior_loss_beta_B, prior_loss_align_beta, mean_recons_A_align_beta, mean_recons_B_align_beta, mean_recons_A_to_B_align_beta, mean_recons_B_to_A_align_beta, mean_recons_A_to_B_align_free_budget, mean_recons_B_to_A_align_free_budget (optional)runid


# this is AE only
./train_joint_run.sh mnist2mnist_exp_v2 8 -1 0.0001 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 run0
./train_joint_run.sh mnist2mnist_exp_v2 8 -1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 run0
./train_joint_run.sh mnist2mnist_exp_v2 100 -1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 run0
./train_joint_run.sh mnist2mnist_exp_v2 8 -1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 run1 # use lr=1-e4
./train_joint_run.sh mnist2mnist_exp_v2 8 -1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 run2 # use layer=512*4,
./train_joint_run.sh mnist2mnist_exp_v2 8 -1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 run3 # use layer=512*4, lr=1e-5
./train_joint_run.sh mnist2mnist_exp_v2 8 -1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 run4 # use layer=512*8, lr=1e-5
./train_joint_run.sh mnist2mnist_exp_v2 100 -1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 run1 # use layer=512*8, lr=1e-5

```