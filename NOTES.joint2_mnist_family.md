```bash
run_ml_docker --docker-extra-args "-p 6007:6007" tensorboard --port 6007 --logdir ~/workspace/scratch/latent_transfer/joint2_mnist_family/

# MNIST <> MNIST, VAE only

run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 2 \
  --sig_extra "_run0_dev4" ;

  # ===

run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 2 \
  --layers "512,512,512,512" \
  --sig_extra "_run0_dev1" ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 4 \
  --layers "512,512,512,512" \
  --sig_extra "_run0_dev1" ;

run_ml_docker ./run_with_available_gpu python3 ./train_joint2_mnist_family.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --config_A "mnist_0_nlatent100" --config_B "mnist_0_nlatent100" \
  --config_classifier_A "mnist_classifier_0" --config_classifier_B "mnist_classifier_0" \
  --n_latent 100 --n_latent_shared 8 \
  --layers "512,512,512,512" \
  --sig_extra "_run0_dev1" ;

```