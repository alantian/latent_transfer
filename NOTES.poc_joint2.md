```bash
run_ml_docker --docker-extra-args "-p 6006:6006" tensorboard --port 6006 --logdir ~/workspace/scratch/latent_transfer/poc_joint2_exp1/



# VAE only
run_ml_docker ./run_with_available_gpu python3 ./poc_joint2_exp1.py --default_scratch "~/workspace/scratch/latent_transfer/" \
  ;

# VAE with domain
run_ml_docker ./run_with_available_gpu python3 ./poc_joint2_exp1.py --default_scratch "~/workspace/scratch/latent_transfer/" \
  --use_domain=true ;

# VAE with domain and SWD
run_ml_docker ./run_with_available_gpu python3 ./poc_joint2_exp1.py --default_scratch "~/workspace/scratch/latent_transfer/" \
  --use_domain=true --unsup_align_loss_beta 0.2 ;

run_ml_docker ./run_with_available_gpu python3 ./poc_joint2_exp1.py --default_scratch "~/workspace/scratch/latent_transfer/" \
  --use_domain=true --unsup_align_loss_beta 0.3 ;

```