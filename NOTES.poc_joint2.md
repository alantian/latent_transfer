```bash
run_ml_docker --docker-extra-args "-p 6006:6006" tensorboard --port 6006 --logdir ~/workspace/scratch/latent_transfer/poc_joint2_exp1/



# VAE only
run_ml_docker ./run_with_available_gpu python3 ./poc_joint2_exp1.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --layers '16,16,16' \
  --sig_extra "_run2" ;

# VAE with domain
run_ml_docker ./run_with_available_gpu python3 ./poc_joint2_exp1.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --layers '16,16,16' \
  --use_domain=true \
  --sig_extra "_run2" ;

# VAE with domain and SWD
run_ml_docker ./run_with_available_gpu python3 ./poc_joint2_exp1.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --layers '16,16,16' \
  --use_domain=true --unsup_align_loss_beta 0.2 \
  --sig_extra "_run2" ;


# VAE with domain and SWD and classifier on shared space

run_ml_docker ./run_with_available_gpu python3 ./poc_joint2_exp1.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --use_domain=true --unsup_align_loss_beta 0.2 \
  --cls_loss_beta 0.05 \
  --sig_extra "_run0" ;

run_ml_docker ./run_with_available_gpu python3 ./poc_joint2_exp1.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --use_domain=true --unsup_align_loss_beta 0.2 \
  --cls_loss_beta 0.15 \
  --sig_extra "_run0" ;

run_ml_docker ./run_with_available_gpu python3 ./poc_joint2_exp1.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --use_domain=true --unsup_align_loss_beta 0.2 \
  --cls_loss_beta 0.5 \
  --sig_extra "_run0" ;


run_ml_docker ./run_with_available_gpu python3 ./poc_joint2_exp1.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --use_domain=true --unsup_align_loss_beta 0.2 \
  --cls_loss_beta 0.5 \
  --sig_extra "_run2" ;


run_ml_docker ./run_with_available_gpu python3 ./poc_joint2_exp1.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --use_domain=true --unsup_align_loss_beta 0.2 \
  --layers '16,16,16' \
  --cls_loss_beta 0.5 \
  --sig_extra "_run2" ;

run_ml_docker ./run_with_available_gpu python3 ./poc_joint2_exp1.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --use_domain=true --unsup_align_loss_beta 0.2 \
  --layers '16,16,16' \
  --cls_loss_beta 0.15 \
  --sig_extra "_run2" ;



run_ml_docker ./run_with_available_gpu python3 ./poc_joint2_exp1.py \
  --default_scratch "~/workspace/scratch/latent_transfer/" \
  --use_domain=true --unsup_align_loss_beta 0.2 \
  --layers '4,4,4' \
  --cls_loss_beta 0.5 \
  --sig_extra "_run2" ;



```