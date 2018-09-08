```
run_ml_docker --docker-extra-args "-p 6006:6006" tensorboard --port 6006 --logdir ~/workspace/scratch/latent_transfer/poc_joint2_exp1/




run_ml_docker ./run_with_available_gpu python3 ./poc_joint2_exp1.py --default_scratch "~/workspace/scratch/latent_transfer/"


```