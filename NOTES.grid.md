
# Grid Search 1

...

```bash
GRID_DIR=~/workspace/scratch/latent_transfer/grid
GRID_CMDS=$GRID_DIR/cmds
GRID_HASH=$GRID_DIR/hash

mkdir -p $GRID_DIR
./grid_1_print.py >> $GRID_CMDS
./grid_2_print.py >> $GRID_CMDS
./grid_3_print.py >> $GRID_CMDS

./host_run_batch_jobs.py --nb_gpu 4 --cmds_file $GRID_CMDS --finished_hash_file $GRID_HASH
```