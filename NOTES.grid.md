
# Grid Search 1

...

```bash
GRID_DIR=~/workspace/scratch/latent_transfer/grid/1
GRID_CMDS=$GRID_DIR/cmds
GRID_HASH=$GRID_DIR/hash

mkdir -p $GRID_DIR
./grid_1_print.py > $GRID_CMDS
rm -rf $GRID_HASH

./host_run_batch_jobs.py --nb_gpu 4 --cmds_file $GRID_CMDS --finished_hash_file $GRID_HASH
```