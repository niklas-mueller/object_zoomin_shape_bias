#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=16
#SBATCH --mem=28G
#SBATCH --partition=rome
#SBATCH --time=50:00:00

. ~/.bashrc
module load 2021
module load Anaconda3/2021.05

conda activate ccnn


cd $HOME/projects/oads_access/test


echo "Starting make_zoom_crop_dataset.py"

# Single core version
# python3 make_zoom_crop_dataset.py


# Multi-core version
NPROC=`nproc --all`

for i in `seq 1 $NPROC`; do
  python3 make_zoom_crop_dataset.py --i $i &
done
wait

echo "Finished make_zoom_crop_dataset.py"