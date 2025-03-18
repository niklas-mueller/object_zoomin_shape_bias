#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=128
#SBATCH --mem=224G
#SBATCH --partition=rome
#SBATCH --time=03:00:00

. ~/.bashrc
module load 2021
module load Anaconda3/2021.05

conda activate ccnn


cd $HOME/projects/coco_results


echo "Starting make_coco_object_crop_dataset.py"

# Single core version
# python3 make_coco_object_crop_dataset.py


# Multi-core version
NPROC=`nproc --all`

for i in `seq 1 $NPROC`; do
  python3 make_coco_object_crop_dataset.py --i $i --nproc 128 --target_dir dataset/coco_objects &
done
wait

echo "Finished make_coco_object_crop_dataset.py"