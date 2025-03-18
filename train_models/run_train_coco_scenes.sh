#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=18
#SBATCH --mem=120G
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=50:00:00

. ~/.bashrc
module load 2021
module load Anaconda3/2021.05

conda activate ccnn

echo "Starting train_dnn_coco_scenes.py"

python train_dnn_coco_scenes.py  --n_epochs 50 --batch_size 16 --n_processes 18 --gpus 1 --nr 0

echo "Finished train_dnn_coco_scenes.py"