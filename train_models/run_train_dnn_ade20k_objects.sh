#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=18
#SBATCH --mem=120G
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=30:00:00

. ~/.bashrc
module load 2021
module load Anaconda3/2021.05

conda activate ccnn

model_type=resnet50
image_size=224
zoom=0
CURRENTDATE=`date +"%Y-%m-%d-%H%M%S"`


echo "Starting train_dnn_ade20k_objects.py"

python3 train_dnn_ade20k_objects.py --image_representation RGB --zoom "${zoom}" --image_size "${image_size}" --output_dir $HOME/projects/prjs0391/ade20k_objects_results/"${model_type}"/rgb_zoom-"${zoom}"/reps/"${CURRENTDATE}_${image_size}x${image_size}" --n_processes 18 --n_epochs 30 --batch_size 128 --model_type "${model_type}"

echo "Finished train_dnn_ade20k_objects.py"