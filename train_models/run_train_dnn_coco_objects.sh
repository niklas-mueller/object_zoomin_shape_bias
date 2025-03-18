#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=72
#SBATCH --mem=480G
#SBATCH --gpus=4
#SBATCH --partition=gpu
#SBATCH --time=30:00:00

. ~/.bashrc
module load 2021
module load Anaconda3/2021.05

conda activate ccnn

model_type=resnet50
image_size=400
zoom=150
CURRENTDATE=`date +"%Y-%m-%d-%H%M%S"`

cd $HOME/projects/oads_access/test

echo "Starting train_dnn_coco_objects.py"

python3 train_dnn_coco_objects.py --image_representation RGB --zoom "${zoom}" --image_size "${image_size}" --output_dir $HOME/projects/prjs0391/coco_objects_results/"${model_type}"/rgb_zoom-"${zoom}"/reps/"${CURRENTDATE}_${image_size}x${image_size}" --n_processes 18 --n_epochs 30 --batch_size 512 --model_type "${model_type}"

# python3 train_dnn_coco_objects.py --image_representation RGB --zoom 80 --image_size 400 --output_dir /home/nmuller/projects/prjs0391/coco_objects_results/resnet50/rgb_zoom-80/reps/test --n_processes 1 --n_epochs 50 --batch_size 4 --model_type resnet50

echo "Finished train_dnn_coco_objects.py"