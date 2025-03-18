#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=18
#SBATCH --mem=120G
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=15:00:00


. ~/.bashrc
module load 2021
module load Anaconda3/2021.05

conda activate ccnn

model_type=fasterrcnn_resnet50_fpn_coco
CURRENTDATE=`date +"%Y-%m-%d-%H%M%S"`


echo "Starting train_dnn.py"

# Train new model
python3 train_dnn.py --image_representation RGB --image_size 400 --input_dir $HOME/projects/data/oads --output_dir /home/nmuller/projects/oads_results/"${model_type}"/rgb/reps/"${CURRENTDATE}" --n_processes 18 --n_epochs 15 --batch_size 16 --model_type "${model_type}"

echo "Finished train_dnn.py"
