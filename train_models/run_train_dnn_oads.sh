#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=18
#SBATCH --mem=120G
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=10:00:00

. ~/.bashrc
module load 2021
module load Anaconda3/2021.05

conda activate ccnn

model_type=resnet50
image_size=400
zoom=0
n_epochs=30
CURRENTDATE=`date +"%Y-%m-%d-%H%M%S"`


echo "Starting train_dnn_oads_objects.py"

########## Train new model

# # Train with zoomed out crops
# python3 train_dnn_oads_objects.py --image_representation RGB_Zoom-"${zoom}" --image_size "${image_size}" --input_dir $HOME/projects/data/oads --output_dir $HOME/projects/prjs0391/oads_results/"${model_type}"/rgb_zoom-"${zoom}"/reps/"${CURRENTDATE}_${image_size}x${image_size}" --n_processes 18 --n_epochs "${n_epochs}" --batch_size 128 --model_type "${model_type}"

# # Train with standard crops
# python3 train_dnn_oads_objects.py --image_representation RGB --image_size "${image_size}" --input_dir $HOME/projects/data/oads --output_dir $HOME/projects/prjs0391/oads_results/"${model_type}"/rgb/reps/"${CURRENTDATE}_${image_size}x${image_size}" --n_processes 18 --n_epochs "${n_epochs}" --batch_size 128 --model_type "${model_type}"


# Train with zoomed-in crops
python3 train_dnn_oads_objects.py --image_representation RGB -imagenet_cropping --image_size "${image_size}" --input_dir $HOME/projects/data/oads --output_dir $HOME/projects/prjs0391/oads_results/"${model_type}"/rgb/reps/"${CURRENTDATE}_800x800_to_${image_size}x${image_size}" --n_processes 18 --n_epochs "${n_epochs}" --batch_size 128 --model_type "${model_type}"

echo "Finished train_dnn_oads_objects.py"
