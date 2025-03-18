#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=18
#SBATCH --mem=120G
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=05:00:00

. ~/.bashrc
module load 2021
module load Anaconda3/2021.05

conda activate ccnn

echo "Starting finetune_dnns_imagenet.py"

# python finetune_dnns_imagenet.py --input_dir /home/nmuller/projects/data/Imagenet --image_representation rgb --finetune_full layer4 -use_jpeg --model_type resnet50 --user_name nmuller -no_save_per_epoch --n_epochs 15
# python finetune_dnns_imagenet.py --input_dir /projects/2/managed_datasets/imagenet --model_type resnet50 --n_epochs 15 --batch_size 150 --limit_classes textureshape --n_processes 18 --image_representation all
python finetune_dnns_imagenet.py --model_type resnet50 --image_representation rgb_zoom-80 --image_size 400 --finetune_full True --n_epochs 15 --input_dir /scratch-nvme/ml-datasets/imagenet/ILSVRC/Data/CLS-LOC/ -no_save_per_epoch --batch_size 150 --limit_classes textureshape --n_processes 18

echo "Finished finetune_dnns_imagenet.py"