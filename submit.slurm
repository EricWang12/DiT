#!/bin/bash

# # SBATCH --time=00:15:00
###SBATCH --node=2
#SBATCH --ntasks=1
# 4 or 8 for each gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=train_dit.out
#SBATCH --job-name=train
#SBATCH --error=error.out

# Give this process 1 task and 1 GPU, then assign four CPUs per task
# (so 4 cores overall).  

# If you want two GPUs:
# #SBATCH --gres=gpu:2
# #SBATCH --cpus-per-task=8
# This example, however, only uses one GPU.


# Output some preliminaries before we begin
date
echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

# Load the Python and CUDA modules
# module load python/python-3.8.0-gcc-9.1.0
# module load cuda/cuda-10.2
# module load anaconda
# List the modules that are loaded
# module list

# Have Nvidia tell us the GPU/CPU mapping so we know
nvidia-smi

echo

# Activate the GPU version of PyTorch
# source activate pytorch-1.8.0+cuda10_2
source activate /home/ewang/miniconda3/envs/dit2
conda info 
# Here we are going to run the PyTorch super_resolution example from the PyTorch examples
# GitHub Repository: https://github.com/pytorch/examples/tree/master/super_resolution

# Run PyTorch Training
echo "Training Start:"
torchrun --nnodes=1 --nproc_per_node=1 train.py --data-path datasets/THuman_random_64 --model "DiT-B/2"
echo

# Now we'll try it out by scaling up one of the testing images from the dataset
echo "Super Resolve Start:"
# time python super_resolve.py --input_image dataset/BSDS300/images/test/16077.jpg --model model_epoch_30.pth --output_filename out.png --cuda
echo

# You're done!
echo "Ending script..."
date

