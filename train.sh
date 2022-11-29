#!/bin/bash
# 
#SBATCH --job-name=test
#SBATCH --output=res/res_%j.txt  # output file
#SBATCH -e err/res_%j.err        # File to which STDERR will be written
#SBATCH --partition=gypsum-m40 # Partition to submit to 
#SBATCH --gpus-per-node=1 
#
#SBATCH --ntasks=1
#SBATCH --time=1-10:00       # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=10000    # Memory in MB per cpu allocated

hostname
python main.py --dataset_name cifar10 --dataset_type image --original_label default --exp model_train --unlearning_method sisa  --original_model scnn
exit
~                                                                                                                                                                                  
