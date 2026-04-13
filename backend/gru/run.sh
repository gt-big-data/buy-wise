#!/bin/bash
#SBATCH --job-name=gru_train_test
#SBATCH --account=paceship-buywise
#SBATCH --output=logs/gru_%j.out
#SBATCH --error=logs/gru_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

module load anaconda3

conda activate /storage/home/hcoda1/2/$USER/conda_envs/myenv

cd ~/model_gru

echo "starting training..."
python train.py

echo "training done, starting testing..."
python test.py

echo "all done"