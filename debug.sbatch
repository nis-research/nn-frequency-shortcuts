#!/bin/bash
#SBATCH -J Training
#SBATCH -c 2
#SBATCH -p dmb
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL 
#load modules
module load nvidia/cuda-11.7
module load nvidia/cuda-11.x_cudnn-8.6
module load anaconda3/2022.05

#source activate


echo $SLURM_NODELIST

cd nn-frequency-shortcuts
source activate debugenv

python -u Evaluation/test_rank.py  --backbone_model resnet18 --model_path results/imagenet10/resnet18/version_0/checkpoints/last.ckpt     --patch_size  7  

# python -u train.py --backbone_model resnet18  --lr 0.01 --dataset imagenet10   --save_dir results/imagenet10/  --image_size 224     --num_class 10            
# python -u train.py --backbone_model alex  --lr 0.01 --save_dir results/syn1/  --dataset synthetic   --num_class 4  --special _complex_special_1_par 
# python -u Evaluation/compute_ADCS.py

conda deactivate
