#!/bin/bash
#SBATCH -J Training
#SBATCH -c 2
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL 
#load modules
module load nvidia/cuda-11.3
module load nvidia/cuda-11.3_cudnn-8.2
module load anaconda3/2022.05

#source activate


echo $SLURM_NODELIST

cd nn-frequency-shortcuts
source activate debugenv
# python -u train.py --backbone_model ViT   --lr 0.01 --dataset imagenet10  --save_dir results_224/imagenet10/  --image_size 224     --num_class 10   > results_224/imagenet10/ViTen_AF_v0.out

# python -u train.py --backbone_model alex  -lr 0.01 --dataset imagenet10   --save_dir /home/wangs1/HFC/results/imagenet10/  --image_size 224     --num_class 10             > /home/wangs1/HFC/results/imagenet10/alexen_AF_v1.out


python -u train.py --backbone_model alex  --lr 0.01 --save_dir results/retrain/synthetic_complex_special_1_par/  --dataset synthetic  --special _complex_special_1_par  --num_class 4 


conda deactivate
