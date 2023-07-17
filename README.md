# What do neural networks learn in image classification?
## This is the official website of 'What do neural networks learn in image classification? A frequency shortcut perspective'

### Introduction

### Synthetic Experiment
* Generation of synthetic data
* Metrics

### Experiments on ImageNet-10
* ADCS
* DFMs
* Metrics
* Visualization

### Datasets to be [dowloaded here](https://drive.google.com/drive/folders/1Ug4WDwQWlFJpdks1woSsY6gWuSMYzNSB?usp=sharing)


### Installation requirement
```
python -u Evaluation/test_rank_vit.py  --backbone_model vit --model_path /home/wangs1/HFC/results_224/imagenet10/ViT0\ /version_3/checkpoints/last.ckpt     --patch_size 2   

```

### Computing ADCS


### 


### Training models
```
python -u train.py           --backbone_model ViT  --weight_alpha 1 --lr 0.01 --dataset imagenet10  --band  \   --save_dir results_224/imagenet10/ --decoder 0  --image_size 224     --num_class 10     --p 0   --masks ViT\.pkl     > results_224/imagenet10/ViTen_AF_v0.out
```
