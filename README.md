## Official website of 'What do neural networks learn in image classification? A frequency shortcut perspective (Paper accepted at ICCV2023)'
#### The paper is available on [arXiv]().

### Introduction

Frequency analysis is useful for understanding the mechanisms of representation learning in neural networks (NNs). Most research in this area focuses on the learning dynamics of NNs for regression tasks, while little for classification. This study empirically investigates the latter and expands the understanding of frequency shortcuts. First, we perform experiments on synthetic datasets, designed to have a bias in different frequency bands. Our results demonstrate that NNs tend to find simple solutions for classification, and what they learn first during training depends on the most distinctive frequency characteristics, which can be either low- or high-frequencies. Second, we confirm this phenomenon on natural images. We
propose a metric to measure class-wise frequency characteristics and a method to identify frequency shortcuts. The results show that frequency shortcuts can be texturebased
or shape-based, depending on what best simplifies the objective. Third, we validate the transferability of frequency shortcuts on out-of-distribution (OOD) test
sets. Our results suggest that frequency shortcuts can be transferred across datasets and cannot be fully avoided by larger model capacity and data augmentation. We recommend
that future research should focus on effective training schemes mitigating frequency shortcut learning.

<p align="center"><img src="figures/teaser.png" width="500"></p>


### Quick start

* Clone this repository:
```
git clone
cd 
```

* Installation
```
pip install -r requirements.txt
```
* Datasets can be [dowloaded](https://drive.google.com/drive/folders/1Ug4WDwQWlFJpdks1woSsY6gWuSMYzNSB?usp=sharing) here

* Computing DFM
```
python -u Evaluation/test_rank_vit.py  --backbone_model vit --model_path /home/wangs1/HFC/results_224/imagenet10/ViT0\ /version_3/checkpoints/last.ckpt     --patch_size 2   

```
* Computing ADCS

* Visualization

* Training models
```
python -u train.py           --backbone_model ViT  --weight_alpha 1 --lr 0.01 --dataset imagenet10  --band  \   --save_dir results_224/imagenet10/ --decoder 0  --image_size 224     --num_class 10     --p 0   --masks ViT\.pkl     > results_224/imagenet10/ViTen_AF_v0.out
```

## Citation

```
@misc{
}
```

