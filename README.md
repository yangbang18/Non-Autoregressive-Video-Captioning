# NACF: Non-Autoregressive Coarse-to-Fine Video Captioning
Official codes for our [AAAI2021 paper](https://arxiv.org/abs/1911.12018).

[2021-01-20] Codes are comming soon.

## Environment
we recommend you to use Anaconda to create a new environment:
```
conda create -n cap python==3.7
conda activate cap
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install tqdm psutil h5py PyYaml
pip install tensorboard==2.2.2 tensorboardX==2.1
```
Here we use `torch 1.6.0` based on `CUDA 10.1`. Another version of `torch` may also work.


## Basic Information
1. supported dataset list
- `Youtube2Text` (i.e., MSVD in the paper)
- `MSRVTT`

2. supported method list
- `ARB`: autoregressive baseline
- `ARB2`: `ARB` w/ visual word generation 
- `NAB`: non-autoregressive baseline
- `NACF`: `NAB` w/ visual word generation & coarse-grained templates

## How to Run
Training
```
CUDA_VISIBLE_DEVICES=0 python train.py \
--dataset `dataset_name` \
--method `method_name` \
--default
```

Evaluation
```
CUDA_VISIBLE_DEVICES=0 python translate.py \
--dataset `dataset_name` \
--method `method_name` \
--default
```

