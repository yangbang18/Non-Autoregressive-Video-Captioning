# Non-Autoregressive Coarse-to-Fine Video Captioning

PyTorch Implementation of the paper:

> **Non-Autoregressive Coarse-to-Fine Video Captioning (AAAI2021)**
>
> Bang Yang, Yuexian Zou\*, Fenglin Liu and Can Zhang.
>
> [[arXiv](https://arxiv.org/abs/1911.12018)] or [[aaai.org](https://ojs.aaai.org/index.php/AAAI/article/view/16421)]

## Updates

**[22 Oct 2023]** This repository is no longer maintained. If you want to reproduce the proposed NACF method in a more modern framework (i.e., `PyTorch Lightning`) or with more advanced video features as inputs (e.g., `CLIP` features), please refer to our latest [repository](https://github.com/yangbang18/CARE).

**[30 Aug 2021]** Update the out-of-date links.

**[16 Jun 2021]** Add detailed instuctions for extracting 3D features of videos.

**[12 Mar 2021]** We have released the codebase, preprocessed data and pre-trained models. 

## Main Contribution
1. The first non-autoregressive decoding-based method for video captioning.
2. A generation task of specific part of speech to alleviate the insufficient training of meaningful words.
3. Visual word-driven flexible decoding algorithms for caption generation.

## Content

- [Environment](#environment)
- [Basic Information](#basic-information)
- [Corpora/Feature Preparation](#corpora/feature-preparation)
- [Pretrained Models](#pretrained-models)
- [Training](#training)
- [Testing](#testing)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)


## Environment
we recommend you use Anaconda to create a new environment:
```
conda create -n cap python==3.7
conda activate cap
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install tqdm psutil h5py PyYaml wget
pip install tensorboard==2.2.2 tensorboardX==2.1
```
Here we use `torch 1.6.0` based on `CUDA 10.1`. Another version of `torch` may also work.


## Basic Information
1. supported datasets
- `Youtube2Text` (i.e., MSVD in the paper)
- `MSRVTT`

2. supported methods, whose configurations can be found in [config/methods.yaml](config/methods.yaml)
- `ARB`: autoregressive baseline
- `ARB2`: `ARB` w/ visual word generation 
- `NAB`: non-autoregressive baseline
- `NACF`: `NAB` w/ visual word generation & coarse-grained templates

## Corpora/Feature Preparation
Preprocessed corpora and extracted features can be downloaded in the `VC_data` folder in [GoogleDrive](https://drive.google.com/drive/folders/1oieaYBCNw5sk3fi1cyorYxcMg2LIVXr8?usp=sharing) or [PKU Yun](https://disk.pku.edu.cn:443/link/74125EB20FEA16EF85AE8BE4A1BE0E80).

* Following the structure below to place corpora and feature files:
    ```
    └── base_data_path
        ├── MSRVTT
        │   ├── feats
        │   │   ├── image_resnet101_imagenet_fps_max60.hdf5
        │   │   └── motion_resnext101_kinetics_duration16_overlap8.hdf5
        │   ├── info_corpus.pkl
        │   └── refs.pkl
        └── Youtube2Text
            ├── feats
            │   ├── image_resnet101_imagenet_fps_max60.hdf
            │   └── motion_resnext101_kinetics_duration16_overlap8.hdf5
            ├── info_corpus.pkl
            └── refs.pkl
    ```
**Please remember to modify `base_data_path` in [config/Constants.py](config/Constants.py)**


Alternatively, you can prepare data on your own (**Note:** some dependencies should be installed, e.g., `nltk`, `pretrainedmodels`).
1. Preprocessing corpora:
    ```
    python prepare_corpora.py --dataset Youtube2Text --sort_vocab
    python prepare_corpora.py --dataset MSRVTT --sort_vocab
    ```
2. Feature extraction:
* Downloading all video files of [Youtube2Text (MSVD)](http://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar) and [MSRVTT](http://ms-multimedia-challenge.com/2017/dataset). Alternative links are available at [this url](https://github.com/yangbang18/CARE/blob/master/README_DATA.md).
* Extracting frames
    ```
    python pretreatment/extract_frames_from_videos.py \
    --video_path $path_to_video_files \
    --frame_path $path_to_save_frames \
    --video_suffix mp4 \
    --frame_suffix jpg \
    --strategy 1 \
    --fps 5 \
    --vframes 60
    ```
    * `--video_suffix` is `mp4` For MSRVTT while `avi` for Youtube2Text. 
    * When extracting frames for Youtube2Text, please pass the argument `--info_path $base_data_path/Youtube2Text/info_corpus.pkl`, which will map video names to vids (e.g., `video0, ..., video1969`).
* Extracting image features
    ```
    python pretreatment/extract_image_feats_from_frames.py \
    --frame_path $path_to_load_frames \
    --feat_path $base_data_path/dataset_name \
    --feat_name image_resnet101_imagenet_fps_max60.hdf5 \
    --model resnet101 \
    --k 0 \
    --frame_suffix jpg \
    --gpu 3
    ```
* Extracting motion features (**Note**: we should extract all frames of videos in advance)
    ```
    python pretreatment/extract_frames_from_videos.py \
    --video_path $path_to_video_files \
    --frame_path $path_to_save_frames \
    --video_suffix mp4 \
    --frame_suffix jpg \
    --strategy 0
    ```
    * The rest part please refer to [this repository](https://github.com/yangbang18/video-classification-3d-cnn).

## Pretrained Models
We have provided the captioning models pre-trained on Youtube2Text (MSVD) and MSRVTT. Please refer to the `experiments` folder in [GoogleDrive](https://drive.google.com/drive/folders/1oieaYBCNw5sk3fi1cyorYxcMg2LIVXr8?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1ZMuoH_QDYjdXT2wVh5fjaw) (extract code `lkmu`).

* Following the structure below to place pre-trained models:
    ```
    └── base_checkpoint_path
        ├── MSRVTT
        │   ├── ARB
        │   │   └── best.pth.tar
        │   ├── ARB2
        │   │   └── best.pth.tar
        │   ├── NAB
        │   │   └── best.pth.tar
        │   └── NACF
        │       └── best.pth.tar
        └── Youtube2Text
    ```

**Please remember to modify `base_checkpoint_path` in [config/Constants.py](config/Constants.py)**



## Training 

```
python train.py --default --dataset `dataset_name` --method `method_name`
```
**Keypoints:** 
- The `--default` argument specifies some necessary settings related to the dataset and method. With this argument, `ARB` should be trained first before the training of `NAB` or `NACF` because `ARB` by default serves as a teacher to re-score the captions generated by `NAB` or `NACF`.
- After training, `train.py` will automatcally call `translate.py` to calculate the performance of the best model you have trained on validation and test sets. If you wanna teriminate this automatic call, pass the `--no_test` argument.
- To run the script on cpu, run with the `--no_cuda` argument.

**Examples:**
```
python train.py --default --dataset MSRVTT --method ARB
python train.py --default --dataset MSRVTT --method NACF
```

## Testing 
```
python translate.py --default --dataset `dataset_name` --method `method_name`
```

**Examples:**
```
# NACF w/o ct
python translate.py --default --dataset MSRVTT --method NACF

# NACF w/ ct
python translate.py --default --dataset MSRVTT --method NACF --use_ct

# NACF using different algorithms
python translate.py --default --dataset MSRVTT --method NACF --use_ct --paradigm mp
python translate.py --default --dataset MSRVTT --method NACF --use_ct --paradigm ef
python translate.py --default --dataset MSRVTT --method NACF --use_ct --paradigm l2r
```


## Citation
Please **[★star]** this repo and **[cite]** the following paper if you feel our code or models useful to your research:

```
@inproceedings{yang2021NACF,
  title={Non-Autoregressive Coarse-to-Fine Video Captioning}, 
  author={Yang, Bang and Zou, Yuexian and Liu, Fenglin and Zhang, Can},     
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={4},
  pages={3119-3127},
  year={2021}
}
```

## Acknowledgements
Code of the decoding part is based on [facebookresearch/Mask-Predict](https://github.com/facebookresearch/Mask-Predict).
