import glob
from tqdm import tqdm
import numpy as np
import os
import argparse
import torch
import pretrainedmodels
from pretrainedmodels import utils
import h5py
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.utils.model_zoo as model_zoo
from PIL import Image


def extract_feats(params, model, load_image_fn, C, H, W):
    model.eval()

    frames_path_list = glob.glob(os.path.join(params['frame_path'], '*'))
    db = h5py.File(params['feat_dir'], 'a')

    for frames_dst in tqdm(frames_path_list):
        video_id = frames_dst.split('/')[-1]
        if int(video_id[5:]) > 10000: 
            # MSR-VTT 2017 has 13,000 videos, but we use MSR-VTT 2016 like previous works
            # So we only need to process video0 ~ video9999
            continue
        if video_id in db.keys():
            continue
        
        image_list = sorted(glob.glob(os.path.join(frames_dst, '*.%s' % params['frame_suffix'])))

        if params['k']: 
            images = torch.zeros((params['k'], C, H, W))
            bound = [int(i) for i in np.linspace(0, len(image_list), params['k']+1)]
            for i in range(params['k']):
                idx = (bound[i] + bound[i+1]) // 2
                if params['model'] == 'googlenet':
                    images[i] = load_image_fn.get(image_list[idx])
                else:
                    images[i] = load_image_fn(image_list[idx])
        else:
            images = torch.zeros((len(image_list), C, H, W))
            for i, image_path in enumerate(image_list):
                images[i] = load_image_fn(image_path)

        with torch.no_grad():
            feats = model(images.cuda())
            
        feats = feats.squeeze().cpu().numpy()

        tqdm.write('%s: %s' % (video_id, str(feats.shape)))
        db[video_id] = feats

    db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_path", type=str, required=True, help='the path to load all the frames')
    parser.add_argument("--feat_path", type=str, required=True, help='the path you want to save the features')
    parser.add_argument("--feat_name", type=str, default='', help='the name of the features file, saved in hdf5 format')

    parser.add_argument("--gpu", type=str, default='0', help='set CUDA_VISIBLE_DEVICES environment variable')
    parser.add_argument("--model", type=str, default='resnet101', help='inceptionresnetv2 | resnet101')
    
    parser.add_argument("--k", type=int, default=60, 
        help='uniformly sample k frames from the existing frames and then extract their features. k=0 will extract all existing frames')
    parser.add_argument("--frame_suffix", type=str, default='jpg')
    
    args = parser.parse_args()
    params = vars(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']

    assert os.path.exists(params['frame_path'])
    if not os.path.exists(params['feat_path']):
        os.makedirs(params['feat_path'])
    assert params['feat_name'], 'You should specify the filename'

    params['feat_dir'] = os.path.join(params['feat_path'], params['feat_name'] + ('' if '.hdf5' in params['feat_name'] else '.hdf5'))

    print('Model: %s' % params['model'])
    print('The extracted features will be saved to --> %s' % params['feat_dir'])

    if params['model'] == 'resnet101':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet101(pretrained='imagenet')
    elif params['model'] == 'resnet152':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet152(pretrained='imagenet')
    elif params['model'] == 'resnet18':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet18(pretrained='imagenet')
    elif params['model'] == 'resnet34':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet34(pretrained='imagenet')
    elif params['model'] == 'inceptionresnetv2':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionresnetv2(
            num_classes=1001, pretrained='imagenet+background')
    else:
        print("doesn't support %s" % (params['model']))

    load_image_fn = utils.LoadTransformImage(model)
    model.last_linear = utils.Identity()

    model = model.cuda()
    #summary(model, (C, H, W))
    extract_feats(params, model, load_image_fn, C, H, W)
