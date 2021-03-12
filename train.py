import json
import os
import opts as opts
import torch
from misc.utils import set_seed
from models import get_model
from misc.run import train_network_all
import warnings
warnings.filterwarnings('ignore')
import random
import pickle
from config import Constants


def get_dir(opt, key, mid_path=''):
    if not opt.get(key, ''):
        return ''
    res = []
    if isinstance(opt[key], list):
        if not opt[key][0]:
            return ''
        for i in range(len(opt[key])):
            res.append(os.path.join(Constants.base_data_path, opt['dataset'], mid_path, opt[key][i]))
    else:
        res = os.path.join(Constants.base_data_path, opt['dataset'], mid_path, opt[key])
    return res


def where_to_save_model(opt):
    return os.path.join(
        Constants.base_checkpoint_path,
        opt['dataset'],
        opt['method'],
        opt['scope']
    )


def print_information(opt, model):
    print(model)
    print('| model {}'.format(opt['method']))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))
    print('dataloader random type: %s' % opt.get('random_type', 'segment_random'))
    print('k best model: %d' % opt.get('k_best_model', 10))
    print('modality: %s' % opt['modality'])
    print('n frames: %d' % opt['n_frames'])
    print('save_checkpoint_every: %d' % opt['save_checkpoint_every'])
    print('max_len: %d' % opt['max_len'])
    print('vocab_size: %d' % opt['vocab_size'])
    print('seed: %d' % opt['seed'])
    print('teacher_path: %s' % opt.get('teacher_path', ""))


def main(opt):
    if opt.get('seed', -1) == -1:
        opt['seed'] = random.randint(1, 65534)
    set_seed(opt['seed'])

    # log files and the best model will be saved at 'checkpoint_path'
    opt["checkpoint_path"] = where_to_save_model(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])

    # get full paths to load features / corpora
    for key in ['feats_a_name', 'feats_m_name', 'feats_i_name', 'feats_o_name', 'feats_t_name'] \
        + ['reference_name', 'info_corpus_name']:
        opt[key[:-5]] = get_dir(opt, key, 'feats' if 'feats' in key else '')
        opt.pop(key)

    # the assignment of 'vocab_size' should be done before defining the model
    opt['vocab_size'] = len(pickle.load(open(opt['info_corpus'], 'rb'))['info']['itow'].keys())
    
    # save training settings
    opt_json = os.path.join(opt["checkpoint_path"], 'opt_info.json')
    with open(opt_json, 'w') as f:
        json.dump(opt, f)
    print('save opt details to %s' % (opt_json))

    model = get_model(opt)
    print_information(opt, model)
    device = torch.device('cuda' if not opt['no_cuda'] else 'cpu')

    if opt.get('pretrained_path', ''):
        print('loading pretrained model from %s' % opt['pretrained_path'])
        model.load_state_dict(torch.load(opt['pretrained_path'])['state_dict'])

    train_network_all(opt, model, device)


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    main(opt)

