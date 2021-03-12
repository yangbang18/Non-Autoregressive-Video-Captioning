import torch
import torch.nn as nn
import numpy as np
import random
import os
from config import Constants
from models import get_model
from collections import OrderedDict


def set_seed(seed=2019):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_sentence(hyp, vocab, break_words=[Constants.EOS, Constants.PAD], skip_words=[]):
    sent = []
    for word_id in hyp:
        if word_id in skip_words:
            continue
        if word_id in break_words:
            break
        word = vocab[word_id]
        sent.append(word)
    return ' '.join(sent)


def get_dict_mapping(opt, teacher_opt):
    if teacher_opt is None:
        return {}
    if teacher_opt['vocab_size'] == opt['vocab_size']:
        return {}

    info = json.load(open(opt["info_json"]))
    vocab = info['ix_to_word']

    teacher_info = json.load(open(teacher_opt["info_json"]))
    teacher_vocab = teacher_info['ix_to_word']
    teacher_w2ix = teacher_info['word_to_ix']
    if vocab == teacher_vocab:
        return {}

    dict_mapping = {}
    for k, v in vocab.items():
        dict_mapping[int(k)] = int(teacher_w2ix[v])
    return dict_mapping


def load_model_and_opt(checkpoint_path, device, return_other_info=False):
    checkpoint = torch.load(checkpoint_path)
    opt = checkpoint['settings']
    model = get_model(opt)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    if not return_other_info:
        return model, opt
    checkpoint.pop('state_dict')
    return model, opt, checkpoint


def remove_repeat_n_grame(sent, n):
    length = len(sent)
    rec = {}
    result_sent = []
    for i in range(length-n+1):
        key = ' '.join(sent[i:i+n])
        if key in rec.keys():
            dis = i - rec[key] - n
            if dis in [0,1]:
                result_sent += sent[:i-dis]
                if i+n <length:
                    result_sent += sent[i+n:]
                return result_sent, False
        else:
            rec[key] = i
    return sent, True


def duplicate(sent):
    sent = sent.split(' ')
    res = {}
    for i in range(4, 0, -1):
        jud = False
        while not jud:
            sent, jud = remove_repeat_n_grame(sent, i)
            if not jud:
                res[i] = res.get(i, 0) + 1
            else:
                break
    res_str = []
    for i in range(1, 5):
        res_str.append('%d-gram: %d' % (i, res.get(i, 0)))
    return ' '.join(sent), '\t'.join(res_str)


def cal_gt_n_gram(data, vocab, splits, n=1):
    gram_count = {}
    gt_sents = {}
    for i in splits['train']:
        k = 'video%d'% int(i)
        caps = data[k]
        for tmp in caps:
            cap = [vocab[wid] for wid in tmp[1:-1]]
            gt_sents[' '.join(cap)] = gt_sents.get(' '.join(cap), 0) + 1
            for j in range(len(cap)-n+1):
                key = ' '.join(cap[j:j+n])
                gram_count[key] = gram_count.get(key, 0) + 1
    return gram_count, gt_sents


def cal_n_gram(data, n=1):
    gram_count = {}
    sents = {}
    ave_length, count = 0, 0
    for k in data.keys():
        for i in range(len(data[k])):
            sents[data[k][i]['caption']] = sents.get(data[k][i]['caption'], 0) + 1
            cap = data[k][i]['caption'].split(' ')
            ave_length += len(cap)
            count += 1
            for j in range(len(cap)-n+1):
                key = ' '.join(cap[j:j+n])
                gram_count[key] = gram_count.get(key, 0) + 1
    return gram_count, sents, ave_length/count, count


def analyze_length_novel_unique(gt_data, data, vocab, splits, n=1, calculate_novel=True):
    novel_count = 0
    hy_res, hy_sents, ave_length, hy_count = cal_n_gram(data, n)
    if calculate_novel:
        gt_res, gt_sents = cal_gt_n_gram(gt_data, vocab, splits, n)
        for k1 in hy_sents.keys():
            if k1 not in gt_sents.keys():
                novel_count += 1

    novel = novel_count / hy_count
    unique = len(hy_sents.keys()) / hy_count
    vocabulary_usage = len(hy_res.keys())

    gram4, _, _, _ = cal_n_gram(data, n=4)
    return ave_length, novel, unique, vocabulary_usage, hy_res, len(gram4)


def get_words_with_specified_tags(word_to_ix, seq, index_set, demand=['NOUN', 'VERB'], ignore_words=['is', 'are', '<mask>']):
    import nltk
    assert isinstance(index_set, set)
    res = nltk.pos_tag(seq.split(' '))
    for w, t in res:
        if Constants.pos_tag_mapping[t] in demand and w not in ignore_words:
            index_set.add(word_to_ix[w])


def load_satisfied_weights(model, checkpoint_path, str_mapping={}, skip_keys=[], strict=False):
    model_dict = model.state_dict()
    checkpoint_dict = torch.load(checkpoint_path)['state_dict']

    new_state_dict = model_dict
    str_mapping_keys = list(str_mapping.keys())
    str_mapping_values = list(str_mapping.values())

    def check(now_key, all_str):
        for i, item in enumerate(all_str):
            if item in now_key:
                return i
        return -1

    success = 0
    for k in model_dict.keys():
        if k in skip_keys:
            continue
        index = check(now_key=k, all_str=str_mapping_keys)
        if index != -1:
            src, trg = str_mapping_keys[index], str_mapping_values[index]
            key = k.replace(src, trg)
        else:
            key = k

        if key in checkpoint_dict:
            new_state_dict[k] = checkpoint_dict[key]
            success += 1
        else:
            assert not strict, 'key {}/{} can not be found in the checkpoint'.format(k, key)
            new_state_dict[k] = model_dict[k]
    print('Successfully loading {}/{} parameters'.format(success, len(new_state_dict)))
    
    model.load_state_dict(new_state_dict)
    return model


def save_checkpoint(state, is_best, filepath='./', filename='checkpoint.pth.tar', best_model_name='best.pth.tar'):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    save_path = os.path.join(filepath, filename) 
    torch.save(state, save_path)
    if is_best:
        best_path = os.path.join(filepath, best_model_name)
        shutil.copyfile(save_path, best_path)


def enlarge(info, beam_size):
    bsz, *rest_shape = info.shape
    if len(rest_shape) == 2:
        info = info.unsqueeze(1).repeat(1, beam_size, 1, 1)
    elif len(rest_shape) == 1:
        info = info.unsqueeze(1).repeat(1, beam_size, 1)
    else:
        info = info.unsqueeze(1).repeat(1, beam_size)
    return info.view(bsz * beam_size, *rest_shape)


def auto_enlarge(info, beam_size):
    if isinstance(info, list):
        if isinstance(info[0], tuple):
            return [
                tuple([enlarge(_, beam_size) for _ in item])
                for item in info
            ]
        else:
            return [enlarge(item, beam_size) for item in info]
    else:
        if isinstance(info, tuple):
            return tuple([enlarge(item, beam_size) for item in info])
        else:
            return enlarge(info, beam_size)
