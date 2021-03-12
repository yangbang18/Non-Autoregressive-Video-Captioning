from config import Constants
import json
from tqdm import tqdm
import string
import nltk
from collections import defaultdict
import os
import wget
import pickle
import numpy as np


def preprocess_MSRVTT(base_path):
    os.makedirs(base_path, exist_ok=True)
    # the official url is http://ms-multimedia-challenge.com/2016
    url = "https://github.com/ybCliff/VideoCaptioning/releases/download/v1.0/videodatainfo_2016.json"
    input_json = os.path.join(base_path, 'videodatainfo.json')
    if not os.path.exists(input_json):
        wget.download(url, out=input_json)

    json_data = json.load(open(input_json, 'r'))
    sentences = json_data['sentences']
    videos = json_data['videos']

    split = {'train': [], 'validate': [], 'test': []}
    for v in videos:
        split[v['split']].append(int(v['id']))

    raw_caps_all = defaultdict(list)
    raw_caps_train = defaultdict(list)
    references = defaultdict(list)

    for item in tqdm(sentences):
        vid = item['video_id']
        tokens = [
            token.lower() \
            for token in item['caption'].split() \
            if token not in string.punctuation
        ]

        raw_caps_all[vid].append(tokens)

        if int(vid[5:]) in split['train']:
            raw_caps_train[vid].append(tokens)

        references[vid].append({
            'image_id': vid, 
            'cap_id': len(references[vid]), 
            'caption': ' '.join(tokens)
        })

    itoc = {}
    split_category = {'train': defaultdict(list), 'validate': defaultdict(list), 'test': defaultdict(list)}
    for item in videos:
        itoc[item["id"]] = item["category"]
        split_category[item['split']][int(item["category"])].append(int(item['id']))

    return {
        'split': split, 
        'raw_caps_train': raw_caps_train, 
        'raw_caps_all': raw_caps_all, 
        'references': references,
        'itoc': itoc,
        'split_category': split_category
    }


def preprocess_Youtube2Text(base_path):
    os.makedirs(base_path, exist_ok=True)

    # the official url is https://www.cs.utexas.edu/users/ml/clamp/videoDescription
    # but we found that the download link no longer available
    url = "https://github.com/ybCliff/VideoCaptioning/releases/download/1.0/msvd_refs.pkl"
    refs_pickle = os.path.join(base_path, 'refs.pkl')
    if not os.path.exists(refs_pickle):
        wget.download(url, out=refs_pickle)
    
    url = "https://github.com/ybCliff/VideoCaptioning/files/3764071/youtube_mapping.txt"
    mapping_txt = os.path.join(base_path, 'youtube_mapping.txt')
    if not os.path.exists(mapping_txt):
        wget.download(url, out=mapping_txt)
    mapping_info = open(mapping_txt, 'r').read().strip().split('\n')
    
    vid2id = {}
    for line in mapping_info:
        _id, vid = line.split()
        vid = vid.replace('vid', 'video')
        vid2id[vid] = _id

    split = {
        'train': [i for i in range(1200)],
        'validate': [i for i in range(1200, 1300)],
        'test': [i for i in range(1300, 1970)]
    }

    raw_caps_all = defaultdict(list)
    raw_caps_train = {}
    
    refs = pickle.load(open(refs_pickle, 'rb'))
    for vid in tqdm(refs.keys()):
        num = int(vid[5:]) # e.g. 'video999', num = 999
        for item in refs[vid]:
            tokens = item['caption'].lower().split()
            raw_caps_all[vid].append(tokens)

        if num in split['train']:
            raw_caps_train[vid] = raw_caps_all[vid]

    return {
        'split': split, 
        'raw_caps_train': raw_caps_train, 
        'raw_caps_all': raw_caps_all, 
        'vid2id': vid2id,
    }


def build_vocab(train_vid2caps, count_thr, sort_vocab=False):
    '''
        args:
            - train_vid2caps (dict): vid-captions pairs from the training set
            - count_thr (int): words that appear <= count_thr will be filtered
        return:
            - vocab (list): vocabulary
    '''

    # count up the number of words
    counts = {}
    for vid, caps in train_vid2caps.items():
        for cap in caps:
            for w in cap:
                counts[w] = counts.get(w, 0) + 1

    bad_words = [w for w, n in counts.items() if n <= count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    total_words = sum(counts.values())

    print('- The number of bad words: %d/%d = %.2f%%' %
          (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('- The number of the vocabulary: %d' % (len(counts) - len(bad_words)))
    print('- The number of UNKs: %d/%d = %.2f%%' %
          (bad_count, total_words, bad_count * 100.0 / total_words))

    candidate_vocab = [(w, n) for w, n in counts.items() if n > count_thr]
    if sort_vocab:
        print('- Sort the vocabulary by the frequency of words, larger the first')
        candidate_vocab = sorted(candidate_vocab, key=lambda x: -x[1])

    vocab = [w for w, _ in candidate_vocab]

    assert len(vocab) == len(counts) - len(bad_words)

    print('- Top 100 words:')
    print(vocab[:100])
    return vocab


def get_length_info(captions):
    length_info = {}
    max_length = 50

    for vid, caps in captions.items():
        length_info[vid] = [0] * max_length
        for cap in caps:
            length = len(cap) - 2 # exclude <bos>, <eos>
            if length >= max_length:
                continue
            length_info[vid][length] += 1

    return length_info


def get_captions_and_pos_tags(raw_caps_all, vocab):
    itow = {i + 6: w for i, w in enumerate(vocab)}
    itow[Constants.PAD] = Constants.PAD_WORD
    itow[Constants.UNK] = Constants.UNK_WORD
    itow[Constants.BOS] = Constants.BOS_WORD
    itow[Constants.EOS] = Constants.EOS_WORD
    itow[Constants.MASK] = Constants.MASK_WORD
    itow[Constants.VIS] = Constants.VIS_WORD

    wtoi = {w: i for i, w in itow.items()}  # inverse table

    ptoi = {}
    ptoi[Constants.PAD_WORD] = Constants.PAD
    ptoi[Constants.UNK_WORD] = Constants.UNK
    ptoi[Constants.BOS_WORD] = Constants.BOS
    ptoi[Constants.EOS_WORD] = Constants.EOS
    ptoi[Constants.MASK_WORD] = Constants.MASK
    ptoi[Constants.VIS_WORD] = Constants.VIS
    tag_start_i = 6

    captions = defaultdict(list)
    pos_tags = defaultdict(list)
    for vid, caps in tqdm(raw_caps_all.items()):
        for cap in caps:
            tag_res = nltk.pos_tag(cap)

            caption_id = [Constants.BOS]
            tagging_id = [Constants.BOS]

            for w, t in zip(cap, tag_res):
                assert t[0] == w
                tag = Constants.pos_tag_mapping[t[1]]

                if w in wtoi.keys():
                    caption_id += [wtoi[w]]
                    if tag not in ptoi.keys():
                        ptoi[tag] = tag_start_i
                        tag_start_i += 1
                    tagging_id += [ptoi[tag]]
                else:
                    caption_id += [Constants.UNK]
                    tagging_id += [Constants.UNK]

            caption_id += [Constants.EOS]
            tagging_id += [Constants.EOS]

            captions[vid].append(caption_id)
            pos_tags[vid].append(tagging_id)

    itop = {i: t for t, i in ptoi.items()}
    return itow, captions, itop, pos_tags
