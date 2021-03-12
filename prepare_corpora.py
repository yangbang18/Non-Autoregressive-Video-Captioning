''' Handling the data io '''
import argparse
from config import Constants
import wget
import os
import pickle
from misc import utils_corpora

# only words that occur more than this number of times will be put in vocab
word_count_threshold = {
    'MSRVTT': 2,
    'Youtube2Text': 0
}


def main(args):
    func_name = 'preprocess_%s' % args.dataset
    preprocess_func = getattr(utils_corpora, func_name, None)
    if preprocess_func is None:
        raise ValueError('We can not find the function %s in misc/utils_corpora.py' % func_name)

    results = preprocess_func(args.base_pth)
    split = results['split']
    raw_caps_train = results['raw_caps_train']
    raw_caps_all = results['raw_caps_all']
    references = results.get('references', None)

    vid2id = results.get('vid2id', None)
    itoc = results.get('itoc', None)
    split_category = results.get('split_category', None)
    
    # create the vocab
    vocab = utils_corpora.build_vocab(
        raw_caps_train, 
        word_count_threshold[args.dataset],
        sort_vocab=args.sort_vocab,
        )
    itow, captions, itop, pos_tags = utils_corpora.get_captions_and_pos_tags(raw_caps_all, vocab)

    length_info = utils_corpora.get_length_info(captions)
    #next_info = get_next_info(captions, split)

    info = {
        'split': split,                # {'train': [0, 1, 2, ...], 'validate': [...], 'test': [...]}
        'vid2id': vid2id,
        'split_category': split_category,
        'itoc': itoc,
        'itow': itow,                       # id to word
        'itop': itop,                       # id to POS tag
        'length_info': length_info,         # id to length info
    }

    pickle.dump({
            'info': info,
            'captions': captions,
            'pos_tags': pos_tags,
        }, 
        open(args.corpus, 'wb')
    )

    if references is not None:
        pickle.dump(
            references,
            open(args.refs, 'wb')
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='VATEX', type=str)
    parser.add_argument('-sort', '--sort_vocab', default=False, action='store_true')
    args = parser.parse_args()
    if args.dataset.lower() == 'msvd':
        args.dataset = 'Youtube2Text'
    
    assert args.dataset in word_count_threshold.keys()
    
    args.base_pth = os.path.join(Constants.base_data_path, args.dataset)
    args.corpus = os.path.join(args.base_pth, 'info_corpus.pkl')
    args.refs = os.path.join(args.base_pth, 'refs.pkl')
    main(args)
