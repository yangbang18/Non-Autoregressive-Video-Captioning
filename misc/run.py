import sys
import os, shutil
import torch
from torch.nn.utils import clip_grad_value_
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import json
from misc.cocoeval import suppress_stdout_stderr, COCOScorer
from dataloader import VideoDataset

from models.Translator import Translator
from config import Constants

from .optim import get_optimizer
from .crit import get_criterion, get_criterion_during_evaluation
from .logger import CsvLogger, k_PriorityQueue
from .utils import (
    to_sentence, 
    get_words_with_specified_tags, 
    duplicate,
    analyze_length_novel_unique,
    load_model_and_opt,
    load_satisfied_weights,
    save_checkpoint,
)
from tensorboardX import SummaryWriter

from collections import defaultdict
import pickle
import time


def prepare_data(data, key, device):
    if data.get(key, None) is None:
        return None
    return data[key].to(device)


def get_forword_results(opt, model, data, device, only_data=False, vocab=None, **kwargs):
    category, labels = map(
            lambda x: x.to(device), 
            [data['category'], data['labels']]
        )

    feats = []
    for char in opt['modality'].lower():
        feat = data.get("feats_%s" % char, None)
        assert feat is not None
        feats.append(feat.to(device))

    if opt['visual_word_generation']:
        tokens = [data['tokens_1'].to(device), data['tokens'].to(device)]
    else:
        tokens = data['tokens'].to(device)

    if only_data:
        # for evaluation
        results = model.encode(feats=feats)
    else:
        results = model(
            feats=feats,
            tgt_tokens=tokens, 
            category=category,
            opt=opt,
            vocab=vocab,
            **kwargs
            )

    if opt['decoding_type'] == 'NARFormer':
        results[Constants.mapping['length'][1]] = prepare_data(data, 'length_target', device)
        start_index = 0
    else:
        start_index = 1

    if opt['visual_word_generation']:    
        results[Constants.mapping['lang'][1]] = [
            data['labels_1'].to(device)[:, start_index:],
            labels[:, start_index:]
        ]
    else:
        results[Constants.mapping['lang'][1]] = labels[:, start_index:]

    if only_data:
        return results, category, labels
    return results


def get_loader(opt, mode, print_info=False, specific=-1, **kwargs):
    dataset = VideoDataset(opt, mode, print_info, specific=specific, **kwargs)
    batch_size = kwargs.get('batch_size', opt['batch_size'])
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True if mode=='train' else False
    )


def run_eval(
        opt, model, crit, loader, vocab, device, 
        json_path='', json_name='', scorer=COCOScorer(), 
        teacher_model=None, dict_mapping={}, 
        no_score=False, print_sent=False, analyze=False, 
        collect_best_candidate_iterative_results=False, collect_path=None, 
        extra_opt={}, summarywriter=None, global_step=0):
    opt.update(extra_opt)
    model.eval()
    if teacher_model is not None:
        teacher_model.eval()

    gt_captions = loader.dataset.get_references()
    pred_captions = defaultdict(list)

    opt['collect_best_candidate_iterative_results'] = collect_best_candidate_iterative_results
    translator = Translator(model=model, opt=opt, teacher_model=teacher_model, dict_mapping=dict_mapping)

    best_candidate_sents = defaultdict(list)
    best_candidate_score = defaultdict(list)

    best_ar_sent = []
    all_time = 0

    if crit is not None:
        crit.reset_loss_recorder()
    
    collect_ar_flag = (opt['decoding_type'] == 'ARFormer' and collect_best_candidate_iterative_results)

    for data in tqdm(loader, ncols=150, leave=False):
        with torch.no_grad():
            encoder_outputs, category, labels = get_forword_results(opt, model, data, device=device, only_data=True, vocab=vocab)
            if crit is not None:
                _ = crit.get_loss(encoder_outputs)

            if teacher_model is not None:
                teacher_encoder_outputs, *_ = get_forword_results(opt, teacher_model, data, device=device, only_data=True, vocab=vocab)
            else:
                teacher_encoder_outputs = None

            if opt['batch_size'] == 1:
                start_time = time.time()
            all_hyp, all_scores = translator.translate_batch(encoder_outputs, category, labels, vocab, teacher_encoder_outputs=teacher_encoder_outputs)
            if opt['batch_size'] == 1:
                all_time += (time.time() - start_time)

            if isinstance(all_hyp, torch.Tensor):
                if len(all_hyp.shape) == 2:
                    all_hyp = all_hyp.unsqueeze(1)
                all_hyp = all_hyp.tolist()
            if isinstance(all_scores, torch.Tensor):
                if len(all_scores.shape) == 2:
                    all_scores = all_scores.unsqueeze(1)
                all_scores = all_scores.tolist()

            video_ids = np.array(data['video_ids']).reshape(-1)

        for k, hyps in enumerate(all_hyp):
            video_id = video_ids[k]
            if not no_score: 
                assert len(hyps) == 1

            for j, hyp in enumerate(hyps):
                sent = to_sentence(hyp, vocab)
                if opt.get('duplicate', False) and opt['decoding_type'] == 'NARFormer':
                    sent, _ = duplicate(sent)
                if print_sent:
                    tqdm.write(video_id + ': ' + sent)

                if not collect_ar_flag:
                    # for evaluation
                    pred_captions[video_id].append({'image_id': video_id, 'caption': sent})
                else:
                    # for collection
                    pred_captions[video_id].append({'caption': sent, 'score': all_scores[k][j]})

        if collect_best_candidate_iterative_results and not collect_ar_flag:
            assert isinstance(all_scores, tuple)
            all_sents = all_scores[0].tolist()
            all_score = all_scores[1].tolist()

            if len(video_ids) != len(all_sents):
                video_ids = np.array(data['video_ids'])[:, np.newaxis].repeat(opt['length_beam_size'], axis=1).reshape(-1)
                assert len(video_ids) == len(all_sents)

            for k, (hyps, scores) in enumerate(zip(all_sents, all_score)):
                video_id = video_ids[k]
                pre_sent_len = 0
                assert len(hyps) == len(scores)  

                for j, (hyp, score) in enumerate(zip(hyps, scores)):
                    sent = to_sentence(hyp, vocab)

                    if not pre_sent_len: 
                        pre_sent_len = len(sent.split(' '))
                    else:
                        assert len(sent.split(' ')) == pre_sent_len

                    tqdm.write(('%10s' % video_id) + '(iteration %d Length %d): ' % (j, len(sent.split(' '))) + sent)
 
                    best_candidate_sents[video_id].append(sent)
                    best_candidate_score[video_id].append(score)

    if collect_best_candidate_iterative_results:
        assert collect_path is not None
        if not collect_ar_flag:
            pickle.dump(
                    [best_candidate_sents, best_candidate_score],
                    open(collect_path, 'wb')
                )
        else:
            pickle.dump(pred_captions, open(collect_path, 'wb'))

    if opt['batch_size'] == 1:
        latency = all_time/len(loader)
        print(latency, len(loader))            

    res = {}
    if analyze:
        ave_length, novel, unique, usage, hy_res, gram4 = analyze_length_novel_unique(loader.dataset.captions, pred_captions, vocab, splits=loader.dataset.splits, n=1)
        res.update({'ave_length': ave_length, 'novel': novel, 'unique': unique, 'usage': usage, 'gram4': gram4})   

    if not no_score:
        with suppress_stdout_stderr():
            valid_score, detail_scores = scorer.score(gt_captions, pred_captions, pred_captions.keys())

        res.update(valid_score)
        metric_sum = opt.get('metric_sum', [1, 1, 1, 1])
        candidate = [res["Bleu_4"], res["METEOR"], res["ROUGE_L"], res["CIDEr"]]
        res['Sum'] = sum([item for index, item in enumerate(candidate) if metric_sum[index]])
        if crit is not None:
            names, metrics = crit.get_loss_info()
            for n, m in zip(names, metrics):
                res[n] = m
    
    if summarywriter is not None:
        for k, v in res.items():
            summarywriter.add_scalar(k, v, global_step=global_step)

    if json_path:
        if not os.path.exists(json_path):
            os.makedirs(json_path)

        with open(os.path.join(json_path, json_name), 'w') as prediction_results:
            json.dump({"predictions": pred_captions, "scores": valid_score}, prediction_results)
            prediction_results.close()

    return res


def run_train(opt, model, crit, optimizer, loader, device, logger=None, epoch=-1, return_all_info=False, **kwargs):
    model.train()
    crit.reset_loss_recorder()
    vocab = loader.dataset.get_vocab()

    for data in tqdm(loader, ncols=150, leave=False):
        optimizer.zero_grad()
        results = get_forword_results(opt, model, data, device=device, only_data=False, vocab=vocab, **kwargs)
        loss = crit.get_loss(results, epoch=epoch)
        loss.backward()

        clip_grad_value_(model.parameters(), opt['grad_clip'])
        optimizer.step()

    name, loss_info = crit.get_loss_info()
    if logger is not None:
        logger.write_text('\t'.join(['%10s: %05.3f' % (item[0], item[1]) for item in zip(name, loss_info)]))

    if return_all_info:
        return loss_info
    return loss_info[0]


def train_network_all(opt, model, device, **kwargs):
    if opt.get('load_teacher_weights', False):
        assert opt.get('teacher_path', None) is not None
        model = load_satisfied_weights(
            model=model, 
            checkpoint_path=opt['teacher_path'],
            str_mapping={'decoder.bert.': 'decoder.'}
        )
    
    model.to(device)
    summarywriter = SummaryWriter(os.path.join(opt['checkpoint_path'], 'trainval'))
    optimizer = get_optimizer(opt, model, summarywriter=summarywriter)
    crit = get_criterion(opt, summarywriter=summarywriter)
    crit_eval = get_criterion_during_evaluation(opt)

    if opt.get('with_teacher', False) and opt['method'] in ['NAB', 'NACF']:
        assert opt.get('teacher_path', None) is not None
        teacher_model, _ = load_model_and_opt(opt['teacher_path'], device)
    else:
        teacher_model = None

    folder_path = os.path.join(opt["checkpoint_path"], 'tmp_models')
    best_model = k_PriorityQueue(
        k_best_model = opt.get('k_best_model', 1), 
        folder_path = folder_path,
        standard = opt.get('standard', ['METEOR', 'CIDEr'])
        )

    train_loader = get_loader(opt, 'train', print_info=False, **kwargs)
    vali_loader = get_loader(opt, 'validate', print_info=False)
    test_loader = get_loader(opt, 'test', print_info=False)
    vocab = vali_loader.dataset.get_vocab()

    logger = CsvLogger(
        filepath=opt["checkpoint_path"], 
        filename='trainning_record.csv', 
        fieldsnames=[
            'epoch', 'train_loss', 
            'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 
            'METEOR', 'ROUGE_L', 'CIDEr', 'Sum']
            + crit.get_fieldsnames()
        )

    start_epoch = 0
    for epoch in tqdm(range(opt['epochs']), ncols=150, leave=False):
        if epoch < start_epoch: 
            continue 

        train_loader.dataset.shuffle()

        logger.write_text("epoch %d lr=%g (ss_prob=%g)" % (epoch, optimizer.get_lr(), opt.get('teacher_prob', 1)))
        # training
        train_loss = run_train(opt, model, crit, optimizer, train_loader, device, logger=logger, epoch=epoch)

        optimizer.epoch_update_learning_rate()

        if (epoch+1) > opt['start_eval_epoch'] and (epoch+1) % opt["save_checkpoint_every"] == 0:
            res = run_eval(opt, model, crit_eval, vali_loader, vocab, device, teacher_model=teacher_model, analyze=True, summarywriter=summarywriter, global_step=epoch)
            res['train_loss'] = train_loss
            res['epoch'] = epoch
            logger.write(res)

            save_checkpoint(
                    {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'validate_result': res, 'settings': opt}, 
                    False, 
                    filepath=opt["checkpoint_path"], 
                    filename='checkpoint.pth.tar'
                )
            
            model_name = 'model_%04d.pth.tar' % res['epoch']
            model_path = os.path.join(folder_path, model_name)
            not_break, info = best_model.check(res, opt, model_path, model_name)
            if not not_break:
                # reach the tolerence
                break
            logger.write_text(info)
    
    if not opt.get('no_test', False):
        model = model.to('cpu')
        del model
        del optimizer
        torch.cuda.empty_cache()
        os.system('python translate.py --default --method {} --dataset {} --record --scope {} --field {} --val_and_test --use_ct'.format(
            opt['method'], opt['dataset'], opt['scope'] if opt['scope'] else '\"\"', ' '.join(opt['field']))
        )

    if opt['k_best_model'] > 1:
        shutil.rmtree(folder_path)
