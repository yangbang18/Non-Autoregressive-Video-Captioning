import torch
import argparse
from tqdm import tqdm
import os
from misc.run import get_loader, run_eval
from misc.logger import CsvLogger
from config import Constants
from misc.utils import load_model_and_opt, get_dict_mapping
from tensorboardX import SummaryWriter
import shutil
from misc.crit import get_criterion_during_evaluation


def prepare_collect_config(option, opt):
    if not os.path.exists(opt.collect_path):
        os.makedirs(opt.collect_path)

    names = [option['dataset'], option['method'], opt.evaluation_mode]
    if opt.not_only_best_candidate:
        names.insert(0, 'nobc')

    if option['decoding_type'] == 'ARFormer':
        parameter = 'bs%d_topk%d.pkl' % (option['beam_size'], option['topk'])
    else:
        names.append(('%s' % ('CT' if option['use_ct'] else '')) + option['paradigm'])
        if option['paradigm'] == 'mp':
            parameter = 'i%db%da%03d.pkl' % (
                option['iterations'], 
                option['length_beam_size'], 
                int(100*option['beam_alpha'])
            ) 
        else:
            parameter = 'q%dqi%db%da%03d.pkl' % (
                option['q'], 
                option['q_iterations'], 
                option['length_beam_size'], 
                int(100*option['beam_alpha'])
            )
    
    filename = '_'.join(names + [parameter])
    opt.collect_path = os.path.join(opt.collect_path, filename)


def main():
    '''Main Function'''
    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-df', '--default', default=False, action='store_true')
    parser.add_argument('-method', '--method', default='ARB', type=str)
    parser.add_argument('-dataset', '--dataset', default='MSRVTT', type=str)
    parser.add_argument('--default_model_name', default='best.pth.tar', type=str)
    parser.add_argument('-scope', '--scope', default='', type=str)
    parser.add_argument('-record', '--record', default=False, action='store_true')
    parser.add_argument('-field', '--field', nargs='+', type=str, default=['seed'])
    parser.add_argument('-val_and_test', '--val_and_test', default=False, action='store_true')

    parser.add_argument('-model_path', '--model_path', type=str)
    parser.add_argument('-teacher_path', '--teacher_path', type=str)

    parser.add_argument('-bs', '--beam_size', type=int, default=5, help='Beam size')
    parser.add_argument('-ba', '--beam_alpha', type=float, default=1.0)
    parser.add_argument('-topk', '--topk', type=int, default=1)

    # NA decoding
    parser.add_argument('-i', '--iterations', type=int, default=5)
    parser.add_argument('-lbs', '--length_beam_size', type=int, default=6)
    parser.add_argument('-q', '--q', type=int, default=1)
    parser.add_argument('-qi', '--q_iterations', type=int, default=1)
    parser.add_argument('-paradigm', '--paradigm', type=str, default='mp')
    parser.add_argument('-use_ct', '--use_ct', default=False, action='store_true')
    parser.add_argument('-md', '--masking_decision', default=False, action='store_true')
    parser.add_argument('-ncd', '--no_candidate_decision', default=False, action='store_true')
    parser.add_argument('--algorithm_print_sent', default=False, action='store_true')

    parser.add_argument('-batch_size', '--batch_size', type=int, default=128)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-em', '--evaluation_mode', type=str, default='test')
    parser.add_argument('-print_sent', action='store_true')
    parser.add_argument('-json_path', type=str, default='')
    parser.add_argument('-json_name', type=str, default='')
    parser.add_argument('-ns', '--no_score', default=False, action='store_true')
    parser.add_argument('-analyze', default=False, action='store_true')

    parser.add_argument('-latency', default=False, action='store_true')
    
    parser.add_argument('-specific', default=-1, type=int)
    parser.add_argument('-collect_path', type=str, default='./collected_captions')
    parser.add_argument('-collect', default=False, action='store_true')
    parser.add_argument('-nobc', '--not_only_best_candidate', default=False, action='store_true')

    opt = parser.parse_args()

    device = torch.device('cuda' if not opt.no_cuda else 'cpu')
    teacher_model = None
    dict_mapping = {}

    if opt.default:
        if opt.dataset.lower() == 'msvd':
            opt.dataset = 'Youtube2Text'
        opt.model_path = os.path.join(
            Constants.base_checkpoint_path,
            opt.dataset,
            opt.method,
            opt.scope,
            opt.default_model_name
        )
        if opt.method in ['NAB', 'NACF']:
            opt.teacher_path = os.path.join(
                Constants.base_checkpoint_path,
                opt.dataset,
                'ARB',
                opt.scope,
                opt.default_model_name
            )
            assert os.path.exists(opt.teacher_path)
    else:
        assert opt.model_path and os.path.exists(opt.model_path)

    model, option, other_info = load_model_and_opt(opt.model_path, device, return_other_info=True)
    if getattr(opt, 'teacher_path', None) is not None:
        print('Loading teacher model from %s' % opt.teacher_path)
        teacher_model, teacher_option = load_model_and_opt(opt.teacher_path, device)
        dict_mapping = get_dict_mapping(option, teacher_option)

    option['reference'] = option['reference'].replace('msvd_refs.pkl', 'refs.pkl')
    option['info_corpus'] = option['info_corpus'].replace('info_corpus_0.pkl', 'info_corpus.pkl')

    if not opt.default:
        _ = option['dataset']
        option.update(vars(opt))
        option['dataset'] = _
    else:
        if option['decoding_type'] != 'NARFormer':
            option['topk'] = opt.topk
            option['beam_size'] = 5
            option['beam_alpha'] = 1.0
        else:
            option['algorithm_print_sent'] = opt.algorithm_print_sent
            option['paradigm'] = opt.paradigm
            option['iterations'] = 5
            option['length_beam_size'] = 6
            option['beam_alpha'] = 1.35 if opt.dataset == 'MSRVTT' else 1.0
            option['q'] = 1
            option['q_iterations'] = 1 if opt.use_ct else 0
        option['use_ct'] = opt.use_ct
    
    if opt.collect:
        prepare_collect_config(option, opt)

    if opt.latency:
        opt.batch_size = 1
        option['batch_size'] = 1

    if opt.val_and_test:
        modes = ['validate', 'test']
        csv_filenames = ['validation_record.csv', 'testing_record.csv']
    else:
        modes = [opt.evaluation_mode]
        csv_filenames = ['validation_record.csv' if opt.evaluation_mode == 'validate' else 'testing_record.csv']
    
    crit = get_criterion_during_evaluation(option)

    for mode, csv_filename in zip(modes, csv_filenames):
        loader = get_loader(option, mode=mode, print_info=True, specific=opt.specific, batch_size=opt.batch_size)
        vocab = loader.dataset.get_vocab()

        if opt.record:
            summarywriter = SummaryWriter(os.path.join(option['checkpoint_path'], mode))
        else:
            summarywriter = None

        metric = run_eval(option, model, crit, loader, vocab, device, 
            teacher_model=teacher_model,
            dict_mapping=dict_mapping,
            json_path=opt.json_path, 
            json_name=opt.json_name, 
            print_sent=opt.print_sent, 
            no_score=opt.no_score,  
            analyze=True if opt.record else opt.analyze, 
            collect_best_candidate_iterative_results=True if opt.collect else False,
            collect_path=opt.collect_path,
            summarywriter=summarywriter,
            global_step=option['seed']
        )
        
        print(metric)
        if opt.record:
            fieldsnames = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4',
                            'METEOR', 'ROUGE_L', 'CIDEr', 'Sum', 
                            'ave_length', 'novel', 'unique', 'usage']
            if crit is not None:
                fieldsnames += crit.get_fieldsnames()
            logger = CsvLogger(filepath=option['checkpoint_path'], filename=csv_filename,
                                    fieldsnames=fieldsnames + opt.field)
            if 'loss' in metric:
                metric.pop('loss')

            for key in opt.field:
                metric[key] = option[key]
            logger.write(metric)
                

if __name__ == "__main__":
    main()

'''
python translate.py -df -method ARB -analyze
python translate.py -df -method ARB2 -analyze
python translate.py -df -method NAB -analyze
python translate.py -df -method NACF -analyze -use_ct
'''
