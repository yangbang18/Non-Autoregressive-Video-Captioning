import argparse
from config import Constants
import os

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='MSRVTT', help='MSRVTT | Youtube2Text')
    parser.add_argument('-m', '--modality', type=str, default='mi')
    parser.add_argument('-df', '--default', default=False, action='store_true')
    parser.add_argument('--scope', type=str, default='')
    parser.add_argument('-field', '--field', nargs='+', type=str, default=['seed'])
    parser.add_argument('--no_cuda', default=False, action='store_true')
    parser.add_argument('--method', type=str, default='', help='ARB: autoregressive baseline \n'
                                                                'ARB2: ARB + visual word generation')

    parser.add_argument('--encoder', type=str, default='Encoder_HighWay', help='specify the encoder if we want')
    parser.add_argument('--decoder', type=str, default='BertDecoder', help='specify the decoder if we want')
    parser.add_argument('--decoding_type', type=str, default='ARFormer', help='ARFormer | NARFormer')
    parser.add_argument('--fusion', type=str, default='temporal_concat', help='temporal_concat | addition')

    model = parser.add_argument_group(title='Model Parameters')
    # Transformer Configurations
    model.add_argument('--dim_hidden', type=int, default=512, help='size of the rnn hidden layer')
    model.add_argument('--num_hidden_layers_decoder', type=int, default=1)
    model.add_argument('--num_attention_heads', type=int, default=8)
    model.add_argument('--intermediate_size', type=int, default=2048)
    model.add_argument('--hidden_act', type=str, default='gelu_new')
    model.add_argument('--hidden_dropout_prob', type=float, default=0.5)
    model.add_argument('--attention_probs_dropout_prob', type=float, default=0.0)
    model.add_argument("--max_len", type=int, default=30, help='max length of captions')
    model.add_argument('--layer_norm_eps', type=float, default=0.00001)
    model.add_argument('--watch', type=int, default=0)
    model.add_argument('--pos_attention', default=False, action='store_true')
    model.add_argument('--enhance_input', type=int, default=2, 
                        help='for NA decoding, 0: without R | 1: re-sampling(R)) | 2: meanpooling(R)')
    model.add_argument('--with_layernorm', default=False, action='store_true')
    
    model.add_argument('-wc', '--with_category', default=False, action='store_true',
                       help='specified for the MSRVTT dataset, use category tags or not')
    model.add_argument('--num_category', type=int, default=20)

    model.add_argument('--encoder_dropout', type=float, default=0.5, 
                        help='strength of dropout in the encoder')
    model.add_argument('--no_encoder_bn', default=False, action='store_true', 
                        help='by default, a BN layer is placed after the encoder outputs of a modality')
    model.add_argument('--norm_type', type=str, default='bn')
    model.add_argument('--dim_word', type=int, default=512, 
                        help='the embedding size of each token in the vocabulary')
    model.add_argument('-tie', '--tie_weights', default=False, action='store_true', 
                        help='share the weights between word embeddings and the projection layer')

    training = parser.add_argument_group(title='Training Parameters')
    training.add_argument('--seed', default=0, type=int, help='for reproducibility')
    training.add_argument('--learning_rate', default=5e-4, type=float, help='the initial larning rate')
    training.add_argument('--decay', default=0.9, type=float, help='the decay rate of learning rate per epoch')
    training.add_argument('--minimum_learning_rate', default=5e-5, type=float, help='the minimum learning rate')
    training.add_argument('--n_warmup_steps', type=int, default=0, help='the number of warmup steps towards the initial lr')
    training.add_argument('--optim', type=str, default='adam', help='adam | rmsprop')
    training.add_argument('--grad_clip', type=float, default=5, help='clip gradients at this value')
    training.add_argument('--weight_decay', type=float, default=5e-4, help='Strength of weight regularization')
    training.add_argument('-e', '--epochs', type=int, default=50, help='number of epochs')
    training.add_argument('-b', '--batch_size', type=int, default=64, help='minibatch size')

    training.add_argument('--pretrained_path', default='', type=str, help='path for the pretrained model')
    # NA decoding
    training.add_argument('--teacher_path', type=str, default='', help='path for the AR-B model')
    training.add_argument('--beta', nargs='+', type=float, default=[0, 1],
                          help='len=2, [lowest masking ratio, highest masking ratio]')
    training.add_argument('--visual_word_generation', default=False, action='store_true')
    training.add_argument('--demand', nargs='+', type=str, default=['VERB', 'NOUN'], 
                          help='pos_tag we want to focus on when training with visual word generation')
    training.add_argument('-nvw', '--nv_weights', nargs='+', type=float, default=[0.8, 1.0],
                          help='len=2, weights of visual word generation and caption generation (or mlm)')
    training.add_argument('--load_teacher_weights', default=False, action='store_true',
                          help='specified for NA-based models, initialize randomly or inherit from the teacher (AR-B)')
    training.add_argument('--no_test', default=False, action='store_true')

    evaluation = parser.add_argument_group(title='Evaluation Parameters')
    evaluation.add_argument('-see', '--start_eval_epoch', type=int, default=0,
                            help='start evaluation after a specific epoch')
    evaluation.add_argument('--tolerence', type=int, default=1000, 
                            help='for early stop')
    evaluation.add_argument('--metric_sum', nargs='+', type=int, default=[1, 1, 1, 1],
                            help='meta sum of the metrics')
    evaluation.add_argument('--standard', nargs='+', type=str, default=['Bleu_4', 'METEOR', 'CIDEr'], #['Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr'],
                            help='the standard of performance to select the best model')
    evaluation.add_argument('-bs', '--beam_size', type=int, default=1,
                            help='specified for AR decoding, the number of candidates')
    evaluation.add_argument('-ba', '--beam_alpha', type=float, default=1.0,
                            help='the preference of sentence length, larger --> longer')
    # NA decoding
    evaluation.add_argument('--paradigm', type=str, default='mp', 
                            help='mp: MaskPredict | l2r: Left2Right | ef: EasyFirst')
    evaluation.add_argument('-lbs', '--length_beam_size', type=int, default=6,
                            help='specified for NA decoding, the number of length candidates')
    evaluation.add_argument('--iterations', type=int, default=5,
                            help='the number of iterations for the MP algorithm')
    evaluation.add_argument('--q', type=int, default=1,
                            help='the number of tokens to update for L2R & EF algorithms')
    evaluation.add_argument('--q_iterations', type=int, default=1,
                            help='the number of iterations for L2R & EF algorithms')
    evaluation.add_argument('--use_ct', default=False, action='store_true', 
                            help='use coarse-grained templates or not, only for methods with visual word generation')
    # checkpoint settings
    evaluation.add_argument('--k_best_model', type=int, default=1,
                            help='checkpoints with top-k performance will be saved')
    evaluation.add_argument('--save_checkpoint_every', type=int, default=1,
                            help='how often to save a model checkpoint (in epoch)?')

    multitask = parser.add_argument_group(title='Multi-Task Parameters')
    multitask.add_argument('--crit', nargs='+', type=str, default=['lang'], help='lang | length')
    multitask.add_argument('--crit_name', nargs='+', type=str, default=['Cap Loss'])
    multitask.add_argument('--crit_scale', nargs='+', type=float, default=[1.0])

    dataloader = parser.add_argument_group(title='Dataloader Parameters')
    dataloader.add_argument('--n_frames', type=int, default=8, help='the number of frames to represent a whole video')
    dataloader.add_argument('--n_caps_per_video', type=int, default=0, 
                            help='the number of captions per video to constitute the training set')
    dataloader.add_argument('--random_type', type=str, default='segment_random', 
                            help='sampling strategy during training: segment_random (default) | all_random | equally_sampling')
    dataloader.add_argument('--load_feats_type', type=int, default=1, 
                            help='load feats from the same frame_ids (0) '
                            'or different frame_ids (1), '
                            'or directly load all feats without sampling (2)')

    # modality information
    dataloader.add_argument('--dim_a', type=int, default=1, help='feature dimension of the audio modality')
    dataloader.add_argument('--dim_m', type=int, default=2048, help='feature dimension of the motion modality')
    dataloader.add_argument('--dim_i', type=int, default=2048, help='feature dimension of the image modality')
    dataloader.add_argument('--dim_o', type=int, default=1, help='feature dimension of the object modality')
    dataloader.add_argument('--dim_t', type=int, default=1)
    dataloader.add_argument('--feats_a_name', nargs='+', type=str, default=[])
    dataloader.add_argument('--feats_m_name', nargs='+', type=str, default=['motion_resnext101_kinetics_duration16_overlap8.hdf5'])
    dataloader.add_argument('--feats_i_name', nargs='+', type=str, default=['image_resnet101_imagenet_fps_max60.hdf5'])
    dataloader.add_argument('--feats_o_name', nargs='+', type=str, default=[])
    dataloader.add_argument('--feats_t_name', nargs='+', type=str, default=[])
    # corpus information
    dataloader.add_argument('--info_corpus_name', type=str, default='info_corpus.pkl')
    dataloader.add_argument('--reference_name', type=str, default='refs.pkl')

    args = parser.parse_args()
    check_dataset(args)
    check_method(args)
    check_valid(args)
    return args


def check_valid(args):
    assert args.load_feats_type in [0, 1, 2]
    if not args.default:
        assert args.scope, \
            "Please add the argument \'--scope $folder_name_to_save_models\'"

def check_dataset(args):
    if args.dataset.lower() == 'msvd':
        args.dataset = 'Youtube2Text'
    
    assert args.dataset in ['Youtube2Text', 'MSRVTT'], \
        "We now only support Youtube2Text (MSVD) and MSRVTT datasets."

    if args.default:
        if args.dataset == 'Youtube2Text':
            args.beta = [0, 1]
            args.max_len = 20
            args.with_category = False
        elif args.dataset == 'MSRVTT':
            args.beta = [0.35, 0.9]
            args.max_len = 30
            args.with_category = True
    
    if args.dataset == 'Youtube2Text':
        assert not args.with_category, \
            "Category information is not available in the Youtube2Text (MSVD) dataset"


def check_method(args):
    if args.method:
        import yaml
        methods = yaml.full_load(open('./config/methods.yaml'))
        assert args.method in methods.keys(), \
            "The method {} can not be found in ./config/methods.yaml".format(args.method)
        for k, v in methods[args.method].items():
            setattr(args, k, v)
    
    if args.decoding_type == 'NARFormer':
        args.crit = ['lang', 'length']
        args.crit_name = ['Cap Loss', 'Length Loss']
        args.crit_scale = [1.0, 1.0]
    args.crit_key = [Constants.mapping[item.lower()] for item in args.crit]

    if args.default:
        if args.decoding_type == 'NARFormer':
            if args.visual_word_generation:
                args.use_ct = True
                args.nv_weights = [0.8, 1.0]
            args.enhance_input = 2
            args.length_beam_size = int(6)
            args.iterations = int(5)
            args.beam_alpha = 1.35 if args.dataset == 'MSRVTT' else 1.0
            args.algorithm_print_sent = True
            args.teacher_path = os.path.join(
                Constants.base_checkpoint_path,
                args.dataset,
                'ARB',
                args.scope,
                'best.pth.tar'
            )
            assert os.path.exists(args.teacher_path)
            args.load_teacher_weights = True
            args.with_teacher = True
        else:
            args.beam_size = int(5.0)
            args.beam_alpha = 1.0
