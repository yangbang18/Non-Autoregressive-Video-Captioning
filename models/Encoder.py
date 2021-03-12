import torch
import torch.nn as nn
from models.Decoder import dict2obj

__all__ = (
    'Encoder_HighWay',
)

class HighWay(nn.Module):
    def __init__(self, hidden_size, with_gate=True):
        super(HighWay, self).__init__()
        self.with_gate = with_gate
        self.w1 = nn.Linear(hidden_size, hidden_size)
        if self.with_gate:
            self.w2 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        #self._init_weights()

    def forward(self, x):
        y = self.tanh(self.w1(x))
        if self.with_gate:
            gate = torch.sigmoid(self.w2(x))
            return gate * x + (1 - gate) * y
        else:
            return x + y

class MultipleStreams(nn.Module):
    def __init__(self, opt, module_func, is_rnn=False):
        super(MultipleStreams, self).__init__()
        self.encoders = []

        modality = opt['modality'].lower()
        for char in modality:
            input_dim = opt.get('dim_' + char, None)
            output_dim = opt.get('dim_hidden', 512)
            dropout = opt.get('encoder_dropout', 0.5)
            assert input_dim is not None, \
                'The modality is {}, but dim_{} can not be found in opt'.format(modality, char)
            
            module = module_func(input_dim, output_dim, dropout)
            self.add_module("Encoder_%s" % char.upper(), module)
            self.encoders.append(module)
 
        self.num_feats = len(modality)
        self.is_rnn = is_rnn

    def forward(self, input_feats):
        assert self.num_feats == len(input_feats)
        if not self.is_rnn:
            encoder_ouputs = [encocder(feats) for encocder, feats in zip(self.encoders, input_feats)]
            encoder_hiddens = [item.mean(1) for item in encoder_ouputs]
        else:
            pass
            # TODO
        
        if getattr(self, 'subsequent_processing', None) is not None:
            return self.subsequent_processing(encoder_ouputs, encoder_hiddens)

        return encoder_ouputs, encoder_hiddens


class Encoder_HighWay(MultipleStreams):
    def __init__(self, opt):
        with_gate = opt.get('gate', True)
        module_func = lambda x,y,z: nn.Sequential(nn.Linear(x, y), HighWay(y, with_gate), nn.Dropout(z))
        super(Encoder_HighWay, self).__init__(opt, module_func)
