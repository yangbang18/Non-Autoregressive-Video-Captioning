import torch
import torch.nn as nn
import math
from config import Constants

__all__ = (
    'Predictor_length', 
    'Auxiliary_Task_Predictor'
)


class Predictor_length(nn.Module):
    def __init__(self, opt, key_name):
        super(Predictor_length, self).__init__()
        self.net = nn.Sequential(
                    nn.Linear(opt['dim_hidden'], opt['dim_hidden']),
                    nn.ReLU(),
                    nn.Dropout(opt['hidden_dropout_prob']),
                    nn.Linear(opt['dim_hidden'], opt['max_len']),
                )
        self.key_name = key_name

    def forward(self, enc_output, **kwargs):
        if isinstance(enc_output, list):
            assert len(enc_output) == 1
            enc_output = enc_output[0]
        assert len(enc_output.shape) == 3

        out = self.net(enc_output.mean(1))
        return {self.key_name: torch.log_softmax(out, dim=-1)}


class Auxiliary_Task_Predictor(nn.Module):
    """docstring for auxiliary_task_predictor"""
    def __init__(self, layers):
        super(Auxiliary_Task_Predictor, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, enc_output, **kwargs):
        results = {}
        for layer in self.layers:
            results.update(layer(enc_output=enc_output, **kwargs))
        return results

        
