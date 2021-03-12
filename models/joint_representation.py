import torch
import torch.nn as nn


class Joint_Representaion_Learner(nn.Module):
    def __init__(self, feats_size, opt):
        super(Joint_Representaion_Learner, self).__init__()
        self.fusion = opt.get('fusion', 'temporal_concat')

        if self.fusion not in ['temporal_concat', 'addition', 'none']:
            raise ValueError('We now only support the fusion type: temporal_concat | addition | none')

        self.norm_list = []
        self.is_bn = (opt.get('norm_type', 'bn').lower() == 'bn')

        if not opt['no_encoder_bn']:
            if self.fusion == 'addition':
                feats_size = [feats_size[0]]
            for i, item in enumerate(feats_size):
                tmp_module = nn.BatchNorm1d(item) if self.is_bn else nn.LayerNorm(item)
                self.norm_list.append(tmp_module)
                self.add_module("%s%d"%('bn' if self.is_bn else 'ln', i), tmp_module)

    def forward(self, encoder_outputs, encoder_hiddens):
        if not isinstance(encoder_hiddens, list):
            encoder_hiddens = [encoder_hiddens]
        encoder_hiddens = torch.stack(encoder_hiddens, dim=0).mean(0)

        if self.fusion == 'none':
            if isinstance(encoder_outputs, list):
                encoder_outputs = torch.cat(encoder_outputs, dim=1)
            return encoder_outputs, encoder_hiddens
        
        if not isinstance(encoder_outputs, list):
            encoder_outputs = [encoder_outputs]

        if self.fusion == 'addition':
            encoder_outputs = torch.stack(encoder_outputs, dim=0).mean(0)

        if len(self.norm_list):
            assert len(encoder_outputs) == len(self.norm_list)
            for i in range(len(encoder_outputs)):
                if self.is_bn:
                    batch_size, seq_len, _ = encoder_outputs[i].shape
                    encoder_outputs[i] = self.norm_list[i](encoder_outputs[i].contiguous().view(batch_size * seq_len, -1)).view(batch_size, seq_len, -1)
                else:
                    encoder_outputs[i] = self.norm_list[i](encoder_outputs[i])

        if self.fusion == 'temporal_concat':
            assert isinstance(encoder_outputs, list)
            encoder_outputs = torch.cat(encoder_outputs, dim=1)
        
        return encoder_outputs, encoder_hiddens
