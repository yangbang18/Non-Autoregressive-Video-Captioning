from config import Constants
import torch
import torch.nn.functional as F

from .algorithms import MaskPredict, Left2Right, EasyFirst
from misc.utils import auto_enlarge

algorithms_mapping = {
    'mp': MaskPredict,
    'l2r': Left2Right,
    'ef': EasyFirst
}

def generate(
            opt, model, teacher_model, 
            encoder_outputs, teacher_encoder_outputs, category, 
            tgt_tokens, tgt_vocab, dict_mapping, length_bias, 
            **kwargs
    ):
    paradigm = opt.get('paradigm', 'mp')
    assert paradigm in ['mp', 'l2r', 'ef']
    algorithm = algorithms_mapping[paradigm](opt, dict_mapping, tgt_vocab)

    length_beam_size = opt['length_beam_size']
    if opt.get('load_generated_captions', False):
        gold_target_len = tgt_tokens.ne(Constants.PAD).sum(-1)
    else:
        gold_target_len = None
    #gold_target_len = tgt_tokens.ne(Constants.PAD).sum(-1) if opt['use_gold_target_len'] else None
    beam_alpha = opt.get('beam_alpha', 1.0)
    #print(beam_alpha)

    pred_length = encoder_outputs['pred_length']
    bsz = pred_length.size(0)
    beam = predict_length_beam(gold_target_len, pred_length, length_beam_size, length_bias, opt)   
    length_beam_size = beam.shape[-1] 
    max_len = beam.max().item()

    length_mask = torch.triu(pred_length.new(max_len, max_len).fill_(1).long(), 1)
    length_mask = torch.stack([length_mask[beam[batch] - 1] for batch in range(bsz)], dim=0)

    if gold_target_len is not None:
        tgt_tokens = tgt_tokens[:, :max_len]
        tgt_tokens[tgt_tokens==Constants.PAD] = Constants.MASK
        tgt_tokens = tgt_tokens.unsqueeze(1).repeat(1, length_beam_size, 1)
    else:
        tgt_tokens = pred_length.new(bsz, length_beam_size, max_len).fill_(Constants.MASK).long()

    tgt_tokens = (1 - length_mask) * tgt_tokens + length_mask * Constants.PAD
    tgt_tokens = tgt_tokens.view(bsz * length_beam_size, max_len)


    inputs_for_decoder = model.prepare_inputs_for_decoder(encoder_outputs, category)
    for key in inputs_for_decoder.keys():
        inputs_for_decoder[key] = auto_enlarge(inputs_for_decoder[key], length_beam_size)

    if teacher_encoder_outputs is not None:
        teacher_inputs_for_decoder = teacher_model.prepare_inputs_for_decoder(teacher_encoder_outputs, category)
        for key in teacher_inputs_for_decoder.keys():
            teacher_inputs_for_decoder[key] = auto_enlarge(teacher_inputs_for_decoder[key], length_beam_size)
    else:
        teacher_inputs_for_decoder = None
    
    hypotheses, lprobs, collect_results = algorithm.generate(model, teacher_model, inputs_for_decoder, teacher_inputs_for_decoder, tgt_tokens)
    
    tgt_lengths = (1 - length_mask).sum(-1)
    hypotheses = hypotheses.view(bsz, length_beam_size, max_len)
    lprobs = lprobs.view(bsz, length_beam_size, max_len)
    tgt_lengths = tgt_lengths.view(bsz, length_beam_size)
    #tgt_lengths = (1 - length_mask).sum(-1)-1

    avg_log_prob = lprobs.sum(-1) / (tgt_lengths.float() ** beam_alpha)
    best_lengths = avg_log_prob.max(-1)[1]                                          # [batch_size]

    best_lengths = best_lengths.unsqueeze(1).unsqueeze(2).repeat(1, 1, max_len)     # [batch_size, 1, max_len]
    
    hypotheses = hypotheses.gather(1, best_lengths).squeeze(1)                      # [batch_size, max_len]
    #lprobs = lprobs.gather(1, best_lengths).squeeze(1)                             = [batch_size, max_len]
    lprobs = None # For speedup
    assert isinstance(collect_results, tuple)
    if collect_results[0]:
        sents, scores, _ = collect_results
        if not opt.get('not_only_best_candidate', False) and not opt.get('collect_last', False):
            sents = [item.view(bsz, length_beam_size, max_len) for item in sents]
            sents = [item.gather(1, best_lengths).squeeze(1) for item in sents]

            scores = [item.view(bsz, length_beam_size, max_len) for item in scores]
            scores = [item.gather(1, best_lengths).squeeze(1) for item in scores]

        lprobs = (torch.stack(sents, dim=1), torch.stack(scores, dim=1))
    
    if kwargs.get('output_attentions', False):
        assert len(collect_results) == 3
        attentions = collect_results[-1]
        assert isinstance(attentions, list) and len(attentions) == 2

        new_attentions = []
        for item in attentions:
            _shape = item.shape # [bsz * length_beam, 1(0) + T, n_head, len_q, len_k]
            assert len(_shape) == 5 and _shape[3] == max_len
            _, num_iterations, num_heads, _, len_k = _shape
            item = item.view(bsz, length_beam_size, *_shape[1:]) # [bsz, length_beam, 1(0) + T, n_head, len_q, len_k]
            best_idx = best_lengths.unsqueeze(2).unsqueeze(2).unsqueeze(-1).repeat(1, 1, num_iterations, num_heads, 1, len_k)
            new_attentions.append(item.gather(1, best_idx).squeeze(1))

        return hypotheses, lprobs, new_attentions
    else:
        return hypotheses, lprobs

    hypotheses = torch.stack([hypotheses[b, l, :] for b, l in enumerate(best_lengths)], dim=0)
    lprobs = torch.stack([lprobs[b, l, :] for b, l in enumerate(best_lengths)], dim=0)

    return hypotheses, lprobs


def predict_length_beam(gold_target_len, predicted_lengths, length_beam_size, length_bias, opt):
    if gold_target_len is not None:
        beam_starts = gold_target_len - (length_beam_size - 1) // 2
        beam_ends = gold_target_len + length_beam_size // 2 + 1
        #beam = torch.stack([torch.arange(7, 12, device=beam_starts.device) for batch in range(gold_target_len.size(0))], dim=0)
        beam = torch.stack([torch.arange(beam_starts[batch], beam_ends[batch], device=beam_starts.device) for batch in range(gold_target_len.size(0))], dim=0)
    else:
        beam = predicted_lengths.topk(length_beam_size, dim=1)[1] + length_bias

    if opt.get('example', ''):
        print(beam)
        if len(opt.get('example_len', [])):
            beam = torch.LongTensor([opt['example_len']]).to(predicted_lengths.device)
    else:
        max_len = opt['max_len'] - 1
        beam[beam < 4] = 4
        beam[beam > max_len] = max_len
    
    # print(beam)
    return beam
