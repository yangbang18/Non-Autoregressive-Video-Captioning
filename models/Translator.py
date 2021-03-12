''' This module will handle the text generation with beam search. '''
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Beam import Beam
import os, json
from config import Constants
from misc.utils import auto_enlarge


class Translator(object):
    ''' Load with trained model and handle the beam search '''

    def __init__(self, model, opt, device=torch.device('cuda'), teacher_model=None, dict_mapping={}):
        self.model = model
        self.model.eval()
        self.opt = opt
        self.device = device
        self.teacher_model = teacher_model
        self.dict_mapping = dict_mapping
        self.length_bias = opt.get('length_bias', 0)

    def get_inst_idx_to_tensor_position_map(self, inst_idx_list):
        ''' Indicate the position of an instance in a tensor. '''
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

    def collect_active_part(self, beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
        ''' Collect tensor parts associated to active instances. '''
        _, *d_hs = beamed_tensor.size()
        n_curr_active_inst = len(curr_active_inst_idx)
        new_shape = (n_curr_active_inst * n_bm, *d_hs)

        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(*new_shape)

        return beamed_tensor

    def auto_collect_active_part(self, beamed_tensor, *args):
        ''' Collect tensor parts associated to active instances. '''
        if isinstance(beamed_tensor, list):
            if isinstance(beamed_tensor[0], tuple):
                # enc_hidden for LSTM based decoder
                return [
                    tuple([self.collect_active_part(_, *args) for _ in item])
                    for item in beamed_tensor
                ]

            return [self.collect_active_part(item, *args) for item in beamed_tensor]
        else:
            if isinstance(beamed_tensor, tuple):
                # enc_hidden for LSTM based decoder
                return tuple([self.collect_active_part(item, *args) for item in beamed_tensor])
            return self.collect_active_part(beamed_tensor, *args)


    def collate_active_info(self, inputs_for_decoder, inst_idx_to_position_map, active_inst_idx_list, n_bm):
        # Sentences which are still active are collected,
        # so the decoder will not run on completed sentences.
        n_prev_active_inst = len(inst_idx_to_position_map)
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)
        
        args = (active_inst_idx, n_prev_active_inst, n_bm)

        for key in inputs_for_decoder.keys():
            inputs_for_decoder[key] = self.auto_collect_active_part(inputs_for_decoder[key], *args)

        active_inst_idx_to_position_map = self.get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        return inputs_for_decoder, active_inst_idx_to_position_map

    def collect_active_inst_idx_list(self, inst_beams, word_prob, inst_idx_to_position_map):
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])

            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]
        return active_inst_idx_list

    def collect_hypothesis_and_scores(self, inst_dec_beams, n_best):
        all_hyp, all_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):
            scores, tk = inst_dec_beams[inst_idx].sort_finished(self.opt.get('beam_alpha', 1.0))
            n_best = min(n_best, len(scores))
            all_scores += [scores[:n_best]]
            hyps = [inst_dec_beams[inst_idx].get_hypothesis_from_tk(t, k) for t, k in tk[:n_best]]
            all_hyp += [hyps]
        return all_hyp, all_scores


    def translate_batch_ARFormer(self, encoder_outputs, category):
        ''' Translation work in one batch '''

        def beam_decode_step(
                inst_dec_beams, len_dec_seq, inputs_for_decoder, inst_idx_to_position_map, n_bm):
            ''' Decode and update beam status, and then return active beam idx '''

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def predict_word(dec_seq, inputs_for_decoder, n_active_inst, n_bm):
                dec_output, *_ = self.model.decoder(dec_seq, **inputs_for_decoder)
                if isinstance(dec_output, list):
                    dec_output = dec_output[-1]
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h

                word_prob = self.model.tgt_word_prj(dec_output)
                word_prob = F.log_softmax(word_prob, dim=1)
                #print(word_prob[0, :10])
                word_prob = word_prob.view(n_active_inst, n_bm, -1)

                return word_prob

            n_active_inst = len(inst_idx_to_position_map)

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            word_prob = predict_word(dec_seq, inputs_for_decoder, n_active_inst, n_bm)

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = self.collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        with torch.no_grad():
            inputs_for_decoder = self.model.prepare_inputs_for_decoder(encoder_outputs, category)
            #-- Repeat data for beam search
            n_bm = self.opt["beam_size"]
            n_inst = inputs_for_decoder['enc_output'].size(0)

            for key in inputs_for_decoder:
                inputs_for_decoder[key] = auto_enlarge(inputs_for_decoder[key], n_bm)

            #-- Prepare beams
            inst_dec_beams = [Beam(n_bm, self.opt["max_len"], device=self.device, specific_nums_of_sents=self.opt.get('topk', 1)) for _ in range(n_inst)]

            #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = self.get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            #-- Decode
            for len_dec_seq in range(1, self.opt["max_len"]):

                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams, len_dec_seq, inputs_for_decoder, inst_idx_to_position_map, n_bm)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                inputs_for_decoder, inst_idx_to_position_map = self.collate_active_info(
                    inputs_for_decoder, inst_idx_to_position_map, active_inst_idx_list, n_bm)

        batch_hyp, batch_scores = self.collect_hypothesis_and_scores(inst_dec_beams, self.opt.get("topk", 1))

        return batch_hyp, batch_scores

    def translate_batch_NARFormer(self, encoder_outputs, teacher_encoder_outputs, category, tgt_tokens, tgt_vocab, **kwargs):
        from decoding import generate
        with torch.no_grad():
            return generate(
                        opt=self.opt,
                        model=self.model,
                        teacher_model=self.teacher_model,
                        encoder_outputs=encoder_outputs, 
                        teacher_encoder_outputs=teacher_encoder_outputs,
                        category=category, 
                        tgt_tokens=tgt_tokens, 
                        tgt_vocab=tgt_vocab, 
                        dict_mapping=self.dict_mapping,
                        length_bias=self.length_bias,
                        **kwargs
                    )

    def translate_batch(self, encoder_outputs, category, tgt_tokens, tgt_vocab, teacher_encoder_outputs=None, **kwargs):
        if self.opt['decoding_type'] == 'NARFormer':
            return self.translate_batch_NARFormer(encoder_outputs, teacher_encoder_outputs, category, tgt_tokens, tgt_vocab, **kwargs)

        func_name = 'translate_batch_%s' % self.opt['decoding_type']
        return getattr(self, func_name, None)(encoder_outputs, category)

