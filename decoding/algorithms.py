from config import Constants
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

def generate_step_with_prob(out, zeros=[]):
    probs = F.softmax(out, dim=-1)
    for item in zeros:
        if len(probs.shape) == 3:
            probs[:, :, item] = 0
        else:
            probs[:, item] = 0
    max_probs, idx = probs.max(dim=-1)
    return idx, max_probs, probs

def to_sentence_with_prob(hyp, prob, vocab, break_words=[Constants.PAD], skip_words=[]):
    tokens = []
    for word_id, p in zip(hyp, prob):
        if word_id in skip_words:
            continue
        if word_id in break_words:
            break
        tokens.append('%12s(%.2f)'%(vocab[word_id], p))
    return ' '.join(tokens)

class Algorithm_Base(object):
    """docstring for Algorithm_Base"""
    def __init__(self, opt, dict_mapping, tgt_vocab):
        super(Algorithm_Base, self).__init__()
        # collect results for analyses
        self.collect_best_candidate_iterative_results = opt.get('collect_best_candidate_iterative_results', False)
        self.collect_last = opt.get('collect_last', False)
        self.collect_results = []
        self.collect_scores = []
        self.collect_attentions = [[], []]

        # knowledge distillation
        self.dict_mapping = dict_mapping

        # if masking_decision = True, teacher will be used to rescore the intermediate sequences
        # if no_candidate_decision = False, teacher will be used to rescore the final sequences
        self.masking_decision = opt.get('masking_decision', False)
        self.no_candidate_decision = opt.get('no_candidate_decision', False)

        self.visual_tag = Constants.VIS

        # use to print sentences if we want
        self.vocab = tgt_vocab # itow
        self.wtoi = {w: i for i, w in self.vocab.items()}
        self.algorithm_print_sent = opt.get('algorithm_print_sent', False)

        self.opt = opt

    def collect_data(self, tgt_tokens, token_probs, attentions, is_last=False):
        # collect results for analyses
        if self.collect_best_candidate_iterative_results and not self.collect_last:
            self.collect_results.append(tgt_tokens.clone())
            self.collect_scores.append(token_probs.clone())
            if self.opt.get('example', ''):
                if self.opt['method'] == 'NACF':
                    '''
                        attentions: [
                            ((self attention), (cross attention))
                        ]
                        n_head, bsz * length_beam, len_q, len_k
                    '''
                    self.collect_attentions[0].append(attentions[0][0][0].permute(1, 0, 2, 3))
                    self.collect_attentions[1].append(attentions[0][0][1].permute(1, 0, 2, 3))
                else:
                    self.collect_attentions[0].append(attentions[0][0].permute(1, 0, 2, 3))
                    self.collect_attentions[1].append(attentions[0][1].permute(1, 0, 2, 3))
        elif self.collect_last and is_last:
            self.collect_results.append(tgt_tokens.clone())
            self.collect_scores.append(token_probs.clone())

    def get_collected_data(self):
        if self.collect_attentions[0]:
            collected_attentions = [torch.stack(item, dim=1) for item in self.collect_attentions] #[bsz * length_beam, 1(0) + T, n_head, len_q, len_k]
        else:
            collected_attentions = [[], []]
            
        return (
            self.collect_results, 
            self.collect_scores, 
            collected_attentions, 
        ) 

    def manual_adjustment(self, tgt_tokens, token_probs, model, decoder_out, num_visual_words_show=5, num_visual_words_keep=2):
        assert num_visual_words_show > num_visual_words_keep

        # manually adjust the words of specified positions
        for i in range(tgt_tokens.size(0)):
            if self.opt.get('manual_words', []):
                assert len(self.opt.get('example_len', [])) == 1
                assert len(self.opt.get('manual_positions', [])) == len(self.opt['manual_words'])

                tgt_tokens[i], token_probs[i], all_probs = generate_step_with_prob(model.tgt_word_prj(decoder_out[i]))
                print(tgt_tokens)
                print(token_probs)

                for word, position in zip(self.opt['manual_words'], self.opt['manual_positions']):
                    assert position < tgt_tokens.size(1)
                    word_id = self.wtoi.get(word, Constants.UNK)
                    assert word_id != Constants.UNK

                    tgt_tokens[i, position] = word_id
                    #token_probs[i, position] = all_probs[position, word_id]
                    token_probs[i, position] = 0.3

                print(tgt_tokens)
                print(token_probs)
            else:
                unknown_num = (tgt_tokens[i].eq(Constants.MASK) | tgt_tokens[i].eq(Constants.PAD)).sum()
                if unknown_num == tgt_tokens.size(1):
                    all_probs = F.softmax(model.tgt_word_prj(decoder_out[i]), dim=-1)
                    modified_probs = all_probs.clone()
                    modified_probs[:, Constants.MASK] = 0
                    token_probs[i], tgt_tokens[i] = modified_probs.max(dim=-1)

                    probs, indices= all_probs.topk(num_visual_words_show, dim=-1, largest=True, sorted=True) # [seq_len, num_visual_words_show]

                    # show top-k predictions of visual word generation
                    for j in range(probs.shape[0]):
                        strs = ['%15s(%.2f)' % (self.vocab[wid.item()], p.item()) for wid, p in zip(indices[j], probs[j])]
                        print('position %02d: %s' % (j, '\t'.join(strs)))

                    # keep specified number of viusal words
                    lower_bound = probs[:, 1].topk(num_visual_words_keep, dim=-1, largest=True, sorted=True)[0][-1]
                    filter_indices = token_probs[i] < lower_bound
                    token_probs[i][filter_indices] = 0
                    tgt_tokens[i][filter_indices] = Constants.MASK

        return tgt_tokens, token_probs

    def get_coarse_grained_templates(self, model, inputs_for_decoder, tgt_tokens, pad_mask):
        mask_ind = tgt_tokens.eq(Constants.MASK)
        tgt_tokens[mask_ind] = self.visual_tag
        tgt_tokens, token_probs, attentions = self.generate_non_autoregressive(model, inputs_for_decoder, tgt_tokens, pad_mask, is_ct=True)
        token_probs[tgt_tokens.eq(Constants.MASK)] = 0.0
        return tgt_tokens, token_probs, attentions

    def generate_non_autoregressive(self, model, inputs_for_decoder, tgt_tokens, pad_mask, zeros=[], tag_replace=None, return_all_probs=False, is_ct=False, debug=False):
        decoder_out, _, attentions = model.decoder(tgt_tokens, **inputs_for_decoder, output_attentions=True)
        if isinstance(decoder_out, list):
            assert len(decoder_out) == 1
            decoder_out = decoder_out[0]

        tgt_tokens, token_probs, all_probs = generate_step_with_prob(model.tgt_word_prj(decoder_out), zeros=zeros)

        if is_ct and self.opt.get('manual', False):
            tgt_tokens, token_probs = self.manual_adjustment(tgt_tokens, token_probs, model, decoder_out)

        tgt_tokens[pad_mask] = Constants.PAD
        token_probs[pad_mask] = 1.0

        if return_all_probs:
            return tgt_tokens, token_probs, all_probs

        if tag_replace is not None:
            source, target = tag_replace
            ind = tgt_tokens.eq(source)
            tgt_tokens[ind] = target
            copy_ = token_probs.clone()
            token_probs[ind] = 0.0
            return tgt_tokens, token_probs, copy_
        return tgt_tokens, token_probs, attentions

    def mapping(self, tgt_tokens):
        tokens = tgt_tokens.clone().flatten()
        for i, token in enumerate(tokens):
            tokens[i] = self.dict_mapping[token.item()]
        return tokens.view(*tgt_tokens.shape)

    def scoring_by_teacher(self, teacher_model, teacher_inputs_for_decoder, tgt_tokens, pad_mask, is_last=False):
        all_ones = tgt_tokens.new(*tgt_tokens.shape).fill_(1).float()
        if teacher_model is None:
            return all_ones

        if is_last:
            if self.no_candidate_decision:
                return all_ones
        else:
            if not self.masking_decision:
                return all_ones

        # if we use knowledge distillation, we should map the tokens
        tokens = self.mapping(tgt_tokens) if self.dict_mapping != {} else tgt_tokens

        # add the <bos> token to the start of the sequences
        tgt_tokens_with_bos = torch.cat([tokens.new(tokens.size(0), 1).fill_(Constants.BOS), tokens], dim=1)

        # forward
        decoder_out, *_ = teacher_model.decoder(tgt_tokens_with_bos[:, :-1], **teacher_inputs_for_decoder)
        if isinstance(decoder_out, list):
            decoder_out = decoder_out[-1]
        probs = F.softmax(teacher_model.tgt_word_prj(decoder_out), dim=-1)
        
        # get the possibility of p(y_t | y_<t, R)
        probs = probs.gather(2, tokens.unsqueeze(2)).squeeze(2)

        # mask sure the possibility of <pad> tokens is 1.0
        probs[pad_mask] = 1.0
        return probs

    def select_worst(self, token_probs, num_mask):
        """
            for each example i
            select num_mask[i] tokens that the model is least confident about to mask out
        """
        masks = torch.zeros(*token_probs.shape, device=token_probs.device)
        for i in range(masks.size(0)):
            ind = token_probs[i, :].topk(max(1, num_mask[i]), largest=False, sorted=False)[1]
            masks[i, ind] = 1
        return masks.bool()

    def print_sent(self, tgt_tokens, token_probs, counter, debug=False):
        if self.algorithm_print_sent or debug:
            sample_ind = 0
            tqdm.write("Iteration %2d: "%counter + \
                to_sentence_with_prob(tgt_tokens[sample_ind].tolist(), token_probs[sample_ind].tolist(), self.vocab)) 


class MaskPredict(Algorithm_Base):
    def __init__(self, opt, dict_mapping, tgt_vocab):
        super().__init__(opt, dict_mapping, tgt_vocab)
        self.use_ct = opt.get('use_ct', False)
        self.T = opt.get('iterations', 5)


    def generate(self, model, teacher_model, inputs_for_decoder, teacher_inputs_for_decoder, tgt_tokens):
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        
        if self.use_ct:
            tgt_tokens, token_probs, attentions = self.get_coarse_grained_templates(model, inputs_for_decoder, tgt_tokens, pad_mask)
        else:
            tgt_tokens, token_probs, attentions = self.generate_non_autoregressive(model, inputs_for_decoder, tgt_tokens, pad_mask)
        
        # if we use coarse-grained templates, it will take one more iteration
        T = self.T + 1 if self.use_ct else self.T
              
        self.print_sent(tgt_tokens, token_probs, counter=0)
        self.collect_data(tgt_tokens, token_probs, attentions)

        for counter in range(1, T):
            corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_inputs_for_decoder, tgt_tokens, pad_mask, is_last=False)

            if self.use_ct and counter == 1:
                # if we use coarse-grained templates, we first complete the sequences
                # i.e., sentence making in Fig. 1(b) in the paper
                mask_ind = (tgt_tokens == Constants.MASK)
            else:
                ratio = (1.0 - (counter / T))
                num_mask = (seq_lens.float() * ratio).long()
                mask_ind = self.select_worst(token_probs * corresponding_probs, num_mask)

            # Mask
            tgt_tokens[mask_ind] = Constants.MASK
            # Predict
            new_tgt_tokens, new_token_probs, attentions = self.generate_non_autoregressive(model, inputs_for_decoder, tgt_tokens, pad_mask)
            # only update those masked tokens and their possibilities
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]
            token_probs[mask_ind] = new_token_probs[mask_ind]

            self.print_sent(tgt_tokens, token_probs, counter=counter)
            self.collect_data(tgt_tokens, token_probs, attentions, is_last=True if counter==T-1 else False)

        # teacher rescoring
        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_inputs_for_decoder, tgt_tokens, pad_mask, is_last=True)
        lprobs = (token_probs * corresponding_probs).log()
        return tgt_tokens, lprobs, self.get_collected_data()

class Left2Right(Algorithm_Base):
    def __init__(self, opt, dict_mapping, tgt_vocab):
        super().__init__(opt, dict_mapping, tgt_vocab)
        self.use_ct = opt.get('use_ct', False)
        self.T = opt.get('q_iterations', 1)
        self.q = opt.get('q', 1)

    def generate(self, model, teacher_model, inputs_for_decoder, teacher_inputs_for_decoder, tgt_tokens):
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        
        attentions = None
        if self.use_ct:
            tgt_tokens, token_probs, attentions = self.get_coarse_grained_templates(model, inputs_for_decoder, tgt_tokens, pad_mask)
            visual_mask = tgt_tokens.ne(Constants.MASK) & tgt_tokens.ne(Constants.PAD)
        else:
            token_probs = tgt_tokens.new(*tgt_tokens.shape).fill_(0).float()
            token_probs[pad_mask] = 1.0

        self.collect_data(tgt_tokens, token_probs, attentions)

        def get_mask_ind(tgt_tokens, seq_lens):
            all_mask_ind = []
            for i in range(tgt_tokens.size(0)):
                item = [j for j in range(seq_lens[i]) if tgt_tokens[i, j] == Constants.MASK]
                all_mask_ind.append(item)
            return all_mask_ind

        def select_left(all_mask_ind, current, q):
            masks = torch.zeros(*token_probs.shape, device=token_probs.device)
            for i in range(masks.size(0)):
                ind = all_mask_ind[i][current:min(current+q,len(all_mask_ind[i]))] if current < len(all_mask_ind[i]) else []
                masks[i, ind] = 1
            return masks.bool()

        all_mask_ind = get_mask_ind(tgt_tokens, seq_lens)

        for counter in range(0, seq_len, self.q):
            mask_ind = select_left(all_mask_ind, counter, self.q)
            if mask_ind.sum() == 0: break

            tgt_tokens[mask_ind] = Constants.MASK
            # Predict
            new_tgt_tokens, new_token_probs, attentions = self.generate_non_autoregressive(model, inputs_for_decoder, tgt_tokens, pad_mask)
            
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]
            
            self.collect_data(tgt_tokens, token_probs, attentions)

        for i in range(self.T):
            if i == 0 and self.use_ct:
                mask_ind = visual_mask
            else:
                refine_ratio = 0.4 * (1.0 - (i / self.T))
                num_mask = (seq_lens.float() * refine_ratio).long()
                mask_ind = self.select_worst(token_probs, num_mask)

            tgt_tokens[mask_ind] = Constants.MASK
            new_tgt_tokens, new_token_probs, attentions = self.generate_non_autoregressive(model, inputs_for_decoder, tgt_tokens, pad_mask)

            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]
            self.collect_data(tgt_tokens, token_probs, attentions)

        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_inputs_for_decoder, tgt_tokens, pad_mask, is_last=True)
        lprobs = (token_probs * corresponding_probs).log()

        return tgt_tokens, lprobs, self.get_collected_data()


class EasyFirst(Algorithm_Base):
    def __init__(self, opt, dict_mapping, tgt_vocab):
        super().__init__(opt, dict_mapping, tgt_vocab)
        self.use_ct = opt.get('use_ct', False)
        self.T = opt.get('q_iterations', 1)
        self.q = opt.get('q', 1)

    def generate(self, model, teacher_model, inputs_for_decoder, teacher_inputs_for_decoder, tgt_tokens):
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        
        attentions = None
        if self.use_ct:
            tgt_tokens, token_probs, attentions = self.get_coarse_grained_templates(model, inputs_for_decoder, tgt_tokens, pad_mask)
            visual_mask = tgt_tokens.ne(Constants.MASK) & tgt_tokens.ne(Constants.PAD)
        else:
            token_probs = tgt_tokens.new(*tgt_tokens.shape).fill_(0).float()
            token_probs[pad_mask] = 1.0

        self.collect_data(tgt_tokens, token_probs, attentions)

        def select_most_confidence(token_probs, mask_ind, q):
            masks = torch.zeros(*token_probs.shape, device=token_probs.device)
            token_probs[~mask_ind] = 0
            remain_length = mask_ind.sum(-1)
            for i in range(masks.size(0)):
                if remain_length[i] == 0:
                    ind = []
                else:
                    ind = token_probs[i, :].topk(min(q, remain_length[i]), largest=True, sorted=False)[1]
                masks[i, ind] = 1
            return masks.bool()

        counter, pre = 0, 0
        while True:
            counter += 1
            mask_ind = tgt_tokens.eq(Constants.MASK)

            remain = mask_ind.sum()
            if remain == 0 or pre == remain: # to avoid dead loop
                break
            pre = remain

            new_tgt_tokens, new_token_probs, attentions = self.generate_non_autoregressive(model, inputs_for_decoder, tgt_tokens, pad_mask)

            most_confidence_ind = select_most_confidence(new_token_probs, mask_ind, self.q)
            token_probs[most_confidence_ind] = new_token_probs[most_confidence_ind]
            tgt_tokens[most_confidence_ind] = new_tgt_tokens[most_confidence_ind]
            
            self.collect_data(tgt_tokens, token_probs, attentions)

        
        for i in range(self.T):
            if i == 0 and self.use_ct:
                mask_ind = visual_mask
            else:
                refine_ratio = 0.4 * (1.0 - (i / self.T))
                num_mask = (seq_lens.float() * refine_ratio).long()
                mask_ind = self.select_worst(token_probs, num_mask)

            tgt_tokens[mask_ind] = Constants.MASK
            new_tgt_tokens, new_token_probs, attentions = self.generate_non_autoregressive(model, inputs_for_decoder, tgt_tokens, pad_mask)
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]

            self.collect_data(tgt_tokens, token_probs, attentions)
        
        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_inputs_for_decoder, tgt_tokens, pad_mask, is_last=True)
        lprobs = (token_probs * corresponding_probs).log()

        return tgt_tokens, lprobs, self.get_collected_data()