import torch
import numpy as np
from config import Constants

class Beam():
    ''' Beam search '''

    def __init__(self, size, max_len, device=False, all_init=False, specific_nums_of_sents=0):

        self.size = size
        self.specific_nums_of_sents = max(self.size, specific_nums_of_sents)
        self._done = False
        self.max_len=max_len

        # The score for each translation on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.full((size,), Constants.PAD, dtype=torch.long, device=device)]
        if all_init:
            # LSTM
            for i in range(size):
                self.next_ys[0][i] = Constants.BOS
        else:
            # ARFormer
            self.next_ys[0][0] = Constants.BOS
        self.finished = []

    def get_current_state(self):
        "Get the outputs till the current timestep."
        return self.get_tentative_hypothesis(so_far=True)

    def get_lastest_state(self):
        "Get the outputs at latest timestep"
        return self.get_tentative_hypothesis(so_far=False)         


    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))



    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def append_one_item(self, content):
        self.finished.append(content)
        if len(self.finished) >= self.specific_nums_of_sents:
            return True
        else:
            return False

    def advance(self, word_prob):
        "Update beam status and check if finished or not."
        num_words = word_prob.size(1)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == Constants.EOS:
                    beam_lk[i] = -1e20
        else:
            beam_lk = word_prob[0]

        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        #print('PREV_KS:', self.prev_ks)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == Constants.EOS:
                self._done = self.append_one_item([self.scores[i].item(), len(self.next_ys) - 1, i])
            if self._done:
                return self._done

        # End condition is when top-of-beam is EOS.

        '''
        if self.next_ys[-1][0].item() == Constants.EOS or len(self.next_ys) == self.max_len:
            self._done = True
            self.all_scores.append(self.scores)
            if not len(self.finished):
                for i in range(self.next_ys[-1].size(0)):
                    self.finished.append([self.scores[i].item(), len(self.next_ys) - 1, i])
        '''
        if len(self.next_ys) == self.max_len:
            self._done = True
            self.all_scores.append(self.scores)
            if not len(self.finished):
                for i in range(self.next_ys[-1].size(0)):
                    self.append_one_item([self.scores[i].item(), len(self.next_ys) - 1, i])
        return self._done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def sort_finished(self, alpha=1.0):
        #pelnety
        for item in self.finished:
            item[0] /= item[1]**alpha
        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        tk = [(t, k) for _, t, k in self.finished]
        return scores, tk

    def get_hypothesis_from_tk(self, timestep, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self, so_far=True):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k, so_far) for k in keys]
            if so_far:
                hyps = [[Constants.BOS] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k, so_far=True):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]
            if not so_far:
                break

        return list(map(lambda x: x.item(), hyp[::-1]))
