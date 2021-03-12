"""PyTorch BERT model. """
import math
import sys
import torch
from torch import nn
import torch.nn.functional as F
from config import Constants

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_new(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}


BertLayerNorm = torch.nn.LayerNorm


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config, return_pos=False):
        super(BertEmbeddings, self).__init__()
        if getattr(config, 'load_word_embeddings', False):
            self.word_embeddings = nn.Embedding(config.vocab_size, 768, padding_idx=Constants.PAD)
            self.word_embeddings_prj = nn.Linear(768, config.dim_hidden)
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.dim_hidden, padding_idx=Constants.PAD)


        self.position_embeddings = nn.Embedding(config.max_len, config.dim_hidden)
        self.category_embeddings = nn.Embedding(config.num_category, config.dim_hidden) if config.with_category else None
        self.return_pos = return_pos

        self.LayerNorm = BertLayerNorm(config.dim_hidden, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.return_pos:
            self.pos_LN = BertLayerNorm(config.dim_hidden, eps=config.layer_norm_eps)
            self.pos_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.tag2hidden = None

    def forward(self, input_ids, category=None, position_ids=None, additional_feats=None, tags=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        if getattr(self, 'word_embeddings_prj', None) is not None:
            words_embeddings = self.word_embeddings_prj(words_embeddings)

        #words_embeddings = self.prj(self.word_embeddings(input_ids))
        position_embeddings = self.position_embeddings(position_ids)
        if self.category_embeddings is not None:
            assert category is not None
            category_embeddings = self.category_embeddings(category).repeat(1, words_embeddings.size(1), 1)

        if not self.return_pos:
            embeddings = words_embeddings + position_embeddings
            if self.category_embeddings is not None:
                embeddings = embeddings + category_embeddings
            
            if additional_feats is not None:
                embeddings = embeddings + additional_feats

            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            return embeddings
        else:
            embeddings = words_embeddings + position_embeddings
            if self.category_embeddings is not None:
                embeddings += category_embeddings

            if additional_feats is not None:
                embeddings += additional_feats

            embeddings = self.dropout(self.LayerNorm(embeddings))
            position_embeddings = self.pos_dropout(self.pos_LN(position_embeddings))

            return embeddings, position_embeddings

    def linear(self, x):
        x = x.matmul(self.word_embeddings.weight.t()) # [batch_size, vocab_size]
        return x


class BertSelfAttention(nn.Module):
    def __init__(self, config, attend_to_enc_output=False):
        super(BertSelfAttention, self).__init__()
        if config.dim_hidden % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.dim_hidden, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.dim_hidden / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.dim_hidden, self.all_head_size)
        if attend_to_enc_output and getattr(config, 'modality_wise_dimensions', False):
            self.key = nn.Linear(config.dim_hidden * len(config.modality), self.all_head_size)
            self.value = nn.Linear(config.dim_hidden * len(config.modality), self.all_head_size)
        else:
            self.key = nn.Linear(config.dim_hidden, self.all_head_size)
            self.value = nn.Linear(config.dim_hidden, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.use_sigmoid_to_get_attprob = getattr(config, 'use_sigmoid_to_get_attprob', False)


    def forward(self, q, k, v, attention_mask, head_mask=None, output_attentions=False):
        d_k, d_v, n_head = self.attention_head_size, self.attention_head_size, self.num_attention_heads

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        q = self.query(q).view(sz_b, len_q, n_head, d_k)
        k = self.key(k).view(sz_b, len_k, n_head, d_k)
        v = self.value(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if attention_mask is not None:
            attention_mask = attention_mask.repeat(n_head, 1, 1) # (n*b) x .. x ..

        attention_scores = torch.bmm(q, k.transpose(1, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask, -10e6)

        if self.use_sigmoid_to_get_attprob:
            attention_probs = torch.sigmoid(attention_scores)
            attention_probs = attention_probs / (attention_probs.sum(-1, keepdims=True) + 1e-12)
        else:
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        outputs = torch.bmm(attention_probs, v)

        outputs = outputs.view(n_head, sz_b, len_q, d_v)
        outputs = outputs.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        return (outputs, attention_probs.view(n_head, sz_b, len_q, len_k)) if output_attentions else (outputs,)


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        if config.num_attention_heads == 1 and getattr(config, 'no_attention_dense', False):
            self.dense = None
        else:
            self.dense = nn.Linear(config.dim_hidden, config.dim_hidden)
        self.LayerNorm = BertLayerNorm(config.dim_hidden, eps=config.layer_norm_eps) if config.with_layernorm else None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, input_tensor=None):
        if self.dense is not None:
            hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if input_tensor is not None:
            hidden_states = hidden_states + input_tensor
        if self.LayerNorm is not None:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, pos=False, with_residual=True, attend_to_enc_output=False):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config, attend_to_enc_output)
        self.output = BertSelfOutput(config)
        self.pos = pos
        self.with_residual = with_residual

    def forward(self, q, k, v, attention_mask, head_mask=None, output_attentions=False):
        self_outputs = self.self(q, k, v, attention_mask, head_mask, output_attentions=output_attentions)
        attention_output = self.output(self_outputs[0], q if self.with_residual else None) #q if not self.pos else v
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.dim_hidden, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.dim_hidden)
        self.LayerNorm = BertLayerNorm(config.dim_hidden, eps=config.layer_norm_eps) if config.with_layernorm else None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        if self.LayerNorm is not None:
            hidden_states = self.LayerNorm(hidden_states)

        return self.dropout(hidden_states)


class BertLayer(nn.Module):
    def __init__(self, config, is_decoder_layer=False):
        super(BertLayer, self).__init__()
        is_parallel_mlm = getattr(config, 'parallel_mlm', False)
        self.attention = BertAttention(config, with_residual=False if is_parallel_mlm else True)
        self.pos_attention = None if not (config.pos_attention and is_decoder_layer) \
            else BertAttention(config, pos=True)
        self.attend_to_enc_output = None if not is_decoder_layer \
            else BertAttention(config, attend_to_enc_output=True)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, non_pad_mask=None, attention_mask=None, 
        enc_output=None, attend_to_enc_output_mask=None,
        attr_probs=None, attend_to_attributes_mask=None, video2attr_raw_scores=None, word_embeddings=None,
        output_attentions=False, head_mask=None, position_embeddings=None, **kwargs):
        all_attentions = ()
        attention_outputs = self.attention(hidden_states, hidden_states, hidden_states, attention_mask, head_mask, output_attentions=output_attentions)
        attention_output = attention_outputs[0]
        all_attentions += attention_outputs[1:]

        if non_pad_mask is not None:
            attention_output = attention_output * non_pad_mask

        if self.pos_attention is not None:
            assert position_embeddings is not None
            attention_outputs = self.pos_attention(position_embeddings, position_embeddings, attention_output, attention_mask, head_mask, output_attentions=output_attentions)
            attention_output = attention_outputs[0]
            all_attentions += attention_outputs[1:]
            if non_pad_mask is not None:
                attention_output = attention_output * non_pad_mask

        if self.attend_to_enc_output is not None:
            assert attend_to_enc_output_mask is not None
            assert enc_output is not None

            attention_outputs = self.attend_to_enc_output(
                attention_output, enc_output, enc_output, 
                attend_to_enc_output_mask, head_mask, 
                output_attentions=output_attentions
            )
            attention_output = attention_outputs[0]
            all_attentions += attention_outputs[1:]
            if non_pad_mask is not None:
                attention_output = attention_output * non_pad_mask

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if non_pad_mask is not None:
            layer_output *= non_pad_mask

        embs = layer_output.sum(1) / non_pad_mask.sum(1)
        outputs = (layer_output, embs,) + (all_attentions, )  # add attentions if we output them
        return outputs


