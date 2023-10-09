"""
transformer model (text only?)

"""


import copy
import math
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fairseq.data import Dictionary

from fairseq.models import FairseqEncoder, FairseqIncrementalDecoder, transformer, FairseqEncoderDecoderModel, \
    register_model_architecture, register_model
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import FairseqDropout, PositionalEmbedding
from fairseq.modules.adaptive_softmax import AdaptiveSoftmax

import fairseq.models.transformer
import fairseq.utils

# http://nlp.seas.harvard.edu/2018/04/03/attention.html
from torch.nn import Parameter


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# _max_size = 100
# _mask_cache = {}
# def subsequent_mask(size, device):
#     "Mask out subsequent positions."
#     global _max_size
#     key = device.type
#     if key not in _mask_cache or (key in _mask_cache and size > _mask_cache[key].shape[0]):
#         _max_size = max(_max_size, size)
#         _mask_cache[key] = torch.triu(torch.ones(1, _max_size, _max_size, dtype=torch.uint8)) == 0
#     return _mask_cache[key][:, :size, :size]

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    """
        mask: 0 -> mask out; 1 -> keep
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # scores = scores.masked_fill(mask == 0, -1e9)
        # when using fp16: value cannot be converted to type at::Half without overflow: -1e+09  # FIXME: ???
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


class TransformerEncoderText(FairseqEncoder):
    def __init__(self, args, dictionary: Dictionary, embed_tokens):
        super().__init__(dictionary)

        n_head = args.encoder_attention_heads
        d_model = args.encoder_embed_dim
        d_feedforward = args.encoder_ffn_embed_dim
        n_layer = args.encoder_layers
        dropout = args.dropout
        self.output_embed_dim = args.decoder_output_dim

        c = copy.deepcopy
        attn = MultiHeadedAttention(h=n_head, d_model=d_model)
        ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_feedforward)
        position = PositionalEncoding(d_model=d_model, dropout=dropout)

        self.embed_tokens = embed_tokens

        self.encoder = Encoder(
            layer=EncoderLayer(size=d_model, self_attn=c(attn), feed_forward=c(ff), dropout=dropout),
            N=n_layer
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.embed_positions = position

    def forward(self, src_tokens, src_lengths, **kwargs):
        """
        :param src_tokens: (batch_size, emb_dim)
        :param src_lengths:
        :param kwargs:
        :return:
        """

        x = self.embed_tokens(src_tokens)
        x = self.embed_positions(x)

        _range = torch.arange(0, src_tokens.shape[1], device=x.device).unsqueeze(0).expand_as(src_tokens)
        src_mask = _range < src_lengths.unsqueeze(1).expand_as(_range)
        src_mask = src_mask.unsqueeze(-2)

        memory = self.encoder.forward(x, src_mask)

        return {'memory': memory, 'src_mask': src_mask}

    def reorder_encoder_out(self, encoder_out, new_order):
        memory, src_mask = [encoder_out[key] for key in ('memory', 'src_mask')]
        memory = memory.index_select(dim=0, index=new_order)
        src_mask = src_mask.index_select(dim=0, index=new_order)
        return {'memory': memory, 'src_mask': src_mask}


class TransformerDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary: Dictionary, embed_tokens, **kwargs):
        super().__init__(dictionary)

        n_head = args.decoder_attention_heads
        d_model = args.decoder_embed_dim
        d_feedforward = args.decoder_ffn_embed_dim
        n_layer = args.decoder_layers
        dropout = args.dropout
        self.output_embed_dim = args.decoder_output_dim

        c = copy.deepcopy
        attn = MultiHeadedAttention(h=n_head, d_model=d_model)
        ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_feedforward)
        position = PositionalEncoding(d_model=d_model, dropout=0.)

        self.embed_tokens = embed_tokens

        self.decoder = Decoder(
            layer=DecoderLayer(size=d_model, self_attn=c(attn), src_attn=c(attn), feed_forward=c(ff), dropout=dropout),
            N=n_layer
        )

        self.adaptive_softmax = None
        self.output_projection = None
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                fairseq.utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.embed_positions = position

    def output_layer(self, features, **kwargs):
        if self.adaptive_softmax is None:
            return self.output_projection(features)
        else:
            return features

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        """
        :param prev_output_tokens: (batch_size, seq_len)
        :param encoder_out: 'memory': (), 'src_mask': ()
        :param incremental_state:
        :param kwargs:
        :return:
        """
        memory, src_mask = encoder_out['memory'], encoder_out['src_mask']
        dec_input = self.embed_tokens(prev_output_tokens)
        dec_input = self.embed_positions(dec_input)

        # if incremental_state is not None:
        #     prev_output_tokens = prev_output_tokens[:, -1:]

        # feed all output tokens to decoder
        tgt_mask = prev_output_tokens != self.dictionary.pad_index
        tgt_mask = tgt_mask.unsqueeze(-2)
        _future_mask = subsequent_mask(prev_output_tokens.shape[1]).to(tgt_mask)
        tgt_mask = tgt_mask & _future_mask

        dec_out = self.decoder(dec_input, memory=memory, src_mask=src_mask, tgt_mask=tgt_mask)
        if incremental_state is not None:  # evaluation
            dec_out = dec_out[:, -1:]

        dec_out = self.output_layer(dec_out)

        # output[1] must have 'attn' key
        additional = {'attn': None}

        return dec_out, additional

    def reorder_incremental_state(self, incremental_state, new_order):
        pass


@register_model('transformer1')
class TransformerModel(FairseqEncoderDecoderModel):
    @staticmethod
    def add_args(parser, **kwargs):
        """
        n_head = args.decoder_attention_heads
        d_model = args.decoder_embed_dim
        d_feedforward = args.decoder_ffn_embed_dim
        n_layer = args.decoder_layers
        dropout = args.dropout
        self.output_embed_dim = args.decoder_output_dim
        """
        parser.add_argument('--encoder_embed_dim', type=int, default=512)
        parser.add_argument('--encoder_ffn_embed_dim', type=int, default=2048)
        parser.add_argument('--encoder_layers', type=int, default=3)
        parser.add_argument('--encoder_attention_heads', type=int, default=8)

        parser.add_argument('--decoder_embed_dim', type=int, default=512)
        parser.add_argument('--decoder_ffn_embed_dim', type=int, default=2048)
        parser.add_argument('--decoder_layers', type=int, default=3)
        parser.add_argument('--decoder_attention_heads', type=int, default=8)

        parser.add_argument('--dropout', type=float, default=0.1)

        parser.add_argument('--adaptive-softmax-cutoff', type=str, default='[5000,20000]')
        parser.add_argument('--adaptive-softmax-dropout', type=float, default=0)
        parser.add_argument('--adaptive-softmax-factor', type=float, default=4)

        parser.add_argument('--tie-adaptive-weights', type=int, default=0)
        parser.add_argument('--tie-adaptive-proj', type=int, default=0)

    @classmethod
    def build_model(cls, args, task, **kwargs):
        dictionary = kwargs.get('dictionary', None)
        if dictionary is None:
            dictionary = task.dictionary

        encoder_embed_tokens = Embedding(len(dictionary), args.encoder_embed_dim, dictionary.pad())
        decoder_embed_tokens = encoder_embed_tokens

        encoder = TransformerEncoderText(args, dictionary, encoder_embed_tokens)
        decoder = TransformerDecoder(args, dictionary, decoder_embed_tokens)

        model = TransformerModel(encoder, decoder)
        model.args = args
        model.dictionary = dictionary
        return model

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return super().parameters(recurse)
        # return self.decoder.parameters(recurse)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    def get_normalized_probs(
        self,
        net_output,
        log_probs,
        sample=None,
    ):
        return self.decoder.get_normalized_probs(net_output, log_probs, sample)


@register_model_architecture('transformer1', 'transformer1')
def transformer1(args):
    """

    :param args:
    :return:
    """
    args.n_layer = getattr(args, 'encoder_embed_dim', 512)
    args.n_layer = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.n_layer = getattr(args, 'encoder_layers', 3)
    args.n_layer = getattr(args, 'encoder_attention_heads', 3)

    args.n_layer = getattr(args, 'decoder_embed_dim', 512)
    args.n_layer = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.n_layer = getattr(args, 'decoder_layers', 3)
    args.n_layer = getattr(args, 'decoder_attention_heads', 3)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )

    args.n_layer = getattr(args, 'dropout', 0.1)

    args.n_layer = getattr(args, 'adaptive_softmax_cutoff', None)
    args.n_layer = getattr(args, 'adaptive_softmax_dropout', 0)
    args.n_layer = getattr(args, 'adaptive_softmax_factor', 4.0)

    # args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)


def main():
    att_feat = torch.rand(16, 20, 2048)
    att_mask = torch.ones(16, 20)
    dictionary = Dictionary()
    for sym in ['a', 'b', 'c']: dictionary.add_symbol(sym)
    enc = TransformerEncoderText(None, dictionary, n_layer=1, d_model=512, d_feedforward=2048, n_head=8, dropout=0.1)
    dec = TransformerDecoder(None, dictionary, n_layer=1, d_model=512, d_feedforward=2048, n_head=8, dropout=0.1)

    encoder_out = enc.forward(None, None, feat_att=att_feat, att_mask=att_mask)
    prev_output_tokens = torch.randint(low=0, high=5, size=(16, 5))
    decoder_out = dec.forward(prev_output_tokens, encoder_out)

    print('encoder_out:', encoder_out[0].shape, encoder_out[1].shape)
    print('decoder_out:', decoder_out.shape)


if __name__ == '__main__':
    main()