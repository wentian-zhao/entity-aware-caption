"""
transformer model (text + image)

"""

import copy
import math
import time
from typing import Iterator, overload, Dict, Union
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fairseq import options
from fairseq.data import Dictionary

from fairseq.models import FairseqEncoder, FairseqIncrementalDecoder, transformer, FairseqEncoderDecoderModel, \
    register_model_architecture, register_model
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import FairseqDropout, PositionalEmbedding, AdaptiveInput
from fairseq.modules.adaptive_softmax import AdaptiveSoftmax
from fairseq.modules.dynamic_convolution import Linear
from torch import Tensor
from torch.nn.modules.module import _IncompatibleKeys

from model.dynamic_conv_decoder import LightConvDecoderMM
from model.gat import GraphEncoder
from .resnet import resnet152

from model.vit.models.modeling import VisionTransformer, CONFIGS

from config import roberta_path

import fairseq.models.transformer
import fairseq.models.lightconv
import fairseq.utils

from fairseq.models.roberta import RobertaModel
from torch.nn import Parameter

from fairseq.logging import metrics


def create_padding_mask(src_tokens, src_lengths):
    padding_mask = torch.zeros(src_tokens.shape[:2],
                               dtype=torch.bool,
                               device=src_tokens.device)

    for i, src_length in enumerate(src_lengths):
        padding_mask[i, src_length:] = 1

    return padding_mask


class TransformerFeatureExtractor(FairseqEncoder):
    def _forward_return_encoder_out(self, encoder_out, enc_padding_mask):
        """

        :param encoder_out: (src_len, batch_size, emb_dim)
        :param enc_padding_mask: (batch_size, src_len)
        :return:
        """
        _encoder_out = transformer.EncoderOut(
            encoder_out=encoder_out,
            encoder_padding_mask=enc_padding_mask,
            encoder_embedding=None,
            encoder_states=None,
            src_lengths=None,
            src_tokens=None
        )
        if self.output_mode == 'transformer':
            return _encoder_out
        else:
            return {
                'encoder_out': encoder_out,
                'encoder_padding_mask': enc_padding_mask,
                'transformer.EncoderOut': _encoder_out
            }

    def reorder_encoder_out(self, encoder_out, new_order):
        if self.output_mode != 'transformer':
            encoder_out = encoder_out['transformer.EncoderOut']

        encoder_padding_mask = encoder_out.encoder_padding_mask
        encoder_embedding = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out if encoder_out.encoder_out is None else encoder_out.encoder_out.index_select(1,
                                                                                                                 new_order))
        new_encoder_padding_mask = (
            encoder_padding_mask if encoder_padding_mask is None else encoder_padding_mask.index_select(0, new_order))
        new_encoder_embedding = (
            encoder_embedding if encoder_embedding is None else encoder_embedding.index_select(0, new_order))

        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        _encoder_out = EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )

        if self.output_mode == 'transformer':
            return _encoder_out
        else:
            return {
                'encoder_out': new_encoder_out,
                'encoder_padding_mask': new_encoder_padding_mask,
                'transformer.EncoderOut': _encoder_out
            }


import logging
logger = logging.getLogger(__name__)
def roberta_from_pretrained(cls, model_name_or_path, checkpoint_file="model.pt", data_name_or_path=".", **kwargs,):
    """
    replace fairseq.models.roberta.RobertaModel.from_pretrained
    :param cls:
    :param model_name_or_path:
    :param checkpoint_file:
    :param data_name_or_path:
    :param kwargs:
    :return:
    """
    from fairseq import hub_utils
    x = hub_utils.from_pretrained(
        model_name_or_path,
        checkpoint_file,
        data_name_or_path,
        archive_map=cls.hub_models(),
        **kwargs,
    )
    cls.upgrade_args(x["args"])
    logger.info(x["args"])
    return hub_utils.GeneratorHubInterface(x["args"], x["task"], x["models"])


class RobertaWrapper(TransformerFeatureExtractor):
    def __init__(self, args, dictionary, output_mode='transformer'):
        super().__init__(dictionary)
        print('init roberta')

        roberta = RobertaModel.from_pretrained(roberta_path, checkpoint_file='model.pt')
        # roberta = roberta_from_pretrained(RobertaModel, roberta_path, checkpoint_file='model.pt')
        roberta.eval()  # disable dropout (or leave in train mode to finetune)
        self.roberta = roberta

        print('finish init roberta')

        self.encoder_embed_dim = args.encoder_embed_dim
        # if self.encoder_embed_dim == 1024:
        #     pass
        # else:
        #     self.encoder_emb = nn.Linear(1024, self.encoder_embed_dim)
        self.encoder_emb = nn.Linear(1024, self.encoder_embed_dim)

        for param in roberta.parameters():
            param.requires_grad = False
        self.output_mode = output_mode

        self.fuse_option = args.roberta_fuse
        if self.fuse_option == 'weighted':
            self.bert_weight = nn.Parameter(torch.Tensor(25))
            nn.init.uniform_(self.bert_weight)
            self.register_parameter('bert_weight', self.bert_weight)

    @metrics.aggregate('train')
    def forward(self, src_tokens, src_lengths=None, **kwargs):
        # torch.cuda.synchronize();
        t0 = time.time()

        self.roberta.eval()

        if self.fuse_option == 'last':
            with torch.no_grad():
                last_layer_features = self.roberta.extract_features(src_tokens)  # (batch_size, src_len, 1024)
                last_layer_features = last_layer_features.detach()
                encoder_out = last_layer_features
        elif self.fuse_option == 'mean':                # seems like this also works
            with torch.no_grad():
                all_layer_features = self.roberta.extract_features(src_tokens, return_all_hiddens=True)
                all_layer_features = torch.stack(all_layer_features, dim=2)         # (batch_size, src_len, 25, 1024)
                encoder_out = all_layer_features.mean(dim=2)
        elif self.fuse_option == 'weighted':            # slow
            with torch.no_grad():
                all_layer_features = self.roberta.extract_features(src_tokens, return_all_hiddens=True)
                all_layer_features = torch.stack(all_layer_features, dim=2).detach()  # (batch_size, src_len, 25, 1024)

            weight = F.softmax(self.bert_weight, dim=0)
            weight = weight.unsqueeze(0).unsqueeze(1).unsqueeze(3)  # (1, 1, 25, 1)
            encoder_out = (all_layer_features * weight).sum(dim=2)

        # with torch.no_grad():
        #     if self.weighted:
        #         all_layer_features = self.roberta.extract_features(src_tokens, return_all_hiddens=True)
        #         weight = F.softmax(self.bert_weight, dim=0)
        #         weight = weight.unsqueeze(0).unsqueeze(1).unsqueeze(3)              # (1, 1, 25, 1024)
        #
        #         all_layer_features = torch.stack(all_layer_features, dim=2)         # (batch_size, src_len, 25, 1024)
        #         encoder_out = (all_layer_features * weight).sum(dim=2)
        #     else:
        #         last_layer_features = self.roberta.extract_features(src_tokens)     # (batch_size, src_len, 1024)
        #         last_layer_features = last_layer_features.detach()
        #         encoder_out = last_layer_features

        encoder_out = encoder_out.transpose(0, 1)                           # (src_len, batch_size, 512)

        enc_padding_mask = create_padding_mask(src_tokens, src_lengths)     # (batch_size, src_len)

        # torch.cuda.synchronize();
        t1 = time.time()
        t_forward = t1 - t0
        if self.training:
            metrics.log_scalar('roberta_forward', t_forward)

        return self._forward_return_encoder_out(encoder_out, enc_padding_mask)

    def reorder_encoder_out(self, encoder_out, new_order):
        return super().reorder_encoder_out(encoder_out, new_order)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if self.weighted:
            return iter([self.bert_weight])
        else:
            return self.encoder_emb.parameters()

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        super().state_dict(destination, prefix, keep_vars)
        keys = list(destination.keys())
        for key in keys:
            if key.startswith(prefix + 'roberta'):
                del destination[key]
        return destination


class ResNetWrapper(TransformerFeatureExtractor):
    def __init__(self, args, dictionary, output_mode='transformer'):
        super().__init__(dictionary)
        self.C = 2048
        self.resnet = resnet152()
        self.output_mode = output_mode

        self.encoder_embed_dim = args.encoder_embed_dim
        self.encoder_emb = nn.Linear(2048, self.encoder_embed_dim)

    @metrics.aggregate('train')
    def forward(self, src_tokens, src_lengths=None, **kwargs):
        # torch.cuda.synchronize();
        t0 = time.time()

        image = kwargs['image']             # (batch_size, 3, 224, 224)
        batch_size = image.shape[0]

        self.resnet.eval()
        with torch.no_grad():
            image_feat = self.resnet(image)     # (batch_size, 2048, 7, 7)
        image_feat = image_feat.reshape(batch_size, 2048, -1)       # (batch_size, 2048, 49)
        n_feat = image_feat.shape[-1]
        image_feat = image_feat.permute(2, 0, 1)                    # (49, batch_size, 2048)
        # image_feat = self.encoder_emb(image_feat)

        encoder_out = image_feat
        enc_padding_mask = torch.zeros((batch_size, n_feat), device=src_tokens.device, dtype=torch.bool)

        # torch.cuda.synchronize();
        t1 = time.time()
        t_forward = t1 - t0
        if self.training:
            metrics.log_scalar('resnet_forward', t_forward)

        return self._forward_return_encoder_out(encoder_out, enc_padding_mask)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.encoder_emb.parameters()

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        super().state_dict(destination, prefix, keep_vars)
        keys = list(destination.keys())
        for key in keys:
            if key.startswith(prefix + 'resnet'):
                del destination[key]
        return destination


class ViTWrapper(TransformerFeatureExtractor):
    def __init__(self, args, dictionary, output_mode='transformer'):
        super().__init__(dictionary)
        self.C = 768

        config = CONFIGS["ViT-B_16"]
        model = VisionTransformer(config, num_classes=1000, zero_head=False, img_size=224, vis=True)
        model.load_from(np.load("/home/wentian/work/extract_image_feature/imagenet21k+imagenet2012_ViT-B_16-224.npz"))
        model.eval()
        self.vit = model

        self.output_mode = output_mode
        self.encoder_embed_dim = args.encoder_embed_dim
        self.encoder_emb = nn.Linear(self.C, self.encoder_embed_dim)

    @metrics.aggregate('train')
    def forward(self, src_tokens, src_lengths=None, **kwargs):
        t0 = time.time()

        image = kwargs['image']             # (batch_size, 3, 224, 224)
        batch_size = image.shape[0]

        self.vit.eval()
        with torch.no_grad():
            logits, image_feat = self.vit.forward_feat(image)     # (batch_size, 196, 768)
        n_feat = image_feat.shape[1]                 # 196
        image_feat = image_feat.permute(1, 0, 2)        # (196, batch_size, 768)
        encoder_out = image_feat
        enc_padding_mask = torch.zeros((batch_size, n_feat), device=src_tokens.device, dtype=torch.bool)

        t1 = time.time()
        t_forward = t1 - t0
        if self.training:
            metrics.log_scalar('vit_forward', t_forward)

        return self._forward_return_encoder_out(encoder_out, enc_padding_mask)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.encoder_emb.parameters()

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        super().state_dict(destination, prefix, keep_vars)
        keys = list(destination.keys())
        for key in keys:
            if key.startswith(prefix + 'vit'):
                del destination[key]
        return destination


class GraphEncoderWrapper(TransformerFeatureExtractor):
    def __init__(self, args, dictionary, output_mode='transformer'):
        super().__init__(dictionary)
        self.use_image_graph = args.use_image_graph
        self.use_text_graph = args.use_text_graph
        self.image_graph_max_size = args.image_graph_max_size
        self.text_graph_max_size = args.text_graph_max_size

        self.graph_max_size = 0
        self.image_graph_start = 0
        if self.use_text_graph:
            self.graph_max_size += self.text_graph_max_size
        if self.use_image_graph:
            self.graph_max_size += self.image_graph_max_size

        self.encoder_embed_dim = args.encoder_embed_dim
        self.roberta_emb_dim = 1024
        self.output_mode = output_mode

        self.d_model = self.roberta_emb_dim

        self.gat = GraphEncoder(d_model=self.roberta_emb_dim, n_layer=args.gat_layers, n_head=args.gat_heads)

        if self.use_image_graph:
            self.obj_feat_emb = nn.Linear(in_features=2048, out_features=self.d_model)
            self.face_feat_emb = nn.Linear(in_features=512, out_features=self.d_model)

    @metrics.aggregate('train')
    def forward(self, src_tokens, src_lengths=None, **kwargs):
        # torch.cuda.synchronize();
        t0 = time.time()

        roberta_output = kwargs['roberta_output']
        encoder_out = roberta_output['encoder_out']  # (seq_len, batch_size, emb_dim)

        seq_len, batch_size, emb_dim = encoder_out.shape
        node_emb = encoder_out.new_zeros(batch_size, self.graph_max_size, emb_dim)
        enc_padding_mask = np.zeros(shape=(batch_size, self.graph_max_size))
        adj_matrix = np.zeros(shape=(batch_size, self.graph_max_size, self.graph_max_size), dtype=np.int8)

        image_graph_start_index = np.zeros(shape=(batch_size), dtype=np.int)

        if self.use_text_graph:
            node_bpe_index = kwargs['node_bpe_index']  # (batch_size, 3, 224, 224)
            text_edges = kwargs['edges']

            # adjacency matrix
            for batch_index in range(batch_size):
                # complete graph
                node_count = len(node_bpe_index[batch_index])
                adj_matrix[batch_index, :node_count, :node_count] = 1

                # # normal graph
                # for u, v in text_edges[batch_index]:
                #     if u >= self.text_graph_max_size or v >= self.text_graph_max_size:
                #         continue
                #     adj_matrix[batch_index, u, u] = 1       # self-loop
                #     adj_matrix[batch_index, v, v] = 1       # self-loop
                #     adj_matrix[batch_index, u, v] = 1
                #     adj_matrix[batch_index, v, u] = 1       # reverse edges

            # node embedding
            for batch_index in range(batch_size):
                enc_padding_mask[batch_index, :min(self.text_graph_max_size, len(node_bpe_index[batch_index]))] = 1
                for node_index, bpe_index in enumerate(node_bpe_index[batch_index]):
                    if node_index >= self.text_graph_max_size:
                        continue
                    s, e = bpe_index
                    if e >= seq_len:
                        adj_matrix[batch_index, node_index, :] = 0
                        adj_matrix[batch_index, :, node_index] = 0
                        continue
                    _emb = encoder_out[s : e, batch_index].mean(dim=0)
                    node_emb[batch_index, node_index] = _emb
                image_graph_start_index[batch_index] = min(len(node_bpe_index[batch_index]), self.text_graph_max_size)

        if self.use_image_graph:
            obj_feat_batch, obj_feat_index = kwargs['obj_feat'], kwargs['obj_feat_index']
            face_feat_batch, face_feat_index = kwargs['face_feat'], kwargs['face_feat_index']

            obj_feat_batch = self.obj_feat_emb(obj_feat_batch)
            face_feat_batch = self.face_feat_emb(face_feat_batch)

            for batch_index in range(batch_size):
                obj_feat = obj_feat_batch[obj_feat_index[batch_index][0] : obj_feat_index[batch_index][1]]
                face_feat = face_feat_batch[face_feat_index[batch_index][0] : face_feat_index[batch_index][1]]
                img_feat = torch.cat([face_feat, obj_feat], dim=0)      # (?, 1024)

                _s = image_graph_start_index[batch_index]
                _e = min(self.graph_max_size, image_graph_start_index[batch_index] + img_feat.shape[0])
                _n_feats = _e - _s
                try:
                    node_emb[batch_index, _s : _e] = img_feat[: _n_feats]
                except:
                    import IPython; IPython.embed()

                _s, _e = self.image_graph_start, self.image_graph_start + _n_feats
                adj_matrix[batch_index, _s : _e, _s : _e] = 1

        adj_matrix = torch.tensor(adj_matrix, device=encoder_out.device, dtype=torch.bool)

        _node_emb = self.gat(node_emb, adj_matrix)

        encoder_out = _node_emb.transpose(0, 1)         # (src_len, batch_size, 512)

        # enc_padding_mask = create_padding_mask(src_tokens, src_lengths)  # (batch_size, src_len)
        enc_padding_mask = torch.tensor(enc_padding_mask, device=encoder_out.device, dtype=torch.bool)

        # torch.cuda.synchronize();
        t1 = time.time()
        t_forward = t1 - t0
        if self.training:
            metrics.log_scalar('graph_encoder_forward', t_forward)

        return self._forward_return_encoder_out(encoder_out, enc_padding_mask)


class EncoderMM(FairseqEncoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.output_mode = 'conv'

        encoders = {}
        if args.use_image:
            if args.image_encoder == 'resnet152':
                encoders['image'] = ResNetWrapper(args, dictionary, output_mode=self.output_mode)
            elif args.image_encoder == 'vit16':
                encoders['image'] = ViTWrapper(args, dictionary, output_mode=self.output_mode)
        if args.use_text:
            encoders['text'] = RobertaWrapper(args, dictionary, output_mode=self.output_mode)
        if args.use_image_graph or args.use_text_graph:
            encoders['graph'] = GraphEncoderWrapper(args, dictionary, output_mode=self.output_mode)

        self.encoders = nn.ModuleDict(encoders)

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        encoder_out = {}
        for key, encoder in self.encoders.items():
            if key == 'graph':
                roberta_output = encoder_out['text']
                kwargs['roberta_output'] = roberta_output
                encoder_out[key] = encoder.forward(src_tokens, src_lengths, **kwargs)
                del kwargs['roberta_output']
            else:
                encoder_out[key] = encoder.forward(src_tokens, src_lengths, **kwargs)

        return encoder_out


    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out_new = {}
        for key, encoder in self.encoders.items():
            encoder_out_new[key] = encoder.reorder_encoder_out(encoder_out[key], new_order)
        return encoder_out_new


@register_model('transformer2')
class TransformerModelBase(FairseqEncoderDecoderModel):
    @staticmethod
    def add_args(parser, **kwargs):
        parser.add_argument('--activation-fn', choices=fairseq.utils.get_available_activation_fns(), help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D', help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D', help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D', help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR', help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N', help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N', help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N', help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N', help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true', help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true', help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR', help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N', help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N', help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N', help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N', help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true', help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true', help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N', help='decoder output dimension (extra linear layer if different from decoder embed dim')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true', help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true', help='share encoder, decoder and output embeddings (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true', help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR', help='comma separated list of adaptive softmax cutoff points. Must be used with adaptive_loss criterion')
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D', help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding', action='store_true', help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true', help='if True, dont scale embeddings')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true', help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true', help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0, help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0, help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None, help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None, help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0, help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8, help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0, help='scalar quantization noise and scalar quantization at training time')

        # added
        # parser.add_argument('--adaptive-softmax-factor', type=float, metavar='D', default=4., help='adaptive softmax factor')
        parser.add_argument('--adaptive-softmax-factor', type=float, metavar='D', default=1., help='adaptive softmax factor')
        parser.add_argument('--tie-adaptive-proj', default=False, action='store_true', help='tie_proj')

    @classmethod
    def build_model(cls, args, task, **kwargs):
        dictionary = task.dictionary

        encoder_embed_tokens = cls.build_embedding(
            args, dictionary, args.encoder_embed_dim, args.encoder_embed_path
        )
        decoder_embed_tokens = encoder_embed_tokens

        encoder = cls.build_encoder(args, src_dict=dictionary, embed_tokens=encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict=dictionary, embed_tokens=decoder_embed_tokens)

        model = cls(encoder, decoder)
        model.args = args
        model.dictionary = dictionary
        return model

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = fairseq.models.transformer.Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = fairseq.utils.parse_embedding(path)
            fairseq.utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return fairseq.models.transformer.TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return fairseq.models.transformer.TransformerDecoder(args, tgt_dict, embed_tokens, no_encoder_attn=getattr(args, "no_cross_attention", False))

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


# TODO: useless?
@register_model('transformer2_roberta')
class TransformerModelMultimodal(TransformerModelBase):
    @staticmethod
    def add_args(parser, **kwargs):
        super(TransformerModelMultimodal, TransformerModelMultimodal).add_args(parser, **kwargs)

    @classmethod
    def build_model(cls, args, task, **kwargs):
        dictionary = task.dictionary

        encoder = cls.build_encoder(args, src_dict=dictionary)

        # TODO: ?
        # roberta = encoder.roberta
        # embed_tokens = roberta.model.encoder.sentence_encoder.embed_tokens
        embed_tokens = cls.build_embedding(args, dictionary, args.encoder_embed_dim)

        decoder = cls.build_decoder(args, tgt_dict=dictionary, embed_tokens=embed_tokens)

        model = cls(encoder, decoder)
        model.args = args
        model.dictionary = dictionary
        return model

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        print('')

        if args.adaptive_input:
            emb = AdaptiveInput(
                vocab_size=len(dictionary),
                padding_idx=padding_idx,
                initial_dim=args.encoder_embed_dim,
                factor=args.adaptive_softmax_factor,
                output_dim=args.encoder_embed_dim,
                cutoff=options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                q_noise=args.quant_noise_pq,
                qn_block_size=args.quant_noise_pq_block_size,
            )
        else:
            emb = fairseq.models.transformer.Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = fairseq.utils.parse_embedding(path)
                fairseq.utils.load_embedding(embed_dict, dictionary, emb)

        return emb

    @classmethod
    def build_encoder(cls, args, src_dict):
        return RobertaWrapper(args, src_dict)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return fairseq.models.transformer.TransformerDecoder(args, tgt_dict, embed_tokens, no_encoder_attn=getattr(args, "no_cross_attention", False))


# roberta + lightconv
@register_model('transformer2_conv')
class TransformerModelConv(TransformerModelBase):
    @staticmethod
    def add_args(parser, **kwargs):
        super(TransformerModelConv, TransformerModelConv).add_args(parser, **kwargs)

        """LightConv and DynamicConv arguments"""
        parser.add_argument("--encoder-kernel-size-list", type=lambda x: fairseq.utils.eval_str_list(x, int),
                            help='list of kernel size (default: "[3,7,15,31,31,31,31]")', )
        parser.add_argument("--decoder-kernel-size-list", type=lambda x: fairseq.utils.eval_str_list(x, int),
                            help='list of kernel size (default: "[3,7,15,31,31,31]")', )
        parser.add_argument("--encoder-glu", type=fairseq.utils.eval_bool, help="glu after in proj")
        parser.add_argument("--decoder-glu", type=fairseq.utils.eval_bool, help="glu after in proj")
        parser.add_argument("--encoder-conv-type", default="dynamic", type=str, choices=["dynamic", "lightweight"], help="type of convolution")
        parser.add_argument("--decoder-conv-type", default="dynamic", type=str, choices=["dynamic", "lightweight"], help="type of convolution")
        parser.add_argument("--weight-softmax", default=True, type=fairseq.utils.eval_bool)
        parser.add_argument("--weight-dropout", type=float, metavar="D", help="dropout probability for conv weights")

    @classmethod
    def build_model(cls, args, task, **kwargs):
        dictionary = task.dictionary

        encoder = cls.build_encoder(args, src_dict=dictionary)
        # TODO: whether share embedding ?
        # roberta = encoder.roberta
        # embed_tokens = roberta.model.encoder.sentence_encoder.embed_tokens
        embed_tokens = cls.build_embedding(args, dictionary, args.encoder_embed_dim)

        decoder = cls.build_decoder(args, tgt_dict=dictionary, embed_tokens=embed_tokens)

        model = cls(encoder, decoder)
        model.args = args
        model.dictionary = dictionary
        return model

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        print('')

        if args.adaptive_input:
            emb = AdaptiveInput(
                vocab_size=len(dictionary),
                padding_idx=padding_idx,
                initial_dim=args.encoder_embed_dim,
                factor=args.adaptive_softmax_factor,
                output_dim=args.encoder_embed_dim,
                cutoff=options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                q_noise=args.quant_noise_pq,
                qn_block_size=args.quant_noise_pq_block_size,
            )
        else:
            emb = fairseq.models.transformer.Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = fairseq.utils.parse_embedding(path)
                fairseq.utils.load_embedding(embed_dict, dictionary, emb)

        return emb

    @classmethod
    def build_encoder(cls, args, src_dict):
        return RobertaWrapper(args, src_dict, output_mode='conv')

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return fairseq.models.lightconv.LightConvDecoder(
            args, tgt_dict, embed_tokens, no_encoder_attn=getattr(args, "no_cross_attention", False)
        )

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


# multiple encoder (roberta+resnet+...) + lightconv
@register_model('transformer2_conv_mm')
class TransformerModelConvMM(TransformerModelConv):
    @staticmethod
    def add_args(parser, **kwargs):
        super(TransformerModelConvMM, TransformerModelConvMM).add_args(parser, **kwargs)
        # moved to task
        # parser.add_argument('--use-image', type=int, default=1)
        # parser.add_argument('--use-text', type=int, default=1)
        # parser.add_argument('--use-text-graph', type=int, default=1)
        # parser.add_argument('--text-graph-max-size', type=int, default=150)
        # parser.add_argument('--use-image-graph', type=int, default=1)
        # parser.add_argument('--image-graph-max-size', type=int, default=20)

        parser.add_argument('--image-encoder', type=str, default='resnet152', choices=['resnet152', 'vit16'])
        parser.add_argument('--roberta-fuse', type=str, default='mean', choices=['last', 'weighted', 'mean'])
        parser.add_argument('--gat-layers', type=int, default=2)
        parser.add_argument('--gat-heads', type=int, default=4)

    @classmethod
    def build_model(cls, args, task, **kwargs):
        dictionary = task.dictionary

        encoder = cls.build_encoder(args, src_dict=dictionary)
        # TODO: whether share embedding ?
        # roberta = encoder.roberta
        # embed_tokens = roberta.model.encoder.sentence_encoder.embed_tokens
        embed_tokens = cls.build_embedding(args, dictionary, args.encoder_embed_dim)

        decoder = cls.build_decoder(args, tgt_dict=dictionary, embed_tokens=embed_tokens)

        model = cls(encoder, decoder)
        model.args = args
        model.dictionary = dictionary
        return model

    @classmethod
    def build_encoder(cls, args, src_dict):
        # return RobertaWrapper(args, src_dict, output_mode='conv')
        return EncoderMM(args, src_dict)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        # return fairseq.models.lightconv.LightConvDecoder(
        #     args, tgt_dict, embed_tokens, no_encoder_attn=getattr(args, "no_cross_attention", False)
        # )
        context_dimensions = {'text': 1024}
        if args.use_image:
            if args.image_encoder == 'resnet152':
                context_dimensions['image'] = 2048
            elif args.image_encoder == 'vit16':
                context_dimensions['image'] = 768
        if args.use_image_graph or args.use_text_graph:
            context_dimensions['graph'] = 1024
        return LightConvDecoderMM(args, tgt_dict, embed_tokens, context_dimensions=context_dimensions)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    def get_normalized_probs(self, net_output, log_probs, sample=None,):
        return self.decoder.get_normalized_probs(net_output, log_probs, sample)

    # TODO: better solution?
    def load_state_dict(self, state_dict, strict=True, args=None):
        return super().load_state_dict(state_dict, False, args)


"""
@register_model_architecture:
The decorated function should take a single argument cfg, which is a omegaconf.DictConfig. 
The decorated function should modify these arguments in-place to match the desired architecture.
called by parse_args_and_arch()
called after model.add_args
"""

@register_model_architecture('transformer2', 'transformer2')
def transformer2(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", False)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(args, "no_token_positional_embeddings", False)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(args, "decoder_output_dim", args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)


# @register_model_architecture('transformer2_mm', 'transformer2_mm')
@register_model_architecture('transformer2_roberta', 'transformer2_roberta')
def transformer2_roberta(args):
    transformer2(args)


@register_model_architecture('transformer2_conv', 'transformer2_conv')
def transformer2_conv(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", True)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(args, "no_token_positional_embeddings", False)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(args, "decoder_output_dim", args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)

    # transformer2(args)

    args.use_image = getattr(args, 'use_image', 1)
    args.use_test = getattr(args, 'use_text', 1)

    args.relu_dropout = getattr(args, "relu_dropout", 0.0)      # used by lightconv

    args.encoder_conv_dim = getattr(args, "encoder_conv_dim", args.encoder_embed_dim)
    args.decoder_conv_dim = getattr(args, "decoder_conv_dim", args.decoder_embed_dim)

    args.encoder_kernel_size_list = getattr(args, "encoder_kernel_size_list", [3, 7, 15, 31])
    args.decoder_kernel_size_list = getattr(args, "decoder_kernel_size_list", [3, 7, 15, 31])
    if len(args.encoder_kernel_size_list) == 1:
        args.encoder_kernel_size_list = (args.encoder_kernel_size_list * args.encoder_layers)
    if len(args.decoder_kernel_size_list) == 1:
        args.decoder_kernel_size_list = (args.decoder_kernel_size_list * args.decoder_layers)
    assert (len(args.encoder_kernel_size_list) == args.encoder_layers), "encoder_kernel_size_list doesn't match encoder_layers"
    assert (len(args.decoder_kernel_size_list) == args.decoder_layers), "decoder_kernel_size_list doesn't match decoder_layers"
    args.encoder_glu = getattr(args, "encoder_glu", True)
    args.decoder_glu = getattr(args, "decoder_glu", True)

    args.final_norm = getattr(args, "final_norm", False)

    args.input_dropout = getattr(args, "input_dropout", 0.1)
    args.weight_dropout = getattr(args, "weight_dropout", 0.1)


@register_model_architecture('transformer2_conv_mm', 'transformer2_conv_mm')
def transformer2_conv_mm(args):
    transformer2_conv(args)
    args.image_encoder = getattr(args, 'image_encoder', 'resnet152')


def main():
    pass


if __name__ == '__main__':
    main()