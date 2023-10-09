import os
import sys

import torch
from fairseq.models import register_model_architecture, register_model, FairseqEncoder
from torch import nn

from model import transformer2_conv, TransformerModelConv, LightConvDecoderMM, ResNetWrapper, RobertaWrapper, \
    TransformerFeatureExtractor


class ObjectEncoder(TransformerFeatureExtractor):
    def __init__(self, args, dictionary, output_mode='transformer', input_dim=2048, type='obj'):
        super().__init__(dictionary)
        self.output_mode = output_mode

        self.encoder_embed_dim = args.encoder_embed_dim
        self.encoder_emb = nn.Linear(input_dim, self.encoder_embed_dim)
        self.type = type
        self.input_dim = input_dim

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        obj_feat_batch, obj_feat_index = kwargs[f'{self.type}_feat'], kwargs[f'{self.type}_feat_index']     # (?, 2048), (batch_size, 2)
        batch_size = obj_feat_index.shape[0]
        max_len = max(1, (obj_feat_index[:, 1] - obj_feat_index[:, 0]).max())
        obj_feat = obj_feat_batch.new_zeros(batch_size, max_len, self.input_dim)
        obj_mask = obj_feat_batch.new_zeros(batch_size, max_len, dtype=torch.bool)

        for batch_index in range(batch_size):
            n_obj = obj_feat_index[batch_index, 1] - obj_feat_index[batch_index, 0]
            obj_feat[batch_index, :n_obj] = obj_feat_batch[obj_feat_index[batch_index, 0] : obj_feat_index[batch_index, 1]]
            obj_mask[batch_index, :n_obj] = 1

        encoder_out = obj_feat.transpose(0, 1)
        enc_padding_mask = obj_mask
        return self._forward_return_encoder_out(encoder_out, enc_padding_mask)


class EncoderMM(FairseqEncoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.output_mode = 'conv'

        encoders = {}
        if args.use_image:
            encoders['image'] = ResNetWrapper(args, dictionary, output_mode=self.output_mode)
        if args.use_text:
            encoders['text'] = RobertaWrapper(args, dictionary, output_mode=self.output_mode)
        if args.use_image_graph:
            encoders['object'] = ObjectEncoder(args, dictionary, output_mode=self.output_mode, input_dim=2048, type='obj')
            encoders['face'] = ObjectEncoder(args, dictionary, output_mode=self.output_mode, input_dim=512, type='face')

        self.encoders = nn.ModuleDict(encoders)

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        encoder_out = {}
        for key, encoder in self.encoders.items():
            encoder_out[key] = encoder.forward(src_tokens, src_lengths, **kwargs)

        return encoder_out

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out_new = {}
        for key, encoder in self.encoders.items():
            encoder_out_new[key] = encoder.reorder_encoder_out(encoder_out[key], new_order)
        return encoder_out_new



# multiple encoder (roberta+resnet+...) + lightconv
# transform & tell encoder
@register_model('transformer2_conv_mm_tt')
class TransformerModelConvMM(TransformerModelConv):
    @staticmethod
    def add_args(parser, **kwargs):
        super(TransformerModelConvMM, TransformerModelConvMM).add_args(parser, **kwargs)

        parser.add_argument('--roberta-fuse', type=str, default='mean', choices=['last', 'weighted', 'mean'])

        # moved to task
        # parser.add_argument('--use-image', type=int, default=1)
        # parser.add_argument('--use-text', type=int, default=1)
        # parser.add_argument('--use-text-graph', type=int, default=0)
        # parser.add_argument('--text-graph-max-size', type=int, default=150)
        # parser.add_argument('--use-image-graph', type=int, default=1)
        # parser.add_argument('--image-graph-max-size', type=int, default=20)

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
        context_dimensions = {'text': 1024}
        if args.use_image:
            context_dimensions['image'] = 2048
        if args.use_image_graph:
            context_dimensions['object'] = 2048
            context_dimensions['face'] = 512
        return LightConvDecoderMM(args, tgt_dict, embed_tokens, context_dimensions=context_dimensions)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    def get_normalized_probs(self, net_output, log_probs, sample=None,):
        return self.decoder.get_normalized_probs(net_output, log_probs, sample)

    def load_state_dict(self, state_dict, strict=True, args=None):
        return super().load_state_dict(state_dict, False, args)



@register_model_architecture('transformer2_conv_mm_tt', 'transformer2_conv_mm_tt')
def transformer2_conv_mm_tt(args):
    transformer2_conv(args)