# from transform-and-tell

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import FairseqIncrementalDecoder
from fairseq.modules import LayerNorm, LightweightConv1dTBC, DynamicConv1dTBC, FairseqDropout, PositionalEmbedding, AdaptiveSoftmax
from torch.nn import Linear
from fairseq import utils

from model.multi_head import MultiHeadAttention


class GehringLinear(nn.Linear):
    """A linear layer with Gehring initialization and weight normalization."""

    def __init__(self, in_features, out_features, dropout=0, bias=True,
                 weight_norm=True):
        self.dropout = dropout
        self.weight_norm = weight_norm
        super().__init__(in_features, out_features, bias)

    def reset_parameters(self):
        # One problem with initialization from the uniform distribution is that
        # the distribution of the outputs has a variance that grows with the
        # number of inputs. It turns out that we can normalize the variance of
        # each neuron’s output to 1 by scaling its weight vector by the square
        # root of its fan-in (i.e. its number of inputs). Dropout further
        # increases the variance of each input, so we need to scale down std.
        # See A.3. in Gehring et al (2017): https://arxiv.org/pdf/1705.03122.
        std = math.sqrt((1 - self.dropout) / self.in_features)
        self.weight.data.normal_(mean=0, std=std)
        if self.bias is not None:
            self.bias.data.fill_(0)

        # Weight normalization is a reparameterization that decouples the
        # magnitude of a weight tensor from its direction. See Salimans and
        # Kingma (2016): https://arxiv.org/abs/1602.07868.
        if self.weight_norm:
            nn.utils.weight_norm(self)


class LightConvDecoderLayerMM(nn.Module):
    def __init__(self, args, kernel_size=0, context_dimensions=None):
        super().__init__()

        decoder_embed_dim = args.decoder_embed_dim
        decoder_conv_dim = args.decoder_conv_dim
        decoder_glu = args.decoder_glu
        decoder_conv_type = args.decoder_conv_type
        weight_softmax = args.weight_softmax
        decoder_attention_heads = args.decoder_attention_heads
        weight_dropout = args.weight_dropout
        dropout = args.dropout
        relu_dropout = args.relu_dropout
        input_dropout = args.input_dropout
        decoder_normalize_before = args.decoder_normalize_before
        attention_dropout = args.attention_dropout
        decoder_ffn_embed_dim = args.decoder_ffn_embed_dim
        # TODO:
        article_embed_size = 1024
        C = 2048

        if context_dimensions is None:
            context_dimensions = {'text': 1024}
        self.context_keys = list(context_dimensions.keys())

        self.embed_dim = decoder_embed_dim
        self.conv_dim = decoder_conv_dim
        if decoder_glu:
            self.linear1 = GehringLinear(self.embed_dim, 2*self.conv_dim)
            self.act = nn.GLU()
        else:
            self.linear1 = GehringLinear(self.embed_dim, self.conv_dim)
            self.act = None
        if decoder_conv_type == 'lightweight':
            self.conv = LightweightConv1dTBC(self.conv_dim, kernel_size, padding_l=kernel_size-1,
                                             weight_softmax=weight_softmax,
                                             num_heads=decoder_attention_heads,
                                             weight_dropout=weight_dropout)
        elif decoder_conv_type == 'dynamic':
            self.conv = DynamicConv1dTBC(self.conv_dim, kernel_size, padding_l=kernel_size-1,
                                         weight_softmax=weight_softmax,
                                         num_heads=decoder_attention_heads,
                                         weight_dropout=weight_dropout)
        else:
            raise NotImplementedError
        self.linear2 = GehringLinear(self.conv_dim, self.embed_dim)

        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.input_dropout = input_dropout
        self.normalize_before = decoder_normalize_before

        self.conv_layer_norm = nn.LayerNorm(self.embed_dim)

        self.context_attns = nn.ModuleDict()
        self.context_attn_lns = nn.ModuleDict()

        for key in self.context_keys:           # iterate in fixed order
            self.context_attns[key] = MultiHeadAttention(
                self.embed_dim, decoder_attention_heads, kdim=context_dimensions[key], vdim=context_dimensions[key],
                dropout=attention_dropout)
            self.context_attn_lns[key] = nn.LayerNorm(self.embed_dim)

        context_size = self.embed_dim * len(self.context_keys)

        self.context_fc = GehringLinear(context_size, self.embed_dim)

        self.fc1 = GehringLinear(self.embed_dim, decoder_ffn_embed_dim)
        self.fc2 = GehringLinear(decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.need_attn = True

    def forward(self, X, contexts, incremental_state):
        """
        Args:
            X (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            contexts: {'key': encoder_output}
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = X
        X = self.maybe_layer_norm(self.conv_layer_norm, X, before=True)
        X = F.dropout(X, p=self.input_dropout, training=self.training)
        X = self.linear1(X)
        if self.act is not None:
            X = self.act(X)
        X = self.conv(X, incremental_state=incremental_state)
        X = self.linear2(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = residual + X
        X = self.maybe_layer_norm(self.conv_layer_norm, X, after=True)

        attn = None
        X_contexts = []

        # Article attention
        for key in self.context_keys:               # iterate in fixed order
            encoder_out = contexts[key]['encoder_out']
            mask = contexts[key]['encoder_padding_mask']

            residual = X
            X_key = self.maybe_layer_norm(
                self.context_attn_lns[key], X, before=True)
            X_key, attn = self.context_attns[key](
                query=X_key,                    # (50, 8, 512)              (32, 8, 1024)
                key=encoder_out,                # (49, 8, 512)              (512, 8, 1024)
                value=encoder_out,              # (49, 8, 512)              (512, 8, 1024)
                key_padding_mask=mask,
                incremental_state=None,
                static_kv=True,
                need_weights=(not self.training and self.need_attn))
            X_key = F.dropout(X_key, p=self.dropout,
                                  training=self.training)
            X_key = residual + X_key
            X_key = self.maybe_layer_norm(
                self.context_attn_lns[key], X_key, after=True)

            X_contexts.append(X_key)        # (40, 8, 512)

        X_context = torch.cat(X_contexts, dim=-1)
        X = self.context_fc(X_context)

        residual = X
        X = self.maybe_layer_norm(self.final_layer_norm, X, before=True)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, p=self.relu_dropout, training=self.training)
        X = self.fc2(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = residual + X
        X = self.maybe_layer_norm(self.final_layer_norm, X, after=True)
        return X, attn

    def maybe_layer_norm(self, layer_norm, X, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(X)
        else:
            return X

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def extra_repr(self):
        return 'dropout={}, relu_dropout={}, input_dropout={}, normalize_before={}'.format(
            self.dropout, self.relu_dropout, self.input_dropout, self.normalize_before)


class LightConvDecoderMM(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary, embed_tokens, context_dimensions=None):
        super().__init__(dictionary)
        self.context_keys = context_dimensions.keys()
        self.context_dimensions = context_dimensions

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        output_embed_dim = args.decoder_output_dim

        final_norm = args.final_norm

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # TODO: try with input_embed_dim

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                LightConvDecoderLayerMM(
                    args=args, kernel_size=args.decoder_kernel_size_list[i], context_dimensions=context_dimensions
                )
                for i in range(args.decoder_layers)
            ]
        )

        self.adaptive_softmax = None

        self.project_out_dim = (
            Linear(embed_dim, output_embed_dim, bias=False)
            if embed_dim != output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(len(dictionary), output_embed_dim)
            )
            nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim ** -0.5)
        self.register_buffer("version", torch.Tensor([2]))

        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        contexts = encoder_out      # input to layer

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens,
                incremental_state=incremental_state,
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                contexts,
                incremental_state,
            )
            inner_states.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)

        return x, {"attn": attn, "inner_states": inner_states}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1
            )
        return self._future_mask[:dim, :dim]
