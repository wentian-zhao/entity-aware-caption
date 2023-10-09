import copy
import math

import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import fairseq.modules
from fairseq.modules import MultiheadAttention

from model.fairseq_transformer import clones, LayerNorm


# class BaseGATLayer(torch.nn.Module):
#     """
#     Base class for all implementations as there is much code that would otherwise be copy/pasted.
#     """
#
#     head_dim = 1
#
#     def __init__(self, num_in_features, num_out_features, num_of_heads, layer_type, concat=True, activation=nn.ELU(),
#                  dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):
#
#         super().__init__()
#
#         # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
#         self.num_of_heads = num_of_heads
#         self.num_out_features = num_out_features
#         self.concat = concat  # whether we should concatenate or average the attention heads
#         self.add_skip_connection = add_skip_connection
#
#         #
#         # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
#         # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
#         #
#
#         self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
#
#         # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
#         # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.
#
#         # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
#         # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
#         self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
#         self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
#
#         # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
#         if bias and concat:
#             self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
#         elif bias and not concat:
#             self.bias = nn.Parameter(torch.Tensor(num_out_features))
#         else:
#             self.register_parameter('bias', None)
#
#         if add_skip_connection:
#             self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
#         else:
#             self.register_parameter('skip_proj', None)
#
#         #
#         # End of trainable weights
#         #
#
#         self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
#         self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
#         self.activation = activation
#         # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
#         # and for attention coefficients. Functionality-wise it's the same as using independent modules.
#         self.dropout = nn.Dropout(p=dropout_prob)
#
#         self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
#         self.attention_weights = None  # for later visualization purposes, I cache the weights here
#
#         self.init_params(layer_type)
#
#     def init_params(self, layer_type):
#         """
#         The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
#             https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow
#         The original repo was developed in TensorFlow (TF) and they used the default initialization.
#         Feel free to experiment - there may be better initializations depending on your problem.
#         """
#         nn.init.xavier_uniform_(self.linear_proj.weight)
#         nn.init.xavier_uniform_(self.scoring_fn_target)
#         nn.init.xavier_uniform_(self.scoring_fn_source)
#
#         if self.bias is not None:
#             torch.nn.init.zeros_(self.bias)
#
#     def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
#         if self.log_attention_weights:  # potentially log for later visualization in playground.py
#             self.attention_weights = attention_coefficients
#
#         # if the tensor is not contiguously stored in memory we'll get an error after we try to do certain ops like view
#         # only imp1 will enter this one
#         if not out_nodes_features.is_contiguous():
#             out_nodes_features = out_nodes_features.contiguous()
#
#         if self.add_skip_connection:  # add skip or residual connection
#             if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
#                 # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
#                 # thus we're basically copying input vectors NH times and adding to processed vectors
#                 out_nodes_features += in_nodes_features.unsqueeze(1)
#             else:
#                 # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
#                 # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
#                 out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
#
#         if self.concat:
#             # shape = (N, NH, FOUT) -> (N, NH*FOUT)
#             out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
#         else:
#             # shape = (N, NH, FOUT) -> (N, FOUT)
#             out_nodes_features = out_nodes_features.mean(dim=self.head_dim)
#
#         if self.bias is not None:
#             out_nodes_features += self.bias
#
#         return out_nodes_features if self.activation is None else self.activation(out_nodes_features)
#
#
# class GATLayerImp3(BaseGATLayer):
#     """
#     Implementation #3 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric
#     But, it's hopefully much more readable! (and of similar performance)
#     It's suitable for both transductive and inductive settings. In the inductive setting we just merge the graphs
#     into a single graph with multiple components and this layer is agnostic to that fact! <3
#     """
#
#     src_nodes_dim = 0  # position of source nodes in edge index
#     trg_nodes_dim = 1  # position of target nodes in edge index
#
#     nodes_dim = 0      # node dimension/axis
#     head_dim = 1       # attention head dimension/axis
#
#     def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
#                  dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):
#
#         # Delegate initialization to the base class
#         super().__init__(num_in_features, num_out_features, num_of_heads, None, concat, activation, dropout_prob,
#                       add_skip_connection, bias, log_attention_weights)
#
#     def forward(self, data):
#         #
#         # Step 1: Linear Projection + regularization
#         #
#
#         in_nodes_features, edge_index = data  # unpack data
#         num_of_nodes = in_nodes_features.shape[self.nodes_dim]
#         assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'
#
#         # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
#         # We apply the dropout to all of the input node features (as mentioned in the paper)
#         # Note: for Cora features are already super sparse so it's questionable how much this actually helps
#         in_nodes_features = self.dropout(in_nodes_features)
#
#         # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
#         # We project the input node features into NH independent output features (one for each attention head)
#         nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
#
#         nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well
#
#         # Step 2: Edge attention calculation
#
#         # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
#         # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
#         # Optimization note: torch.sum() is as performant as .sum() in my experiments
#         scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
#         scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)
#
#         # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
#         # the possible combinations of scores we just prepare those that will actually be used and those are defined
#         # by the edge index.
#         # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
#         scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
#         scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)
#
#         # shape = (E, NH, 1)
#         attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
#         # Add stochasticity to neighborhood aggregation
#         attentions_per_edge = self.dropout(attentions_per_edge)
#
#         # Step 3: Neighborhood aggregation
#
#         # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
#         # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
#         nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
#
#         # This part sums up weighted and projected neighborhood feature vectors for every target node
#         # shape = (N, NH, FOUT)
#         out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)
#
#         # Step 4: Residual/skip connections, concat and bias
#
#         out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
#         return (out_nodes_features, edge_index)
#
#     #
#     # Helper functions (without comments there is very little code so don't be scared!)
#     #
#
#     def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
#         """
#         As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
#         Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
#         into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
#         in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
#         (where 1-3 is overloaded notation it represents the edge 1-3 and it's (exp) score) and similarly for 2-3 and 3-3
#          i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.
#         Note:
#         Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
#         and it's a fairly common "trick" used in pretty much every deep learning framework.
#         Check out this link for more details:
#         https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning
#         """
#         # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
#         scores_per_edge = scores_per_edge - scores_per_edge.max()
#         exp_scores_per_edge = scores_per_edge.exp()  # softmax
#
#         # Calculate the denominator. shape = (E, NH)
#         neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)
#
#         # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
#         # possibility of the computer rounding a very small number all the way to 0.
#         attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)
#
#         # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
#         return attentions_per_edge.unsqueeze(-1)
#
#     def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
#         # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
#         trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)
#
#         # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
#         size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
#         size[self.nodes_dim] = num_of_nodes
#         neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)
#
#         # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
#         # target index)
#         neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)
#
#         # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
#         # all the locations where the source nodes pointed to i (as dictated by the target index)
#         # shape = (N, NH) -> (E, NH)
#         return neighborhood_sums.index_select(self.nodes_dim, trg_index)
#
#     def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
#         size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
#         size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
#         out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)
#
#         # shape = (E) -> (E, NH, FOUT)
#         trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
#         # aggregation step - we accumulate projected, weighted node features for all the attention heads
#         # shape = (E, NH, FOUT) -> (N, NH, FOUT)
#         out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)
#
#         return out_nodes_features
#
#     def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
#         """
#         Lifts i.e. duplicates certain vectors depending on the edge index.
#         One of the tensor dims goes from N -> E (that's where the "lift" comes from).
#         """
#         src_nodes_index = edge_index[self.src_nodes_dim]
#         trg_nodes_index = edge_index[self.trg_nodes_dim]
#
#         # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
#         scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
#         scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
#         nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)
#
#         return scores_source, scores_target, nodes_features_matrix_proj_lifted
#
#     def explicit_broadcast(self, this, other):
#         # Append singleton dimensions until this.dim() == other.dim()
#         for _ in range(this.dim(), other.dim()):
#             this = this.unsqueeze(-1)
#
#         # Explicitly expand so that shapes are the same
#         return this.expand_as(other)


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
    def __init__(self, h, d_model, dropout=0.1, attn_type='dot', alpha=0.2):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.attn_type = attn_type
        assert attn_type in ('dot', 'cat')
        # We assume d_v always equals d_k
        self.d_model = d_model
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        if attn_type == 'cat':
            self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)
            self.a = nn.Parameter(data=torch.zeros(self.h, self.d_k * 2))
            self.register_parameter('a', self.a)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.a)

    def forward(self, query, key, value, mask=None):
        """
        :param query: (batch_size, n_keys, d_model)
        :param key: (batch_size, n_keys, d_model)
        :param value: (batch_size, n_keys, d_model)
        :param mask: (batch_size, n_keys, n_keys)   mask == 0 -> mask out; mask == 1 -> keep
        :return:    (batch_size, n_keys, d_model)
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        n_keys = query.size(1)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]        # query: (batch_size, n_heads, n_keys, d_k)

        # 2) Apply attention on all the projected vectors in batch.
        # x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        dropout = self.dropout
        d_k = query.size(-1)
        if self.attn_type == 'dot':
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)        # (batch_size, n_head, n_keys, n_keys)
        elif self.attn_type == 'cat':
            q_i = query.unsqueeze(3).expand(nbatches, self.h, n_keys, n_keys, self.d_k)
            k_i = key.unsqueeze(2).expand(nbatches, self.h, n_keys, n_keys, self.d_k)
            s = torch.cat([q_i, k_i], dim=-1)       # (batch_size, n_head, n_keys1, n_keys2, d_k * 2)

            s = s.unsqueeze(4)
            _a = self.a.view(1, self.h, 1, 1, self.d_k * 2, 1).expand(nbatches, self.h, n_keys, n_keys, self.d_k * 2, 1)
            scores = torch.matmul(s, _a).squeeze(5).squeeze(4)
        # mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # activation
        if self.attn_type == 'cat':
            scores = self.leaky_relu(scores)

        p_attn = F.softmax(scores, dim=-1)
        p_attn = torch.where(torch.isnan(p_attn), torch.zeros_like(p_attn), p_attn)       # replace nan with 0
        if dropout is not None:
            p_attn = dropout(p_attn)
        x, self.attn = torch.matmul(p_attn, value), p_attn

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class GraphEncoder(nn.Module):
    def __init__(self, d_model, n_layer, n_head, attn_type='cat'):
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head

        attn = MultiHeadedAttention(h=n_head, d_model=d_model, attn_type=attn_type)
        # attn = fairseq.modules.MultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=0.1)
        self.gat_layers = clones(attn, n_layer)
        self.norm = LayerNorm(d_model)

        self.init_parameters()

    def init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, node_emb, adj):
        """
        :param node_emb: (batch_size, n_nodes, node_emb_dim)
        :param adj: (batch_size, n_nodes, n_nodes)
        :return:
        """
        batch_size, n_nodes, node_emb_dim = node_emb.shape
        x = node_emb

        # # padding elements are indicated by 1s.
        # key_padding_mask = (adj + adj.transpose(1, 2)).sum(dim=2).to(torch.bool).logical_not()
        # # the non-zero positions are not allowed to attend while the zero positions will be unchanged.
        # attn_mask = adj.to(torch.bool).logical_not().unsqueeze(1).expand(batch_size, self.n_head, n_nodes, n_nodes).reshape(batch_size * self.n_head, n_nodes, n_nodes)

        for i in range(self.n_layer):
            x = self.gat_layers[i](x, x, x, mask=adj)
            # x = x.transpose(0, 1)
            # attn, attn_weight = self.gat_layers[i](x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            # x = attn.transpose(0, 1)    # (batch_size, n_nodes, node_emb_dim)
        return self.norm(x)


# =======================


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


if __name__ == '__main__':
    batch_size = 2
    n_nodes = 5
    node_dim = 256

    node_emb = torch.rand((batch_size, n_nodes, node_dim), dtype=torch.float)
    adj = torch.LongTensor(np.random.randint(0, 2, (batch_size, n_nodes, n_nodes)))

    print(node_emb.shape, adj.shape)

    g = GraphEncoder(d_model=node_dim, n_layer=2, n_head=4, attn_type='cat')
    out = g.forward(node_emb, adj)
    # g = SpGAT(in_features=node_dim, out_features=node_dim, dropout=0.1, alpha=1e-2,
    #                           nlayers=2, nheads=4)
    # out = g.forward(node_emb, adj)
    print(out.shape)