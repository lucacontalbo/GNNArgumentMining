import math
from typing import Dict, List, Optional, Tuple, Union

import torch

from torch_geometric.nn.inits import glorot
from torch import nn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import HeteroDictLinear, HeteroLinear
from torch_geometric.nn.inits import ones
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax
from torch_geometric.utils.hetero import construct_bipartite_edge_index

class HGTConv(MessagePassing):
    r"""The Heterogeneous Graph Transformer (HGT) operator from the
    `"Heterogeneous Graph Transformer" <https://arxiv.org/abs/2003.01332>`_
    paper.

    .. note::

        For an example of using HGT, see `examples/hetero/hgt_dblp.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        hetero/hgt_dblp.py>`_.

    Args:
        in_channels (int or Dict[str, int]): Size of each input sample of every
            node type, or :obj:`-1` to derive the size from the first input(s)
            to the forward method.
        out_channels (int): Size of each output sample.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
            See :meth:`torch_geometric.data.HeteroData.metadata` for more
            information.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if out_channels % heads != 0:
            raise ValueError(f"'out_channels' (got {out_channels}) must be "
                             f"divisible by the number of heads (got {heads})")

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.edge_types_map = {
            edge_type: i
            for i, edge_type in enumerate(metadata[1])
        }

        self.dst_node_types = {key[-1] for key in self.edge_types}

        self.kqv_lin = HeteroDictLinear(self.in_channels,
                                        self.out_channels * 3)

        self.out_lin = HeteroDictLinear(self.out_channels, self.out_channels,
                                        types=self.node_types)

        dim = out_channels // heads
        num_types = heads * len(self.edge_types)

        self.k_rel = HeteroLinear(dim, dim, num_types, bias=False,
                                  is_sorted=True)
        self.v_rel = HeteroLinear(dim, dim, num_types, bias=False,
                                  is_sorted=True)

        self.edge_lin = HeteroDictLinear(self.out_channels,
                                            self.out_channels,
                                            types=[et[1] for et in self.edge_types])

        self.skip = ParameterDict({
            node_type: Parameter(torch.empty(1))
            for node_type in self.node_types
        })

        self.p_rel = ParameterDict()
        for edge_type in self.edge_types:
            edge_type = '__'.join(edge_type)
            self.p_rel[edge_type] = Parameter(torch.empty(1, heads))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.kqv_lin.reset_parameters()
        self.out_lin.reset_parameters()
        self.k_rel.reset_parameters()
        self.v_rel.reset_parameters()
        self.edge_lin.reset_parameters()
        ones(self.skip)
        ones(self.p_rel)

    def _cat(self, x_dict: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, int]]:
        """Concatenates a dictionary of features."""
        cumsum = 0
        outs: List[Tensor] = []
        offset: Dict[str, int] = {}
        for key, x in x_dict.items():
            outs.append(x)
            offset[key] = cumsum
            cumsum += x.size(0)
        return torch.cat(outs, dim=0), offset

    def _construct_src_node_feat(
        self, k_dict: Dict[str, Tensor], v_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj]
    ) -> Tuple[Tensor, Tensor, Dict[EdgeType, int]]:
        """Constructs the source node representations."""
        cumsum = 0
        num_edge_types = len(self.edge_types)
        H, D = self.heads, self.out_channels // self.heads

        # Flatten into a single tensor with shape [num_edge_types * heads, D]:
        ks: List[Tensor] = []
        vs: List[Tensor] = []
        type_list: List[Tensor] = []
        offset: Dict[EdgeType] = {}
        for edge_type in edge_index_dict.keys():
            src = edge_type[0]
            N = k_dict[src].size(0)
            offset[edge_type] = cumsum
            cumsum += N

            # construct type_vec for curr edge_type with shape [H, D]
            edge_type_offset = self.edge_types_map[edge_type]
            type_vec = torch.arange(H, dtype=torch.long).view(-1, 1).repeat(
                1, N) * num_edge_types + edge_type_offset

            type_list.append(type_vec)
            ks.append(k_dict[src])
            vs.append(v_dict[src])

        ks = torch.cat(ks, dim=0).transpose(0, 1).reshape(-1, D)
        vs = torch.cat(vs, dim=0).transpose(0, 1).reshape(-1, D)
        type_vec = torch.cat(type_list, dim=1).flatten()

        k = self.k_rel(ks, type_vec).view(H, -1, D).transpose(0, 1)
        v = self.v_rel(vs, type_vec).view(H, -1, D).transpose(0, 1)

        return k, v, offset

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],  # Support both.
        edge_attr_dict: Optional[Dict[EdgeType, Tensor]] = None
    ) -> Dict[NodeType, Optional[Tensor]]:
        r"""Runs the forward pass of the module.

        Args:
            x_dict (Dict[str, torch.Tensor]): A dictionary holding input node
                features  for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :class:`torch.Tensor` of
                shape :obj:`[2, num_edges]` or a
                :class:`torch_sparse.SparseTensor`.

        :rtype: :obj:`Dict[str, Optional[torch.Tensor]]` - The output node
            embeddings for each node type.
            In case a node type does not receive any message, its output will
            be set to :obj:`None`.
        """
        F = self.out_channels
        H = self.heads
        D = F // H

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        # Compute K, Q, V over node types:
        kqv_dict = self.kqv_lin(x_dict)
        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k.view(-1, H, D)
            q_dict[key] = q.view(-1, H, D)
            v_dict[key] = v.view(-1, H, D)

        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(
            k_dict, v_dict, edge_index_dict)

        edge_index, edge_attr = construct_bipartite_edge_index(
            edge_index_dict, src_offset, dst_offset, edge_attr_dict=self.p_rel,
            num_nodes=k.size(0))

        if edge_attr_dict is not None:
            edge_attr = self.edge_lin(edge_attr_dict)

        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr)

        # Reconstruct output node embeddings dict:
        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            if node_type in self.dst_node_types:
                out_dict[node_type] = out[start_offset:end_offset]

        # Transform output node embeddings:
        a_dict = self.out_lin({
            k:
            torch.nn.functional.gelu(v) if v is not None else v
            for k, v in out_dict.items()
        })

        # Iterate over node types:
        for node_type, out in out_dict.items():
            out = a_dict[node_type]

            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
                f'heads={self.heads})')


class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''
    def __init__(self, n_hid, max_len = 240, dropout = 0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)
    def forward(self, x, t):
        return x + self.lin(self.emb(t))
    

class HGTConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = True, use_RTE = True, **kwargs):
        super(HGTConv, self).__init__(node_dim=0, aggr='add', **kwargs)

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.total_rel     = num_types * num_relations * num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.use_norm      = use_norm
        self.use_RTE       = use_RTE
        self.att           = None
        
        
        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        
        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        '''
            TODO: make relation_pri smaller, as not all <st, rt, tt> pair exist in meta relation list.
        '''
        self.relation_pri   = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(num_types))
        self.drop           = nn.Dropout(dropout)
        
        if self.use_RTE:
            self.emb            = RelTemporalEncoding(in_dim)
        
        glorot(self.relation_att)
        glorot(self.relation_msg)
        
    def forward(self, node_inp, node_type, edge_index, edge_type, edge_time):
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type, \
                              edge_type=edge_type, edge_time = edge_time)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_time):
        '''
            j: source, i: target; <j, i>
        '''
        data_size = edge_index_i.size(0)
        '''
            Create Attention and Message tensor beforehand.
        '''
        res_att     = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg     = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        
        for source_type in range(self.num_types):
            sb = (node_type_j == int(source_type))
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type] 
            for target_type in range(self.num_types):
                tb = (node_type_i == int(target_type)) & sb
                q_linear = self.q_linears[target_type]
                for relation_type in range(self.num_relations):
                    '''
                        idx is all the edges with meta relation <source_type, relation_type, target_type>
                    '''
                    idx = (edge_type == int(relation_type)) & tb
                    if idx.sum() == 0:
                        continue
                    '''
                        Get the corresponding input node representations by idx.
                        Add tempotal encoding to source representation (j)
                    '''
                    target_node_vec = node_inp_i[idx]
                    source_node_vec = node_inp_j[idx]
                    if self.use_RTE:
                        source_node_vec = self.emb(source_node_vec, edge_time[idx])
                    '''
                        Step 1: Heterogeneous Mutual Attention
                    '''
                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(k_mat.transpose(1,0), self.relation_att[relation_type]).transpose(1,0)
                    res_att[idx] = (q_mat * k_mat).sum(dim=-1) * self.relation_pri[relation_type] / self.sqrt_dk
                    '''
                        Step 2: Heterogeneous Message Passing
                    '''
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    res_msg[idx] = torch.bmm(v_mat.transpose(1,0), self.relation_msg[relation_type]).transpose(1,0)   
        '''
            Softmax based on target node's id (edge_index_i). Store attention value in self.att for later visualization.
        '''
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        return res.view(-1, self.out_dim)


    def update(self, aggr_out, node_inp, node_type):
        '''
            Step 3: Target-specific Aggregation
            x = W[node_type] * gelu(Agg(x)) + x
        '''
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx]))
            '''
                Add skip connection with learnable weight self.skip[t_id]
            '''
            alpha = torch.sigmoid(self.skip[target_type])
            if self.use_norm:
                res[idx] = self.norms[target_type](trans_out * alpha + node_inp[idx] * (1 - alpha))
            else:
                res[idx] = trans_out * alpha + node_inp[idx] * (1 - alpha)
        return res

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)