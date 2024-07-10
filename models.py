import torch

from transformers import AutoModel
from torch import nn
from torch_geometric.nn import GCNConv

from layers import HGTConv
from utils import get_device


class GRLayer(torch.autograd.Function):
    @staticmethod
    def forward(bkw, x, lambda_value=0.01):
        bkw.lambda_value = torch.tensor(lambda_value)
        return x.reshape_as(x)

    @staticmethod
    def backward(bkw, prev_gradient):
        post_gradient = prev_gradient.clone()
        return bkw.lambda_value * post_gradient.neg(), None


class AdversarialNet(torch.nn.Module):
  def __init__(self, config):
    super(AdversarialNet, self).__init__()

    self.num_classes = config["num_classes"]
    self.num_classes_adv = config["num_classes_adv"]
    self.embed_size = config["embed_size"]
    self.first_last_avg = config["first_last_avg"]

    self.plm = AutoModel.from_pretrained(config["model_name"])
    config = self.plm.config
    config.type_vocab_size = 4
    self.plm.embeddings.token_type_embeddings = nn.Embedding(
      config.type_vocab_size, config.hidden_size
    )
    self.plm._init_weights(self.plm.embeddings.token_type_embeddings)

    for param in self.plm.parameters():
      param.requires_grad = True

    self.linear_layer = torch.nn.Linear(in_features=self.embed_size, out_features=self.num_classes)
    self.linear_layer_adv = torch.nn.Linear(in_features=self.embed_size, out_features=self.num_classes_adv)
    self.task_linear = torch.nn.Linear(in_features=self.embed_size, out_features=2)

    self.multi_head_att = torch.nn.MultiheadAttention(self.embed_size, 8, batch_first=True)
    self.Q = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
    self.K = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
    self.V = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)

    self._init_weights(self.linear_layer)
    self._init_weights(self.linear_layer_adv)
    self._init_weights(self.Q)
    self._init_weights(self.K)
    self._init_weights(self.V)
    self._init_weights(self.multi_head_att)
    self._init_weights(self.task_linear)

  def _init_weights(self, module):
    """Initialize the weights"""
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(mean=0.0, std=self.plm.config.initializer_range)
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()

  @torch.autocast(device_type="cuda")
  def forward(self, ids_sent1, segs_sent1, att_mask_sent1, visualize=False):
    out_sent1 = self.plm(ids_sent1, token_type_ids=segs_sent1, attention_mask=att_mask_sent1, output_hidden_states=True)

    last_sent1, first_sent1 = out_sent1.hidden_states[-1], out_sent1.hidden_states[1]

    if self.first_last_avg:
      embed_sent1 = torch.div((last_sent1 + first_sent1), 2)
    else:
      embed_sent1 = last_sent1

    tar_mask_sent1 = (segs_sent1 == 0).long()
    tar_mask_sent2 = (segs_sent1 == 1).long()

    H_sent1 = torch.mul(tar_mask_sent1.unsqueeze(2), embed_sent1)
    H_sent2 = torch.mul(tar_mask_sent2.unsqueeze(2), embed_sent1)

    K_sent1 = self.K(H_sent1)
    V_sent1 = self.V(H_sent1)
    Q_sent2 = self.Q(H_sent2)

    att_output = self.multi_head_att(Q_sent2, K_sent1, V_sent1)

    H_sent = torch.mean(att_output[0], dim=1)

    if visualize:
      return H_sent
    
    if self.training:
      batch_size = H_sent.shape[0]
      samples = H_sent[:batch_size // 2, :]
      samples_adv = H_sent[batch_size // 2:, ]

      predictions = self.linear_layer(samples)
      predictions_adv = self.linear_layer_adv(samples_adv)

      mean_grl = GRLayer.apply(torch.mean(embed_sent1, dim=1), .01)
      task_prediction = self.task_linear(mean_grl)

      return predictions, predictions_adv, task_prediction
    else:
      predictions = self.linear_layer(H_sent)

      return predictions


class BaselineModel(torch.nn.Module):
  def __init__(self, config):
    super(BaselineModel, self).__init__()

    self.num_classes = config["num_classes"]
    self.embed_size = config["embed_size"]
    self.first_last_avg = config["first_last_avg"]

    self.plm = AutoModel.from_pretrained(config["model_name"])
    config = self.plm.config
    config.type_vocab_size = 4
    self.plm.embeddings.token_type_embeddings = nn.Embedding(
      config.type_vocab_size, config.hidden_size
    )
    self.plm._init_weights(self.plm.embeddings.token_type_embeddings)

    for param in self.plm.parameters():
      param.requires_grad = True

    self.linear_layer = torch.nn.Linear(in_features=self.embed_size, out_features=self.num_classes)
    self.multi_head_att = torch.nn.MultiheadAttention(self.embed_size, 8, batch_first=True)
    self.Q = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
    self.K = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
    self.V = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)

    self._init_weights(self.linear_layer)
    self._init_weights(self.Q)
    self._init_weights(self.K)
    self._init_weights(self.V)
    self._init_weights(self.multi_head_att)

  def _init_weights(self, module):
    """Initialize the weights"""
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(mean=0.0, std=self.plm.config.initializer_range)
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()

  @torch.autocast(device_type="cuda")
  def forward(self, ids_sent1, segs_sent1, att_mask_sent1, visualize=False):
    out_sent1 = self.plm(ids_sent1, token_type_ids=segs_sent1, attention_mask=att_mask_sent1, output_hidden_states=True)

    last_sent1, first_sent1 = out_sent1.hidden_states[-1], out_sent1.hidden_states[1]

    if self.first_last_avg:
      embed_sent1 = torch.div((last_sent1 + first_sent1), 2)
    else:
      embed_sent1 = last_sent1

    tar_mask_sent1 = (segs_sent1 == 0).long()
    tar_mask_sent2 = (segs_sent1 == 1).long()

    H_sent1 = torch.mul(tar_mask_sent1.unsqueeze(2), embed_sent1)
    H_sent2 = torch.mul(tar_mask_sent2.unsqueeze(2), embed_sent1)

    K_sent1 = self.K(H_sent1)
    V_sent1 = self.V(H_sent1)
    Q_sent2 = self.Q(H_sent2)

    att_output = self.multi_head_att(Q_sent2, K_sent1, V_sent1)

    H_sent = torch.mean(att_output[0], dim=1)

    if visualize:
      return H_sent

    predictions = self.linear_layer(H_sent)

    return predictions



class BaselineModelWithHGT(torch.nn.Module):
  def __init__(self, config, metadata):
    super(BaselineModelWithHGT, self).__init__()

    device = get_device()
    self.num_classes = config["num_classes"]
    self.embed_size = config["embed_size"]
    self.embed_size_gnn = self.embed_size // 2
    self.first_last_avg = config["first_last_avg"]

    self.plm = AutoModel.from_pretrained(config["model_name"])
    config = self.plm.config
    config.type_vocab_size = 4
    self.plm.embeddings.token_type_embeddings = nn.Embedding(
      config.type_vocab_size, config.hidden_size
    )
    self.plm._init_weights(self.plm.embeddings.token_type_embeddings)

    for param in self.plm.parameters():
      param.requires_grad = True

    self.linear_layer = torch.nn.Linear(in_features=self.embed_size, out_features=self.num_classes)
    self.multi_head_att = torch.nn.MultiheadAttention(self.embed_size, 8, batch_first=True)
    self.Q = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
    self.K = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
    self.V = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)

    self.convs = torch.nn.ModuleList([HGTConv(self.embed_size_gnn,self.embed_size_gnn, metadata=metadata) for _ in range(3)])
    self.bns_gnn_hgt = torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features=self.embed_size_gnn) for _ in range(3)])
    self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features=self.embed_size) for _ in range(2)])

    self.node_lin = {}
    self.edge_lin = {}
    self.node_lin_post = {}
    for node_type in metadata[0]:
      self.node_lin[node_type] = [torch.nn.Linear(in_features=300, out_features=self.embed_size_gnn)]
      self.node_lin[node_type].append(torch.nn.Linear(in_features=self.embed_size_gnn, out_features=self.embed_size_gnn))
      self.node_lin[node_type] = torch.nn.ModuleList(self.node_lin[node_type]).to(device)
      self.bns_gnn_node = {
        node_type: torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features=self.embed_size_gnn) for _ in range(2)]).to(device)
      }

    for node_type in metadata[0]:
      self.node_lin_post[node_type] = [torch.nn.Linear(in_features=self.embed_size_gnn, out_features=self.embed_size_gnn)]
      self.node_lin_post[node_type].append(torch.nn.Linear(in_features=self.embed_size_gnn, out_features=self.embed_size_gnn))
      self.node_lin_post[node_type] = torch.nn.ModuleList(self.node_lin_post[node_type]).to(device)
      self.bns_gnn_node_post = {
        node_type: torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features=self.embed_size_gnn) for _ in range(2)]).to(device)
      }
    
    for edge_type in metadata[1]:
      self.edge_lin[edge_type[1]] = [torch.nn.Linear(in_features=300, out_features=self.embed_size_gnn)]
      self.edge_lin[edge_type[1]].append(torch.nn.Linear(in_features=self.embed_size_gnn, out_features=self.embed_size_gnn))
      self.edge_lin[edge_type[1]] = torch.nn.ModuleList(self.edge_lin[edge_type[1]]).to(device)
      self.bns_gnn_edge = {
        edge_type: torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features=self.embed_size_gnn) for i in range(2)]).to(device)
      }

    self.post_concat = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
    self.relu = torch.nn.ReLU()

    self._init_weights(self.linear_layer)
    self._init_weights(self.Q)
    self._init_weights(self.K)
    self._init_weights(self.V)
    self._init_weights(self.multi_head_att)
    self._init_weights(self.convs)
    self._init_weights(self.node_lin)
    self._init_weights(self.edge_lin)
    self._init_weights(self.node_lin_post)
    self._init_weights(self.post_concat)


  def _init_weights(self, module):
    """Initialize the weights"""
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(mean=0.0, std=self.plm.config.initializer_range)
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()
    if isinstance(module, nn.ModuleList):
      for d in module:
        self._init_weights(d)
    

  def reshape_graph_embeddings(self, out, graph_masking, batch_mapping, batch_size=64):
    """
    reshape_graph_embeddings: out (torch.Tensor graph_batch_size x embedding size) -> (torch.Tensor batch_size x 2 x embedding_size)
    """

    nodes = []

    for i in range(batch_size):
      indices = (batch_mapping == i).nonzero(as_tuple=False)
      indices_args = (graph_masking[i, :] == 1).nonzero(as_tuple=False)
      if len(indices_args) > 0 and len(indices) > 0:
        nodes_i = out[indices, :][indices_args, :].reshape(-1, self.embed_size_gnn)
      else:
        #zero embeddings if there is no graph for a specific point
        nodes_i = torch.zeros(2, self.embed_size_gnn, dtype=torch.float32).to("cuda")
      nodes.append(nodes_i)
    
    return torch.stack(nodes, dim=0).to("cuda")

  @torch.autocast(device_type="cuda")
  def forward(self, ids_sent1, segs_sent1, att_mask_sent1, graph, graph_masking, edge_dict, visualize=False):
    out_sent1 = self.plm(ids_sent1, token_type_ids=segs_sent1, attention_mask=att_mask_sent1, output_hidden_states=True)

    last_sent1, first_sent1 = out_sent1.hidden_states[-1], out_sent1.hidden_states[1]

    if self.first_last_avg:
      embed_sent1 = torch.div((last_sent1 + first_sent1), 2)
    else:
      embed_sent1 = last_sent1

    H_sent = torch.mean(embed_sent1, dim=1)

    try:
      num_nodes = len(graph.x)
    except:
      num_nodes = len(graph.x_dict)

    if num_nodes == 0:
      graph_embedding = [0] * len(ids_sent1)
      graph_embedding = torch.tensor(graph_embedding, dtype=torch.long)
    else:

      #print(graph.edge_index_dict)
      x_dict, edge_index_dict = graph.x_dict, graph.edge_index
      print(x_dict["node"].shape)
      print(edge_index_dict)

      x_dict = {
        node_type: self.bns_gnn_node[node_type][1](
          self.relu(
            self.node_lin[node_type][1](
              self.bns_gnn_node[node_type][0](
                self.relu(
                  self.node_lin[node_type][0](x)
                )
              )
            )
          )
        )
        for node_type, x in x_dict.items()
      }
      rel_dict = {
        edge_type: self.bns_gnn_edge[edge_type][1](
          self.relu(
            self.edge_lin[edge_type][1](
              self.bns_gnn_edge[edge_type][0](
                self.relu(
                  self.edge_lin[edge_type][0](x)
                )
              )
            )
          )
        )
        for edge_type, x in edge_dict.items()
      }

      for i in range(len(self.convs)):
        out = self.bns_gnn_hgt[i](self.relu(self.convs[i](x_dict, edge_index_dict, edge_attr_dict=rel_dict)))
      
      out = {
        node_type: self.bns_gnn_node_post[node_type][1](
          self.relu(
            self.node_lin_post[node_type][1](
              self.bns_gnn_node_post[node_type][0](
                self.relu(
                  self.node_lin_post[node_type][0](x)
                )
              )
            )
          )
        )
        for node_type, x in out.x_dict.items()
      }

    out = self.reshape_graph_embeddings(out["node"], graph_masking, graph.batch, len(ids_sent1))
    out = out.view(out.shape[0], -1)
    out = self.bns[0](self.relu(self.post_concat(out)))

    att_output = self.bns[1](H_sent) + out

    if visualize:
      return H_sent

    predictions = self.linear_layer(att_output)

    return predictions


class BaselineModelWithGNN(torch.nn.Module):
  def __init__(self, config):
    super(BaselineModelWithGNN, self).__init__()

    self.num_classes = config["num_classes"]
    self.embed_size = config["embed_size"]
    self.embed_size_gnn = self.embed_size // 2
    self.first_last_avg = config["first_last_avg"]

    self.plm = AutoModel.from_pretrained(config["model_name"])
    config = self.plm.config
    config.type_vocab_size = 4
    self.plm.embeddings.token_type_embeddings = nn.Embedding(
      config.type_vocab_size, config.hidden_size
    )
    self.plm._init_weights(self.plm.embeddings.token_type_embeddings)

    for param in self.plm.parameters():
      param.requires_grad = True

    self.linear_layer = torch.nn.Linear(in_features=self.embed_size, out_features=self.num_classes)
    self.multi_head_att = torch.nn.MultiheadAttention(self.embed_size, 8, batch_first=True)
    self.Q = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
    self.K = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
    self.V = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)

    self.convs = torch.nn.ModuleList([GCNConv(self.embed_size_gnn,self.embed_size_gnn) for i in range(3)])

    self.pre_mlp1 = torch.nn.Linear(in_features=300, out_features=self.embed_size_gnn)
    self.pre_mlp2 = torch.nn.Linear(in_features=self.embed_size_gnn, out_features=self.embed_size_gnn)
    self.post_mlp1 = torch.nn.Linear(in_features=self.embed_size_gnn, out_features=self.embed_size_gnn)
    self.post_mlp2 = torch.nn.Linear(in_features=self.embed_size_gnn, out_features=self.embed_size_gnn)
    self.post_concat = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
    self.relu = torch.nn.ReLU()
    self.bns_gnn = torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features=self.embed_size_gnn) for i in range(7)])
    self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features=self.embed_size) for i in range(2)])

    self._init_weights(self.linear_layer)
    self._init_weights(self.Q)
    self._init_weights(self.K)
    self._init_weights(self.V)
    self._init_weights(self.multi_head_att)
    self._init_weights(self.convs)
    self._init_weights(self.pre_mlp1)
    self._init_weights(self.pre_mlp2)
    self._init_weights(self.post_mlp1)
    self._init_weights(self.post_mlp2)
    self._init_weights(self.post_concat)


  def _init_weights(self, module):
    """Initialize the weights"""
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(mean=0.0, std=self.plm.config.initializer_range)
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()
    if isinstance(module, nn.ModuleList):
      for d in module:
        self._init_weights(d)
    

  def reshape_graph_embeddings(self, out, graph_masking, batch_mapping, batch_size=64):
    """
    reshape_graph_embeddings: out (torch.Tensor graph_batch_size x embedding size) -> (torch.Tensor batch_size x 2 x embedding_size)
    """

    nodes = []

    for i in range(batch_size):
      indices = (batch_mapping == i).nonzero(as_tuple=False)
      indices_args = (graph_masking[i, :] == 1).nonzero(as_tuple=False)
      if len(indices_args) > 0 and len(indices) > 0:
        nodes_i = out[indices, :][indices_args, :].reshape(-1, self.embed_size_gnn)
      else:
        #zero embeddings if there is no graph for a specific point
        nodes_i = torch.zeros(2, self.embed_size_gnn, dtype=torch.float32).to("cuda")
      nodes.append(nodes_i)
    
    return torch.stack(nodes, dim=0).to("cuda")

  @torch.autocast(device_type="cuda")
  def forward(self, ids_sent1, segs_sent1, att_mask_sent1, graph, graph_masking, visualize=False):
    out_sent1 = self.plm(ids_sent1, token_type_ids=segs_sent1, attention_mask=att_mask_sent1, output_hidden_states=True)

    last_sent1, first_sent1 = out_sent1.hidden_states[-1], out_sent1.hidden_states[1]

    if self.first_last_avg:
      embed_sent1 = torch.div((last_sent1 + first_sent1), 2)
    else:
      embed_sent1 = last_sent1

    H_sent = torch.mean(embed_sent1, dim=1)

    if len(graph.x) == 0:
      graph_embedding = [0] * len(ids_sent1)
      graph_embedding = torch.tensor(graph_embedding, dtype=torch.long)
    else:
      x, edge_index = graph.x, graph.edge_index
      print(x.shape)

      x = self.bns_gnn[0](self.relu(self.pre_mlp1(x)))
      x = self.bns_gnn[1](self.relu(self.pre_mlp2(x)))

      for i in range(len(self.convs)):
        out = self.bns_gnn[i+2](self.relu(self.convs[i](x, edge_index)))
      
      out = self.bns_gnn[len(self.convs)+2](self.relu(self.post_mlp1(out)))
      out = self.bns_gnn[len(self.convs)+3](self.relu(self.post_mlp2(out)))

    out = self.reshape_graph_embeddings(out, graph_masking, graph.batch, len(ids_sent1))
    out = out.view(out.shape[0], -1)
    out = self.bns[0](self.relu(self.post_concat(out)))

    att_output = self.bns[1](H_sent) + out

    if visualize:
      return H_sent

    predictions = self.linear_layer(att_output)

    return predictions
