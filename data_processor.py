import torch
import codecs
import json
import pandas as pd
import numpy as np
import functools
import ast
from pathlib import Path

from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch, HeteroData
from transformers import AutoTokenizer, pipeline
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from utils import get_device, get_json

JSON_PATH = Path("json/")
PRETRAINED_EMB_PATHS = Path("pretrained_embs/")

class dataset(Dataset):
    def __init__(self, examples):
        super(dataset, self).__init__()
        self.examples = examples

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)


def collate_fn(examples):
    ids_sent1, segs_sent1, att_mask_sent1, graph, graph_masking, edge_dict, labels = zip(*examples)
    dev = get_device()

    ids_sent1 = torch.tensor(list(ids_sent1), dtype=torch.long).to(dev)
    segs_sent1 = torch.tensor(list(segs_sent1), dtype=torch.long).to(dev)
    att_mask_sent1 = torch.tensor(list(att_mask_sent1), dtype=torch.long).to(dev)
    graph = Batch.from_data_list(list(graph)).to(dev)
    graph_masking = torch.tensor(list(graph_masking), dtype=torch.long).to(dev)
    try:
        labels = torch.tensor(list(labels), dtype=torch.long).to(dev)
    except:
        pass

    return ids_sent1, segs_sent1, att_mask_sent1, graph, graph_masking, edge_dict[0], labels

def collate_fn_adv(examples):
    ids_sent1, segs_sent1, att_mask_sent1, graph, graph_masking, edge_dict, labels = map(list, zip(*examples))
    dev = get_device()

    ids_sent1 = torch.tensor(ids_sent1, dtype=torch.long).to(dev)
    segs_sent1 = torch.tensor(segs_sent1, dtype=torch.long).to(dev)
    att_mask_sent1 = torch.tensor(att_mask_sent1, dtype=torch.long).to(dev)
    position_sep = torch.tensor(position_sep, dtype=torch.long).to(dev)

    return ids_sent1, segs_sent1, att_mask_sent1, position_sep, labels

            

class DataProcessor:

  def __init__(self,config, device):
    self.config = config
    self.device = device
    self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
    self.max_sent_len = config["max_sent_len"]
    self.max_num_nodes = 1024
    self.graphrelation2words = get_json(JSON_PATH / "graphrelation2words.json")

  def __str__(self,):
    pattern = """General data processor: \n\n Tokenizer: {}\n\nMax sentence length: {}""".format(self.config["model_name"], self.max_sent_len)
    return pattern

  def _get_examples(self, dataset, dataset_type="train"):
    examples = []

    for row in tqdm(dataset, desc="tokenizing..."):
      id, sentence1, sentence2, graph, arg0_pos, arg1_pos, edge_dict, label = row

      sentence1_length = len(self.tokenizer.encode(sentence1))
      sentence2_length = len(self.tokenizer.encode(sentence2))

      ids_sent1 = self.tokenizer.encode(sentence1, sentence2)
      segs_sent1 = [0] * sentence1_length + [1] * (sentence2_length)

      graph_masking = [0] * self.max_num_nodes
      if arg0_pos != -1 and arg1_pos != -1:
        graph_masking[arg0_pos] = 1
        graph_masking[arg1_pos] = 1

      assert len(ids_sent1) == len(segs_sent1)

      pad_id = self.tokenizer.encode(self.tokenizer.pad_token, add_special_tokens=False)[0]

      if len(ids_sent1) < self.max_sent_len:
        res = self.max_sent_len - len(ids_sent1)
        att_mask_sent1 = [1] * len(ids_sent1) + [0] * res
        ids_sent1 += [pad_id] * res
        segs_sent1 += [0] * res
      else:
        ids_sent1 = ids_sent1[:self.max_sent_len]
        segs_sent1 = segs_sent1[:self.max_sent_len]
        att_mask_sent1 = [1] * self.max_sent_len

      example = [ids_sent1, segs_sent1, att_mask_sent1, graph, graph_masking, edge_dict, label]

      examples.append(example)

    print(f"finished preprocessing examples in {dataset_type}")

    return examples
  
  def _get_nodes_and_edges_from_graph(self, graph: list[list[str]], remove_duplicates: bool = False, already_found_graph = None) -> tuple[dict, list[list[int]], list[str]]:
    """
    _get_nodes_and_edges_from_graph: list[list[str]] -> (dict, list[list[int]], list[str])

    This method extracts nodes and edges from graphs represented as walks (list of strings, where the strings represent the nodes).
    Nodes are returned as a dict mapping the node text into an id.
    Edges are returned in the format expected by PyG .edge_index() method.
    """

    if already_found_graph is None:
        counter = 0
        node_ids = {}
        edge_index = [[],[]]
        edge_type = []
        links_added = set()
        general_node = "graph one"
        node_ids[general_node] = counter
        counter += 1
    else:
        node_ids = already_found_graph["node_ids"]
        edge_index = already_found_graph["edge_index"]
        edge_type = already_found_graph["edge_type"]
        links_added = set()
        counter = len(node_ids)

        general_node = "graph two"
        node_ids[general_node] = counter
        counter += 1

    for j, walk in enumerate(graph):
      #if j == 50: break
      for i in range(0,len(walk),2):
        #walk[i] = walk[i].strip()
        if walk[i] not in node_ids.keys():
           node_ids[walk[i]] = counter
           if (node_ids[walk[i]], node_ids[general_node]) not in links_added:
             edge_index[0].append(node_ids[walk[i]])
             edge_index[1].append(node_ids[general_node])
             edge_type.append("belongs")
             links_added.add((node_ids[walk[i]], node_ids[general_node]))
           counter += 1
      
      for i in range(1,len(walk),2):
        #walk[i] = walk[i].strip()
        if remove_duplicates and (node_ids[walk[i-1]], node_ids[walk[i+1]]) in links_added:
          continue

        edge_index[0].append(node_ids[walk[i-1]])
        edge_index[1].append(node_ids[walk[i+1]])
        edge_type.append(walk[i])

        links_added.add((node_ids[walk[i-1]], node_ids[walk[i+1]]))

    return node_ids, edge_index, edge_type

  @functools.cache
  def _get_glove_embeddings(self, emb_size: int) -> tuple[torch.tensor, dict]:
    words = []
    word2idx = {}
    embeddings = []
    idx = 0

    with open(PRETRAINED_EMB_PATHS / f"glove/glove.6B.{emb_size}d.txt", "rb") as reader:
      for l in reader:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        embeddings.append([float(el) for el in line[1:]])

    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    return embeddings, word2idx
  
  @functools.cache
  def _get_word2vec_embeddings(self):
     pass
     
  def _get_embedding_from_word(self, text, word_embeddings, word2idx, emb_size=300):
    node_emb = []
    if text == '':
      #fallback for empty nodes
      node_emb.append(torch.normal(mean=0, std=1, size=(emb_size,), dtype=torch.float32))
    else:
      for word in text.split():
        if word not in word2idx.keys():
          word_emb = torch.normal(mean=0, std=1, size=(emb_size,), dtype=torch.float32)
        else:
          word_emb = torch.tensor(word_embeddings[word2idx[word], :], dtype=torch.float32)
        node_emb.append(word_emb)

    node_emb = torch.stack(node_emb, dim=0)
    node_emb = torch.mean(node_emb, dim=0)

    return node_emb

  def _get_node_features(self, node_ids: dict, pre_trained_emb = "glove", emb_size=300) -> torch.tensor:
    if pre_trained_emb == "glove":
       word_embeddings, word2idx = self._get_glove_embeddings(emb_size)
    elif pre_trained_emb == "word2vec":
       word_embeddings, word2idx = self._get_word2vec_embeddings()
    else:
       raise ValueError(f"Unknown pre-trained embeddings {pre_trained_emb}. Please choose between 'glove' and 'word2vec'")

    node_embeddings = torch.tensor([], dtype=torch.float32)

    for text, id in node_ids.items():
      stripped_text = text.strip()
      if stripped_text == '':
        node_emb = self._get_embedding_from_word(text, word_embeddings, word2idx, emb_size)
      else:
        node_emb = self._get_embedding_from_word(stripped_text, word_embeddings, word2idx, emb_size)

      node_embeddings = torch.cat([node_embeddings, node_emb.unsqueeze(0)], dim=0)
    
    return node_embeddings
    
  def _get_edge_features(self, edge_types, pre_trained_emb = "glove", emb_size=300):
    if pre_trained_emb == "glove":
       word_embeddings, word2idx = self._get_glove_embeddings(emb_size)
    elif pre_trained_emb == "word2vec":
       word_embeddings, word2idx = self._get_word2vec_embeddings()
    else:
       raise ValueError(f"Unknown pre-trained embeddings {pre_trained_emb}. Please choose between 'glove' and 'word2vec'")

    edge_embeddings = torch.tensor([], dtype=torch.float32)

    for edge_type in edge_types:
       translated_edge_type = self.graphrelation2words[edge_type.strip()]
       edge_emb = self._get_embedding_from_word(translated_edge_type, word_embeddings, word2idx)
       edge_embeddings = torch.cat([edge_embeddings, edge_emb.unsqueeze(0)], dim=0)

    return edge_embeddings

  def graph_to_pyg(self, graph1, graph2=None, emb_size=300, text1="", text2=""):
    node_ids, edge_index, edge_type = self._get_nodes_and_edges_from_graph(graph1)
    if graph2 is not None:
        node_ids, edge_index, edge_type = self._get_nodes_and_edges_from_graph(graph2, already_found_graph={"node_ids": node_ids, "edge_index": edge_index, "edge_type": edge_type})

    node_feature_matrix = self._get_node_features(node_ids, emb_size=emb_size).to(get_device())
    edge_feature_matrix = self._get_edge_features(edge_type, emb_size=emb_size).to(get_device())

    if self.config["use_hgraph"] or self.config["use_rgcn"]:
      data = HeteroData()
      data["node"].x = node_feature_matrix

      edges = {link: [[], []] for link in set(edge_type)}
      for link_name, edge_index_pair in zip(edge_type, torch.tensor(edge_index).transpose(0,1)):
        edges[link_name][0].append(edge_index_pair[0])
        edges[link_name][1].append(edge_index_pair[1])
      
      #for k,v in self.graphrelation2words.items():
      #  data["node", v, "node"].edge_index = torch.tensor([], dtype=torch.int64)

      for k,v in edges.items():
        data["node", self.graphrelation2words[k], "node"].edge_index = torch.tensor(v, dtype=torch.int64)
      
      edge_dict = {}
      relations = set(edge_type)
      for i, rel in enumerate(edge_type):
        if rel in edge_dict.keys():
          continue
        edge_dict[("node",rel,"node")] = edge_feature_matrix[i,:].repeat(edge_type.count(rel)).reshape(-1, emb_size)

        if len(relations) == len(edge_dict.keys()):
          break
  
    else:
      data = Data(x=node_feature_matrix, edge_index=torch.tensor(edge_index, dtype=torch.int64), edge_attr=edge_feature_matrix)
      edge_dict = {}

    if len(node_ids.keys()) > 0:
      arg0_pos, arg1_pos = node_ids["graph one"], node_ids["graph two"]

      """longest_node1 = get_longest_string(graph1)
      longest_node2 = get_longest_string(graph2)
      if longest_node1 is None and longest_node2 is not None:
          longest_node1 = longest_node2
      elif longest_node2 is None and longest_node1 is not None:
          longest_node2 = longest_node1
      elif longest_node1 is None and longest_node2 is None:
          raise ValueError("graph is empty")

      arg0_pos, arg1_pos = node_ids[longest_node1], node_ids[longest_node2]"""
    else:
      arg0_pos, arg1_pos = -1, -1

    return data, arg0_pos, arg1_pos, edge_dict

def get_longest_string(graph):
    max_length = 0
    element = None

    for el in graph:
        if len(el[0]) > max_length:
            max_length = len(el[0])
            element = el[0]

    return element

class DiscourseMarkerProcessor(DataProcessor):

  def __init__(self, config, device):
    super(DiscourseMarkerProcessor, self).__init__(config, device)

    self.mapping = get_json(JSON_PATH / "word_to_target.json")
    self.id_to_word = get_json(JSON_PATH / "id_to_word.json")

  def process_dataset(self, dataset, name="train"):
    result = []
    new_dataset = []

    for sample in dataset:
      if self.id_to_word[str(sample["label"])] not in self.mapping.keys():
        continue

      new_dataset.append([sample["sentence1"], sample["sentence2"], self.mapping[self.id_to_word[str(sample["label"])]]])

    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    labels = []

    for i, sample in tqdm(enumerate(new_dataset), desc="processing labels..."):
      labels.append([sample[-1]])

    print("one hot encoding...")
    labels = one_hot_encoder.fit_transform(labels)

    for i, (sample, label) in tqdm(enumerate(zip(new_dataset, labels)), desc="creating results..."):
      result.append([f"{name}_{i}", sample[0], sample[1], HeteroData(), -1, -1, {}, label])

    examples = self._get_examples(result, name)
    return examples


class StudentEssayProcessor(DataProcessor):

  def __init__(self, config, device):
    super(StudentEssayProcessor,self).__init__(config, device)


  def read_input_files(self, file_path, name="train", pipe=None):

      result_train = []
      result_dev = []
      result_test = []

      df = pd.read_csv(file_path, index_col=0)
      for i,row in tqdm(df.iterrows()):
        sample_id = row.iloc[0]
        sent = row.iloc[1].strip()
        target = row.iloc[2].strip()
        if pipe is not None:
          ds_marker = pipe(f"{sent}</s></s>{target}")[0]["label"]
          ds_marker = ds_marker.replace("_", " ")
          ds_marker = ds_marker[0].upper() + ds_marker[1:]
          target = target[0].lower() + target[1:]
          target = ds_marker + " " + target

        label = row.iloc[4]
        split = row.iloc[5]
        graph1, graph2 = ast.literal_eval(row.iloc[10]), ast.literal_eval(row.iloc[11])
        graph, arg0_pos, arg1_pos, edge_dict = self.graph_to_pyg(graph1, graph2)

        if not label:
          l = [1,0]
        else:
          l = [0,1]
              
        if split == "train":
          result_train.append([sample_id, sent, target, graph, arg0_pos, arg1_pos, edge_dict, l])
        elif split == "dev":
          result_dev.append([sample_id, sent, target, graph, arg0_pos, arg1_pos, edge_dict, l])
        elif split == "test":
          result_test.append([sample_id, sent, target, graph, arg0_pos, arg1_pos, edge_dict, l])
        else:
          raise ValueError(f"unknown dataset split: {split}")

      examples_train = self._get_examples(result_train, name)
      examples_dev = self._get_examples(result_dev, name)
      examples_test = self._get_examples(result_test, name)

      return examples_train, examples_dev, examples_test


class DebateProcessor(DataProcessor):

  def __init__(self, config, device):
    super(DebateProcessor,self).__init__(config, device)

  def read_input_files(self, file_path, name="train", pipe=None):
      result_train = []
      result_dev = []
      result_test = []

      df = pd.read_csv(file_path, index_col=0)
      for i,row in df.iterrows():
        sample_id = row.iloc[0]
        sent = row.iloc[1].strip()
        target = row.iloc[2].strip()

        label = row.iloc[4]
        split = row.iloc[5]
        graph1, graph2 = ast.literal_eval(row.iloc[10]), ast.literal_eval(row.iloc[11])
        graph, arg0_pos, arg1_pos, edge_dict = self.graph_to_pyg(graph1, graph2)

        if pipe is not None:
          ds_marker = self.pipe(f"{sent}</s></s>{target}")[0]["label"]
          ds_marker = ds_marker.replace("_", " ")
          ds_marker = ds_marker[0].upper() + ds_marker[1:]
          target = target[0].lower() + target[1:]
          target = ds_marker + " " + target

        l = [0,0]
        if not label:
          l = [1,0]
        else:
          l = [0,1]

        if split == "train":
          result_train.append([sample_id, sent, target, graph, arg0_pos, arg1_pos, edge_dict, l])
        elif split == "dev":
          result_dev.append([sample_id, sent, target, graph, arg0_pos, arg1_pos, edge_dict, l])
        elif split == "test":
          result_test.append([sample_id, sent, target, graph, arg0_pos, arg1_pos, edge_dict, l])
        else:
          raise ValueError(f"unknown dataset split: {split}")

      examples_train = self._get_examples(result_train, name)
      examples_dev = self._get_examples(result_dev, name)
      examples_test = self._get_examples(result_test, name)

      return examples_train, examples_dev, examples_test


class MARGProcessor(DataProcessor):

  def __init__(self, config, device):
    super(MARGProcessor, self).__init__(config, device)
    #self.pipe = pipeline("text-classification", model="sileod/roberta-base-discourse-marker-prediction")

  def read_input_files(self, file_path, name="train", pipe=None):

      result_train = []
      result_dev = []
      result_test = []

      df = pd.read_csv(file_path)
      df = df.drop("Unnamed: 0", axis="columns")
      for i,row in tqdm(df.iterrows()):
              sample_id = row.iloc[0]
              sent = row.iloc[1].strip()
              target = row.iloc[2].strip()

              if pipe is not None:
                ds_marker = self.pipe(f"{sent}</s></s>{target}")[0]["label"]
                ds_marker = ds_marker.replace("_", " ")
                ds_marker = ds_marker[0].upper() + ds_marker[1:]
                target = target[0].lower() + target[1:]
                target = ds_marker + " " + target

              label = row.iloc[3].strip()
              split = row.iloc[-3]
              graph1, graph2 = ast.literal_eval(row.iloc[9]), ast.literal_eval(row.iloc[10])
              graph, arg0_pos, arg1_pos, edge_dict = self.graph_to_pyg(graph1, graph2, text1=sent, text2=target)

              l=[0,0,0]
              if label == 'support':
                l = [1,0,0]
              elif label == 'attack':
                l = [0,1,0]
              elif label == 'neither':
                l = [0,0,1]

              if split == "train":
                result_train.append([sample_id, sent, target, graph, arg0_pos, arg1_pos, edge_dict, l])
              elif split == "dev":
                result_dev.append([sample_id, sent, target, graph, arg0_pos, arg1_pos, edge_dict, l])
              elif split == "test":
                result_test.append([sample_id, sent, target, graph, arg0_pos, arg1_pos, edge_dict, l])
              else:
                raise ValueError(f"unknown dataset split: {split}")

      examples_train = self._get_examples(result_train, name)
      examples_dev = self._get_examples(result_dev, name)
      examples_test = self._get_examples(result_test, name)

      return examples_train, examples_dev, examples_test


class StudentEssayWithDiscourseInjectionProcessor(StudentEssayProcessor):

  def __init__(self, config, device):
    super(StudentEssayWithDiscourseInjectionProcessor, self).__init__(config, device)
    self.pipe = pipeline("text-classification", model="sileod/roberta-base-discourse-marker-prediction")

  def read_input_files(self, file_path, name="train"):
      """
      Reads input files in tab-separated format.
      Will split file_paths on comma, reading from multiple files.
      """

      examples = super().read_input_files(file_path, name, pipe=self.pipe)

      return examples


class DebateWithDiscourseInjectionProcessor(DebateProcessor):

  def __init__(self, config, device):
    super(DebateWithDiscourseInjectionProcessor,self).__init__(config, device)
    self.pipe = pipeline("text-classification", model="sileod/roberta-base-discourse-marker-prediction")


  def read_input_files(self, file_path, name="train"):
      """
      Reads input files in tab-separated format.
      Will split file_paths on comma, reading from multiple files.
      """

      examples = super().read_input_files(file_path, name, pipe=self.pipe)

      return examples


class MARGWithDiscourseInjectionProcessor(DataProcessor):

  def __init__(self, config, device):
    super(MARGWithDiscourseInjectionProcessor,self).__init__(config, device)
    self.pipe = pipeline("text-classification", model="sileod/roberta-base-discourse-marker-prediction")

  def read_input_files(self, file_path, name="train"):
      """
      Reads input files in tab-separated format.
      Will split file_paths on comma, reading from multiple files.
      """

      examples = super().read_input_files(file_path, name, pipe=self.pipe)

      return examples
