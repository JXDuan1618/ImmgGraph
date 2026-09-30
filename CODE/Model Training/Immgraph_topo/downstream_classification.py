import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
import dgl
from dgl.nn import SAGEConv, HeteroGraphConv
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import csv
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import json
from sklearn.metrics import explained_variance_score, mean_absolute_percentage_error, median_absolute_error, max_error
import scipy.stats as stats
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_recall_curve, roc_curve
import re
import copy
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split


# ==============================
# Reproducibility
# ==============================
def set_seed(seed=1):
    """Set all random seeds for reproducibility"""
    import os
    import random
    import numpy as np
    import torch

    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Set deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # Set DGL random seed if available
    try:
        import dgl
        dgl.seed(seed)
    except:
        pass

    print(f"✅ All random seeds set to {seed} for reproducibility")


set_seed(1)


def normalize_graphs_with_train_stats(train_graphs, test_graphs, node_types):

    train_stats = {}
    for ntype in node_types:
        feats = []
        for g in train_graphs:
            if ntype in g.ntypes and "feat" in g.nodes[ntype].data:
                feats.append(g.nodes[ntype].data["feat"])

        if feats:
            all_feat = torch.cat(feats, dim=0)
            mean = all_feat.mean(dim=0, keepdim=True)
            std = all_feat.std(dim=0, keepdim=True) + 1e-6
            train_stats[ntype] = {'mean': mean, 'std': std}


    train_graphs_normalized = []
    for g in train_graphs:
        g_new = copy.deepcopy(g)
        for ntype in g_new.ntypes:
            if ntype in train_stats and "feat" in g_new.nodes[ntype].data:
                feat = g_new.nodes[ntype].data["feat"]
                feat = (feat - train_stats[ntype]['mean']) / train_stats[ntype]['std']
                g_new.nodes[ntype].data["feat"] = feat
        train_graphs_normalized.append(g_new)


    test_graphs_normalized = []
    for g in test_graphs:
        g_new = copy.deepcopy(g)
        for ntype in g_new.ntypes:
            if ntype in train_stats and "feat" in g_new.nodes[ntype].data:
                feat = g_new.nodes[ntype].data["feat"]
                feat = (feat - train_stats[ntype]['mean']) / train_stats[ntype]['std']
                g_new.nodes[ntype].data["feat"] = feat
        test_graphs_normalized.append(g_new)

    return train_graphs_normalized, test_graphs_normalized, train_stats


# ==============================
# Load Labels
# ==============================
print("Loading Excel labels...")
cls_xlsx = r"patient_classification_290.xlsx"
assert os.path.exists(cls_xlsx), f"Excel not found: {cls_xlsx}"

df = pd.read_excel(cls_xlsx, dtype={"serial_id": str})


df["serial_id"] = df["serial_id"].astype(str).str.strip().str.upper()

# ====== Configuration: target column for multiclass classification ======
LABEL_COL = "Histopathology"

raw = df[LABEL_COL].astype(str).str.strip().str.lower()

uniq_vals = [v for v in sorted(raw.unique()) if v not in {"nan", ""}]
assert len(uniq_vals) > 0, f"[Label] Column {LABEL_COL} contains no valid values. Please check the Excel column name or contents."

auto_mapping = {v:i for i, v in enumerate(uniq_vals)}


df["label"] = raw.map(auto_mapping).astype("Int64")
df = df.dropna(subset=["label"]).copy()
df["label"] = df["label"].astype(int)


NUM_CLASSES = len(auto_mapping)


y_map = dict(zip(df["serial_id"], df["label"]))

print(f"[Label] Target column: {LABEL_COL}")
print(f"[Label] Class mapping: {auto_mapping}")
print(f"[Label] count:\n{df['label'].value_counts().sort_index().to_string()}")
print(f"✅ Loaded {len(y_map)} labels from Excel (serial_id & {LABEL_COL}).")
print(f"✅ Loaded {len(y_map)} labels from Excel (serial_id & TERT_mutation).")

# ==============================
# Load Graphs
# ==============================
print("Loading heterograph files...")
graph_dir = r"subgraph_ZZU_new"
graph_paths = [f for f in os.listdir(graph_dir) if f.endswith(".dgl")]

# Step 2: process graphs
all_graphs = []
all_labels = []
graph_paths = sorted(graph_paths)
all_serial_ids = []
all_filenames = []

max_dims = defaultdict(int)
for fname in graph_paths:
    if not fname.lower().endswith(".dgl"):
        continue
    g = dgl.load_graphs(os.path.join(graph_dir, fname))[0][0]
    for ntype in g.ntypes:
        if "h" in g.nodes[ntype].data:
            max_dims[ntype] = max(max_dims[ntype], g.nodes[ntype].data["h"].shape[1])

print("Max feature dims per node type:", dict(max_dims))

# load Graph
for fname in graph_paths:
    if not fname.lower().endswith(".dgl"):
        continue

    base = os.path.splitext(fname)[0]
    m = re.match(r"^patient_(.+?)_subgraph$", base, flags=re.IGNORECASE)
    if not m:
        print(f"skip: {fname}")
        continue

    sid = m.group(1).upper()
    if sid not in y_map:
        print(f"no ID label: {sid} (file={fname}),skip")
        continue

    graphs, _ = dgl.load_graphs(os.path.join(graph_dir, fname))
    if len(graphs) == 0:
        print("⚠️ Empty graph file:", fname)
        continue
    g = graphs[0]


    for ntype in g.ntypes:
        if "h" not in g.nodes[ntype].data:
            g.nodes[ntype].data["h"] = torch.zeros(
                (g.num_nodes(ntype), max_dims.get(ntype, 1)),
                dtype=torch.float32
            )
            continue

        feat = torch.nan_to_num(g.nodes[ntype].data["h"], nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)


        cur_dim = feat.shape[1]
        target_dim = max_dims.get(ntype, cur_dim)
        if cur_dim < target_dim:
            pad = torch.zeros(feat.shape[0], target_dim - cur_dim, dtype=torch.float32)
            feat = torch.cat([feat, pad], dim=1)
        elif cur_dim > target_dim:
            feat = feat[:, :target_dim]

        g.nodes[ntype].data["feat"] = feat


        for k in list(g.nodes[ntype].data.keys()):
            if k != "feat":
                del g.nodes[ntype].data[k]


    for etype in g.canonical_etypes:
        for key in list(g.edges[etype].data.keys()):
            if key != "_ID":
                del g.edges[etype].data[key]

    all_graphs.append(g)
    all_labels.append(int(y_map[sid]))
    all_serial_ids.append(sid)
    all_filenames.append(fname)

print("Dataset samples:")
for i in range(min(10, len(all_labels))):
    print(f"{i}: patient ID={all_serial_ids[i]}, graph file={all_filenames[i]}, y={all_labels[i]}")

pos_weight = None
pos_rate = np.mean(all_labels) if len(all_labels) else 0.5
if 0 < pos_rate < 1:
    pos_weight = torch.tensor((1 - pos_rate) / max(pos_rate, 1e-6), dtype=torch.float32)
print(
    f"[Class balance] pos_rate={pos_rate:.3f}, pos_weight={float(pos_weight) if pos_weight is not None else None}")
# ==============================
# Dataset
# ==============================
class GraphDataset(Dataset):
    def __init__(self, graphs, labels, serial_ids, augment=False):
        self.graphs = graphs
        self.labels = labels  # (time, event)
        self.serial_ids = serial_ids
        self.augment = augment

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        label = self.labels[idx]
        serial_id = self.serial_ids[idx]  # 获取 serial_i

        if self.augment and random.random() < 0.5:
            torch.manual_seed(idx)
            for ntype in graph.ntypes:
                if "feat" in graph.nodes[ntype].data:
                    noise = torch.randn_like(graph.nodes[ntype].data["feat"]) * 0.1
                    graph.nodes[ntype].data["feat"] = graph.nodes[ntype].data["feat"] + noise

        return graph, label, serial_id  # 返回 serial_id



def collate_fn(batch):
    graphs, labels, serial_ids = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    ys = torch.tensor(labels, dtype=torch.float32)
    return batched_graph, ys, serial_ids

def collate_fn_multicls(batch):
    graphs, labels, serial_ids = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    ys = torch.tensor(labels, dtype=torch.long)
    return batched_graph, ys, serial_ids



# ==============================
# Heterogeneous Graph Neural Network
# ==============================
class HeteroSAGEConv(nn.Module):
    def __init__(self, in_dims, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Create GNN layers for each node type
        self.gnn_layers = nn.ModuleDict()
        for ntype, in_dim in in_dims.items():
            layers = []
            for i in range(num_layers):
                if i == 0:
                    layers.append(SAGEConv(in_dim, hidden_dim, aggregator_type='mean'))
                else:
                    layers.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean'))
            self.gnn_layers[ntype] = nn.ModuleList(layers)

        # Add layer normalization for better training stability
        self.layer_norms = nn.ModuleDict()
        for ntype in in_dims.keys():
            self.layer_norms[ntype] = nn.LayerNorm(hidden_dim)

    def forward(self, g):
        # Apply GNN layers to each node type separately
        for ntype in g.ntypes:
            if ntype in self.gnn_layers:
                x = g.nodes[ntype].data["feat"]
                for i, layer in enumerate(self.gnn_layers[ntype]):
                    # Create a subgraph for this node type only
                    subg = g.node_type_subgraph([ntype])
                    x = layer(subg, x)

                    # Apply layer normalization for better training stability
                    if ntype in self.layer_norms:
                        x = self.layer_norms[ntype](x)

                    if i < self.num_layers - 1:  # Don't apply activation after last layer
                        x = F.relu(x)
                        x = F.dropout(x, p=self.dropout, training=self.training)

                g.nodes[ntype].data["h"] = x
        return g


class ImprovedGraphEmbeddingMLP(nn.Module):
    def __init__(self, in_dims, hidden_dim=128, out_dim=3, dropout=0.5, num_gnn_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout

        # Add GNN layers
        self.gnn = HeteroSAGEConv(in_dims, hidden_dim, num_gnn_layers, dropout)

        # MLP for final prediction - Simplified architecture to reduce overfitting
        self.mlp = None

    def forward(self, g):
        # Apply GNN layers first
        g = self.gnn(g)

        feats = []
        batch_num_nodes = []

        for ntype in g.ntypes:
            x = g.nodes[ntype].data["h"]  # Use GNN output
            batch_num_nodes.append(g.batch_num_nodes(ntype))
            feats.append(torch.split(x, tuple(g.batch_num_nodes(ntype).cpu().numpy())))

        num_graphs = g.batch_size
        graph_feats = []
        for i in range(num_graphs):
            node_vecs = [f[i].flatten(start_dim=0) for f in feats]
            graph_feat = torch.cat(node_vecs, dim=0)
            graph_feats.append(graph_feat)

        flat_feat = torch.stack(graph_feats, dim=0)  # ➤ [batch_size, total_feat_dim]

        if self.mlp is None:
            input_dim = flat_feat.shape[1]
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                # Reduced complexity: only one hidden layer instead of two
                nn.Linear(self.hidden_dim, self.out_dim)
            ).to(flat_feat.device)
            self.init_weights()

        return self.mlp(flat_feat)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# ==============================
# Graph Attention Network (GAT) Model
# ==============================
class HeteroGATConv(nn.Module):
    def __init__(self, in_dims, hidden_dim, num_heads=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # Create GAT layers for each node type
        self.gat_layers = nn.ModuleDict()
        for ntype, in_dim in in_dims.items():
            layers = []
            for i in range(num_layers):
                if i == 0:
                    # layers.append(dgl.nn.GATConv(in_dim, hidden_dim // num_heads, num_heads, dropout=dropout))
                    layers.append(
                        dgl.nn.GATConv(in_dim, hidden_dim // num_heads, num_heads, feat_drop=dropout, attn_drop=dropout,
                                       allow_zero_in_degree=True))
                else:
                    # layers.append(dgl.nn.GATConv(hidden_dim, hidden_dim // num_heads, num_heads, dropout=dropout))
                    layers.append(
                        dgl.nn.GATConv(in_dim, hidden_dim // num_heads, num_heads, feat_drop=dropout, attn_drop=dropout,
                                       allow_zero_in_degree=True))
            self.gat_layers[ntype] = nn.ModuleList(layers)

    def forward(self, g):
        # Apply GAT layers to each node type separately
        for ntype in g.ntypes:
            if ntype in self.gat_layers:
                x = g.nodes[ntype].data["feat"]
                for i, layer in enumerate(self.gat_layers[ntype]):
                    # Create a subgraph for this node type only
                    subg = g.node_type_subgraph([ntype])
                    x = layer(subg, x)
                    if i < self.num_layers - 1:  # Don't apply activation after last layer
                        x = F.relu(x)
                        x = F.dropout(x, p=self.dropout, training=self.training)
                g.nodes[ntype].data["h"] = x
        return g


class GATGraphEmbeddingMLP(nn.Module):
    def __init__(self, in_dims, hidden_dim=128, out_dim=3, dropout=0.5, num_gat_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout

        # Add GAT layers
        self.gat = HeteroGATConv(in_dims, hidden_dim, num_heads=4, num_layers=num_gat_layers, dropout=dropout)

        # MLP for final prediction
        self.mlp = None  # 延迟初始化

    def forward(self, g):
        # Apply GAT layers first
        g = self.gat(g)

        feats = []
        batch_num_nodes = []

        for ntype in g.ntypes:
            x = g.nodes[ntype].data["h"]  # Use GAT output
            batch_num_nodes.append(g.batch_num_nodes(ntype))
            feats.append(torch.split(x, tuple(g.batch_num_nodes(ntype).cpu().numpy())))

        num_graphs = g.batch_size
        graph_feats = []
        for i in range(num_graphs):
            node_vecs = [f[i].flatten(start_dim=0) for f in feats]
            graph_feat = torch.cat(node_vecs, dim=0)
            graph_feats.append(graph_feat)

        flat_feat = torch.stack(graph_feats, dim=0)  # ➤ [batch_size, total_feat_dim]

        if self.mlp is None:
            input_dim = flat_feat.shape[1]
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.BatchNorm1d(self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.hidden_dim // 2, self.out_dim)
            ).to(flat_feat.device)
            self.init_weights()

        return self.mlp(flat_feat)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# ==============================
# Graph Convolutional Network (GCN) Model
# ==============================
class HeteroGCNConv(nn.Module):
    def __init__(self, in_dims, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Create GCN layers for each node type
        self.gcn_layers = nn.ModuleDict()
        for ntype, in_dim in in_dims.items():
            layers = []
            for i in range(num_layers):
                if i == 0:
                    layers.append(dgl.nn.GraphConv(in_dim, hidden_dim, norm='both'))
                else:
                    layers.append(dgl.nn.GraphConv(hidden_dim, hidden_dim, norm='both'))
            self.gcn_layers[ntype] = nn.ModuleList(layers)

    def forward(self, g):
        # Apply GCN layers to each node type separately
        for ntype in g.ntypes:
            if ntype in self.gcn_layers:
                x = g.nodes[ntype].data["feat"]
                for i, layer in enumerate(self.gcn_layers[ntype]):
                    # Create a subgraph for this node type only
                    subg = dgl.add_self_loop(g.node_type_subgraph([ntype]))
                    x = layer(subg, x)
                    if i < self.num_layers - 1:  # Don't apply activation after last layer
                        x = F.relu(x)
                        x = F.dropout(x, p=self.dropout, training=self.training)
                g.nodes[ntype].data["h"] = x
        return g


class GCNGraphEmbeddingMLP(nn.Module):
    def __init__(self, in_dims, hidden_dim=128, out_dim=1, dropout=0.5, num_gcn_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout

        # Add GCN layers
        self.gcn = HeteroGCNConv(in_dims, hidden_dim, num_gcn_layers, dropout)

        # MLP for final prediction
        self.mlp = None  # 延迟初始化

    def forward(self, g):
        # Apply GCN layers first
        g = self.gcn(g)

        feats = []
        batch_num_nodes = []

        for ntype in g.ntypes:
            x = g.nodes[ntype].data["h"]  # Use GCN output
            batch_num_nodes.append(g.batch_num_nodes(ntype))
            feats.append(torch.split(x, tuple(g.batch_num_nodes(ntype).cpu().numpy())))

        num_graphs = g.batch_size
        graph_feats = []
        for i in range(num_graphs):
            node_vecs = [f[i].flatten(start_dim=0) for f in feats]
            graph_feat = torch.cat(node_vecs, dim=0)
            graph_feats.append(graph_feat)

        flat_feat = torch.stack(graph_feats, dim=0)  # ➤ [batch_size, total_feat_dim]

        if self.mlp is None:
            input_dim = flat_feat.shape[1]
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.BatchNorm1d(self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.hidden_dim // 2, self.out_dim)
            ).to(flat_feat.device)
            self.init_weights()

        return self.mlp(flat_feat)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# ==============================
# Graph Transformer Model
# ==============================
class GraphTransformer(nn.Module):
    def __init__(self, in_dims, hidden_dim=128, num_heads=8, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection for each node type
        self.input_proj = nn.ModuleDict()
        for ntype, in_dim in in_dims.items():
            self.input_proj[ntype] = nn.Linear(in_dim, hidden_dim)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])

    def forward(self, g):
        # Project input features
        for ntype in g.ntypes:
            if ntype in self.input_proj:
                x = g.nodes[ntype].data["feat"]
                x = self.input_proj[ntype](x)
                g.nodes[ntype].data["h"] = x

        # Apply transformer layers
        for ntype in g.ntypes:
            if ntype in self.input_proj:
                x = g.nodes[ntype].data["h"]
                # Reshape for transformer: (batch_size, seq_len, hidden_dim)
                batch_size = g.batch_size
                num_nodes = g.batch_num_nodes(ntype)

                # Split by graph
                node_lists = torch.split(x, tuple(num_nodes.cpu().numpy()))

                # Pad sequences to same length
                max_nodes = max(num_nodes)
                padded_x = []
                for nodes in node_lists:
                    if nodes.shape[0] < max_nodes:
                        pad = torch.zeros(max_nodes - nodes.shape[0], self.hidden_dim, device=nodes.device)
                        nodes = torch.cat([nodes, pad], dim=0)
                    padded_x.append(nodes)

                # Stack and apply transformer
                x = torch.stack(padded_x, dim=0)  # (batch_size, max_nodes, hidden_dim)

                for layer in self.transformer_layers:
                    x = layer(x)

                # Unpad and restore to graph
                unpadded_x = []
                for i, num_node in enumerate(num_nodes):
                    unpadded_x.append(x[i, :num_node])

                g.nodes[ntype].data["h"] = torch.cat(unpadded_x, dim=0)

        return g


class TransformerGraphEmbeddingMLP(nn.Module):
    def __init__(self, in_dims, hidden_dim=128, out_dim=1, dropout=0.5, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout

        # Add Transformer layers
        self.transformer = GraphTransformer(in_dims, hidden_dim, num_heads=8, num_layers=num_layers, dropout=dropout)

        # MLP for final prediction
        self.mlp = None

    def forward(self, g):
        # Apply Transformer layers first
        g = self.transformer(g)

        feats = []
        batch_num_nodes = []

        for ntype in g.ntypes:
            x = g.nodes[ntype].data["h"]  # Use Transformer output
            batch_num_nodes.append(g.batch_num_nodes(ntype))
            feats.append(torch.split(x, tuple(g.batch_num_nodes(ntype).cpu().numpy())))

        num_graphs = g.batch_size
        graph_feats = []
        for i in range(num_graphs):
            node_vecs = [f[i].flatten(start_dim=0) for f in feats]  # [各类型节点展平拼接]
            graph_feat = torch.cat(node_vecs, dim=0)
            graph_feats.append(graph_feat)

        flat_feat = torch.stack(graph_feats, dim=0)  # ➤ [batch_size, total_feat_dim]

        if self.mlp is None:
            input_dim = flat_feat.shape[1]
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.BatchNorm1d(self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.hidden_dim // 2, self.out_dim)
            ).to(flat_feat.device)
            self.init_weights()

        return self.mlp(flat_feat)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# ==============================
# Ensemble Model
# ==============================
class EnsembleGraphModel(nn.Module):
    def __init__(self, in_dims, hidden_dim=128, out_dim=1, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout

        # Multiple models
        self.sage_model = ImprovedGraphEmbeddingMLP(in_dims, hidden_dim, out_dim, dropout, num_gnn_layers=1)
        self.gat_model = GATGraphEmbeddingMLP(in_dims, hidden_dim, out_dim, dropout, num_gat_layers=1)
        self.gcn_model = GCNGraphEmbeddingMLP(in_dims, hidden_dim, out_dim, dropout, num_gcn_layers=1)

        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, g):
        # Get predictions from all models
        sage_pred = self.sage_model(g)
        gat_pred = self.gat_model(g)
        gcn_pred = self.gcn_model(g)

        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_pred = (weights[0] * sage_pred +
                         weights[1] * gat_pred +
                         weights[2] * gcn_pred)

        return ensemble_pred


# ==============================
# Improved Cox Partial Likelihood Loss
# ==============================
class ImprovedCoxPHLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, risk, time, event):
        risk = risk.view(-1)
        time = time.view(-1)
        event = event.view(-1)

        # Sort in descending order of time
        idx = torch.argsort(time, descending=True)
        risk = risk[idx]
        event = event[idx]

        # More conservative clamping
        risk = torch.clamp(risk, min=-20.0, max=20.0)

        exp_risk = torch.exp(risk)
        log_cumsum = torch.log(torch.cumsum(exp_risk, dim=0) + 1e-8)

        # The difference
        diff = risk - log_cumsum

        # Negative mean partial log-likelihood
        loss = -torch.mean(diff * event)

        return loss


# ==============================
# Improved Training & Evaluation with Hyperparameters
# ==============================
def train_model_with_hyperparams_cls(model, train_loader, val_loader, epochs=80, lr=1e-4, device="cuda",
                                     fold=0, weight_decay=1e-3, l2_lambda=1e-4,
                                     train_idx=None, val_idx=None, pos_weight=None):
    model.to(device)

    # Trigger one forward pass to initialize the MLP
    with torch.no_grad():
        g0, y0, _ = next(iter(train_loader))
        _ = model(g0.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce_loss  = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # ====== Early stopping & best-epoch caching ======
    best_val_auc   = -float("inf")
    best_state     = None          # Model weights from the best epoch
    best_records   = None          # Per-sample records from the best epoch (Serial_ID, y_true, y_prob)
    best_val_prob  = None          # Predicted probabilities from the best epoch (tensor)
    best_val_y     = None          # Labels from the best epoch (tensor)
    best_epoch     = -1
    patience       = 0
    early_stop     = 25

    for epoch in range(epochs):
        # === Train ===
        model.train()
        total_loss = 0.0
        train_logits, train_y = [], []

        for g, ys, _ in train_loader:
            g = g.to(device)
            y = ys.to(device)
            logit = model(g).view(-1)
            loss  = bce_loss(logit, y)

            if l2_lambda > 0:
                l2 = torch.tensor(0., device=device)
                for p in model.parameters():
                    l2 += torch.norm(p, p=2)
                loss = loss + l2_lambda * l2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            train_logits.append(logit.detach().cpu()); train_y.append(y.detach().cpu())

        train_logits = torch.cat(train_logits); train_y = torch.cat(train_y)
        train_prob   = torch.sigmoid(train_logits)
        train_pred   = (train_prob >= 0.5).int()
        train_acc    = accuracy_score(train_y, train_pred)
        train_f1     = f1_score(train_y, train_pred, zero_division=0)
        try:
            train_auc = roc_auc_score(train_y, train_prob)
        except:
            train_auc = float('nan')

        # === Val ===
        model.eval()
        val_logits, val_y = [], []
        cur_records = []
        with torch.no_grad():
            for g, ys, serial_ids in val_loader:
                g = g.to(device)
                logit = model(g).view(-1)
                val_logits.append(logit.cpu()); val_y.append(ys)

                prob = torch.sigmoid(logit).cpu().numpy().tolist()
                for sid, yy, pp in zip(serial_ids, ys.numpy().tolist(), prob):
                    cur_records.append({"Serial_ID": sid, "y_true": int(yy), "y_prob": float(pp)})

        val_logits = torch.cat(val_logits); val_y = torch.cat(val_y)
        val_prob   = torch.sigmoid(val_logits)
        val_pred   = (val_prob >= 0.5).int()
        val_acc    = accuracy_score(val_y, val_pred)
        val_f1     = f1_score(val_y, val_pred, zero_division=0)
        try:
            val_auc = roc_auc_score(val_y, val_prob)
            val_ap  = average_precision_score(val_y, val_prob)
        except:
            val_auc, val_ap = float('nan'), float('nan')

        scheduler.step(val_auc)
        print(f"Epoch {epoch+1}/{epochs} | loss {total_loss:.4f} | "
              f"Train AUC {train_auc:.3f} F1 {train_f1:.3f} Acc {train_acc:.3f} || "
              f"Val AUC {val_auc:.3f} AP {val_ap:.3f} F1 {val_f1:.3f} Acc {val_acc:.3f}")


        if val_auc > best_val_auc + 1e-6:
            best_val_auc  = val_auc
            patience      = 0
            best_epoch    = epoch + 1

            best_state    = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_records  = list(cur_records)
            best_val_prob = val_prob.detach().cpu().clone()
            best_val_y    = val_y.detach().cpu().clone()
        else:
            patience += 1
            if patience >= early_stop:
                print("Early stopping!")
                break

    # ====== Restore the best model weights and write the CSV based on the best epoch ======
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        # In the unlikely case that best_state was never updated, use the final epoch
        best_records  = cur_records
        best_val_prob = val_prob.detach().cpu()
        best_val_y    = val_y.detach().cpu()

    try:
        precision, recall, thresholds = precision_recall_curve(best_val_y.numpy(), best_val_prob.numpy())
        f1s = 2 * precision * recall / (precision + recall + 1e-12)
        best_thr = thresholds[np.nanargmax(f1s)] if len(thresholds) > 0 else 0.5
    except Exception:
        best_thr = 0.5

    for r in best_records:
        r["pred_label@0.5"]   = int(r["y_prob"] >= 0.5)
        r["pred_label@bestF1"] = int(r["y_prob"] >= float(best_thr))
        r["bestF1_threshold"] = float(best_thr)
        r["best_epoch"]       = int(best_epoch)

    os.makedirs(r"outputs", exist_ok=True)
    out_csv = fr"outputs\fold_{fold+1}_val_pred_Histopathology.csv"
    pd.DataFrame(best_records).to_csv(out_csv, index=False)
    print(f"✅ Saved BEST-epoch (epoch={best_epoch}) validation predictions to {out_csv}")

    return float(best_val_auc)

def train_model_with_hyperparams_multicls(
    model, train_loader, val_loader, epochs=80, lr=1e-4, device="cuda",
    fold=0, weight_decay=1e-3, l2_lambda=1e-4, train_idx=None, val_idx=None,
    num_classes=NUM_CLASSES, class_weights=None
):
    model.to(device)

    with torch.no_grad():
        g0, y0, _ = next(iter(train_loader))
        _ = model(g0.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    best_metric = -float("inf")
    best_state  = None
    best_epoch  = -1
    patience    = 0
    early_stop  = 25
    best_records = None

    best_train_records = None

    for epoch in range(epochs):
        # ===== Train =====
        model.train()
        total_loss = 0.0
        tr_logits, tr_y = [], []
        cur_train_records = []

        for g, ys, serial_ids in train_loader:
            g = g.to(device)
            y = ys.to(device).long()
            logits = model(g)
            loss = criterion(logits, y)

            if l2_lambda > 0:
                l2 = torch.tensor(0., device=device)
                for p in model.parameters():
                    l2 += torch.norm(p, p=2)
                loss = loss + l2_lambda * l2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            tr_logits.append(logits.detach().cpu());
            tr_y.append(y.detach().cpu())


            prob = torch.softmax(logits.detach(), dim=1).cpu().numpy()
            for sid, yy, pp in zip(serial_ids, ys.numpy().tolist(), prob):
                row = {"Serial_ID": sid, "y_true": int(yy)}
                for c in range(num_classes):
                    row[f"y_prob_c{c}"] = float(pp[c])
                cur_train_records.append(row)

        tr_logits = torch.cat(tr_logits); tr_y = torch.cat(tr_y)
        tr_prob   = torch.softmax(tr_logits, dim=1)                    # [N, K]
        tr_pred   = tr_prob.argmax(dim=1)

        from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
        train_acc = accuracy_score(tr_y, tr_pred)
        train_f1  = f1_score(tr_y, tr_pred, average="macro", zero_division=0)

        try:
            train_auc = roc_auc_score(tr_y, tr_prob, multi_class="ovr", average="macro")
        except Exception:
            train_auc = float("nan")

        # ===== Val =====
        model.eval()
        va_logits, va_y, cur_records = [], [], []
        with torch.no_grad():
            for g, ys, serial_ids in val_loader:
                g = g.to(device)
                logits = model(g)
                va_logits.append(logits.cpu()); va_y.append(ys)

                prob = torch.softmax(logits, dim=1).cpu().numpy()
                for sid, yy, pp in zip(serial_ids, ys.numpy().tolist(), prob):
                    row = {"Serial_ID": sid, "y_true": int(yy)}
                    for c in range(num_classes):
                        row[f"y_prob_c{c}"] = float(pp[c])
                    cur_records.append(row)

        va_logits = torch.cat(va_logits); va_y = torch.cat(va_y)
        va_prob   = torch.softmax(va_logits, dim=1)
        va_pred   = va_prob.argmax(dim=1)

        val_acc = accuracy_score(va_y, va_pred)
        val_f1  = f1_score(va_y, va_pred, average="macro", zero_division=0)
        try:
            val_auc = roc_auc_score(va_y, va_prob, multi_class="ovr", average="macro")
        except Exception:
            val_auc = float("nan")

        scheduler.step(val_f1)
        print(f"Epoch {epoch+1}/{epochs} | loss {total_loss:.4f} | "
              f"Train AUC {train_auc:.3f} F1 {train_f1:.3f} Acc {train_acc:.3f} || "
              f"Val AUC {val_auc:.3f} F1 {val_f1:.3f} Acc {val_acc:.3f}")

        metric_now = val_f1
        if metric_now > best_metric + 1e-6:
            best_metric = metric_now
            patience = 0
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_records = list(cur_records)
            best_train_records = list(cur_train_records)
        else:
            patience += 1
            if patience >= early_stop:
                print("Early stopping!")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    for r in best_records:
        probs = [r[f"y_prob_c{c}"] for c in range(num_classes)]
        r["pred_label"] = int(np.argmax(probs))
        r["best_epoch"] = int(best_epoch)

    os.makedirs(r"outputs", exist_ok=True)
    out_csv = fr"outputs\fold_{fold+1}_val_pred_Histopathology.csv"
    pd.DataFrame(best_records).to_csv(out_csv, index=False)
    print(f"✅ Saved BEST-epoch (epoch={best_epoch}) validation predictions to {out_csv}")

    if best_train_records is not None:
        for r in best_train_records:
            probs = [r[f"y_prob_c{c}"] for c in range(num_classes)]
            r["pred_label"] = int(np.argmax(probs))
            r["best_epoch"] = int(best_epoch)

        train_csv = fr"outputs\fold_{fold + 1}_train_pred_Histopathology.csv"
        pd.DataFrame(best_train_records).to_csv(train_csv, index=False)
        print(f"✅ Saved BEST-epoch (epoch={best_epoch}) training predictions to {train_csv}")

    return float(best_metric)

# ==============================
# Optuna Objective Function for Hyperparameter Optimization
# ==============================
def run_optuna_optimization_cls(n_trials=100, seed=42):
    print("Starting Optuna HPO (classification)...")

    def objective(trial):
        hidden_dim     = trial.suggest_int('hidden_dim', 32, 128, step=32)
        dropout        = trial.suggest_float('dropout', 0.2, 0.8)
        lr             = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        num_gnn_layers = trial.suggest_int('num_gnn_layers', 1, 2)
        weight_decay   = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
        l2_lambda      = trial.suggest_float('l2_lambda', 1e-6, 1e-2, log=True)

        in_dims = {ntype: all_graphs[0].nodes[ntype].data["feat"].shape[1]
                   for ntype in all_graphs[0].ntypes}

        dataset = GraphDataset(all_graphs, all_labels, all_serial_ids)
        y_vec   = np.array(all_labels, dtype=int)

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pos_rate = y_vec.mean()
        pos_weight = torch.tensor((1 - pos_rate) / max(pos_rate, 1e-6), dtype=torch.float32)

        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)), y_vec)):
            print(f"\n===== Fold {fold+1} (trial {trial.number}) =====")
            set_seed(seed + fold)

            model = ImprovedGraphEmbeddingMLP(in_dims, hidden_dim, out_dim=1,
                                              dropout=dropout, num_gnn_layers=num_gnn_layers)

            train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx),
                                      batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0)
            val_loader   = DataLoader(torch.utils.data.Subset(dataset, val_idx),
                                      batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0)

            best_auc = train_model_with_hyperparams_cls(
                model, train_loader, val_loader,
                epochs=150, lr=lr, device=device, fold=fold,
                weight_decay=weight_decay, l2_lambda=l2_lambda,
                train_idx=train_idx, val_idx=val_idx,
                pos_weight=pos_weight
            )
            fold_scores.append(best_auc)
            trial.report(float(np.mean(fold_scores)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print("Best AUC:", study.best_value, "Params:", study.best_params)
    results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials)
    }
    out_json = r'outputs\metric\best_params_cls_Histopathology_1.json'
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Optuna results saved to: {out_json}")
    return study

def run_optuna_optimization_multicls(train_graphs_all, train_labels_all, train_serial_ids_all,
                                     n_trials=100, seed=42):
    print("Starting Optuna HPO (multi-class classification)...")

    def objective(trial):
        hidden_dim     = trial.suggest_int('hidden_dim', 32, 256, step=32)
        dropout        = trial.suggest_float('dropout', 0.2, 0.8)
        lr             = trial.suggest_float('lr', 1e-5, 5e-3, log=True)
        num_gnn_layers = trial.suggest_int('num_gnn_layers', 1, 2)
        weight_decay   = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        l2_lambda      = trial.suggest_float('l2_lambda', 1e-6, 1e-3, log=True)


        in_dims = {ntype: all_graphs[0].nodes[ntype].data["feat"].shape[1]
                   for ntype in all_graphs[0].ntypes}

        # datasets
        dataset = GraphDataset(train_graphs_all, train_labels_all, train_serial_ids_all)
        y_vec = np.array(train_labels_all, dtype=int)
        kf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        device  = "cuda" if torch.cuda.is_available() else "cpu"

        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)), y_vec)):

            print(f"\n===== Fold {fold + 1} (trial {trial.number}) =====")
            set_seed(seed + fold)

            train_graphs_raw = [train_graphs_all[i] for i in train_idx]
            val_graphs_raw = [train_graphs_all[i] for i in val_idx]

            train_graphs, val_graphs, train_stats = normalize_graphs_with_train_stats(
                train_graphs_raw,
                val_graphs_raw,
                node_types=list(in_dims.keys())
            )

            train_labels = [train_labels_all[i] for i in train_idx]
            val_labels = [train_labels_all[i] for i in val_idx]
            train_serial_ids = [train_serial_ids_all[i] for i in train_idx]
            val_serial_ids = [train_serial_ids_all[i] for i in val_idx]

            train_dataset = GraphDataset(train_graphs, train_labels, train_serial_ids)
            val_dataset = GraphDataset(val_graphs, val_labels, val_serial_ids)

            from sklearn.utils.class_weight import compute_class_weight
            train_y_fold = y_vec[train_idx]
            classes = np.arange(NUM_CLASSES)
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_y_fold)
            class_w = torch.tensor(weights, dtype=torch.float32)

            # NUM_CLASSES
            model = ImprovedGraphEmbeddingMLP(
                in_dims, hidden_dim=hidden_dim, out_dim=NUM_CLASSES,
                dropout=dropout, num_gnn_layers=num_gnn_layers
            )

            # DataLoader: use the normalized data
            train_loader = DataLoader(train_dataset,
                                      batch_size=16, shuffle=False, collate_fn=collate_fn_multicls, num_workers=0)
            val_loader = DataLoader(val_dataset,
                                    batch_size=16, shuffle=False, collate_fn=collate_fn_multicls, num_workers=0)

            best_metric = train_model_with_hyperparams_multicls(
                model, train_loader, val_loader,
                epochs=150, lr=lr, device=device, fold=fold,
                weight_decay=weight_decay, l2_lambda=l2_lambda,
                train_idx=train_idx, val_idx=val_idx,
                num_classes=NUM_CLASSES, class_weights=class_w
            )

            fold_scores.append(best_metric)

            trial.report(float(np.mean(fold_scores)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print("Best (macro-F1):", study.best_value, "Params:", study.best_params)

    # save JSON
    results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials)
    }
    out_json = r'outputs\metric\best_params_cls_Histopathology.json'
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Optuna (multicls) results saved to: {out_json}")
    return study


def calculate_metrics(true_values, predicted_values):

    true_values = np.array(true_values).flatten()
    predicted_values = np.array(predicted_values).flatten()


    try:
        # MAPE
        mape = mean_absolute_percentage_error(true_values, predicted_values) * 100
    except:
        mape = np.nan

    # MedAE
    medae = median_absolute_error(true_values, predicted_values)

    # EVS
    evs = explained_variance_score(true_values, predicted_values)

    # ME
    me = max_error(true_values, predicted_values)

    try:
        # LogMSE
        log_mse = np.mean(np.square(np.log1p(true_values) - np.log1p(predicted_values)))
    except:
        log_mse = np.nan

    try:
        # SMAPE
        smape = 100 * np.mean(2 * np.abs(predicted_values - true_values) /
                              (np.abs(predicted_values) + np.abs(true_values)))
    except:
        smape = np.nan

    # MBE
    mbe = np.mean(predicted_values - true_values)

    # NMSE
    if np.var(true_values) != 0:
        nmse = np.mean(np.square(predicted_values - true_values)) / np.var(true_values)
    else:
        nmse = np.nan

    # RAE
    denominator = np.sum(np.abs(true_values - np.mean(true_values)))
    if denominator != 0:
        rae = np.sum(np.abs(predicted_values - true_values)) / denominator
    else:
        rae = np.nan

    # RSE
    denominator = np.sum(np.square(true_values - np.mean(true_values)))
    if denominator != 0:
        rse = np.sum(np.square(predicted_values - true_values)) / denominator
    else:
        rse = np.nan

    try:
        # Poisson_Deviance
        safe_true = np.maximum(true_values, 1e-10)
        safe_pred = np.maximum(predicted_values, 1e-10)
        poisson_deviance = 2 * np.sum(safe_true * np.log(safe_true / safe_pred) - (safe_true - safe_pred))
    except:
        poisson_deviance = np.nan

    try:
        # Gamma_Deviance
        safe_true = np.maximum(true_values, 1e-10)
        safe_pred = np.maximum(predicted_values, 1e-10)
        gamma_deviance = 2 * np.sum(np.log(safe_pred / safe_true) + (safe_true / safe_pred) - 1)
    except:
        gamma_deviance = np.nan


    if len(true_values) > 1:
        try:
            pearson_corr, _ = stats.pearsonr(true_values, predicted_values)
        except:
            pearson_corr = np.nan

        try:
            spearman_corr, _ = stats.spearmanr(true_values, predicted_values)
        except:
            spearman_corr = np.nan

        try:
            kendall_corr, _ = stats.kendalltau(true_values, predicted_values)
        except:
            kendall_corr = np.nan
    else:
        pearson_corr = np.nan
        spearman_corr = np.nan
        kendall_corr = np.nan

    try:
        # MGE
        safe_true = np.maximum(true_values, 1e-10)
        safe_pred = np.maximum(predicted_values, 1e-10)
        mge = np.exp(np.mean(np.abs(np.log(safe_pred / safe_true))))
    except:
        mge = np.nan

    try:
        # MSLE
        msle = np.mean(np.square(np.log1p(predicted_values) - np.log1p(true_values)))
        rmsle = np.sqrt(msle)
    except:
        msle = np.nan
        rmsle = np.nan

    # WAE
    weights = np.ones_like(true_values)
    wae = np.average(np.abs(predicted_values - true_values), weights=weights)

    # Direction_Accuracy
    if len(true_values) > 1:
        try:
            direction_accuracy = np.mean((np.diff(true_values) * np.diff(predicted_values)) > 0)
        except:
            direction_accuracy = np.nan
    else:
        direction_accuracy = np.nan

    return mape, medae, evs, me, log_mse, smape, mbe, nmse, rae, rse, poisson_deviance, gamma_deviance, pearson_corr, spearman_corr, kendall_corr, mge, msle, rmsle, wae, direction_accuracy


# save the results of each fold
def save_metrics_to_csv(metrics, file_path):

    columns = ["Fold", "MAPE", "MedAE", "EVS", "ME", "LogMSE", "SMAPE", "MBE", "NMSE",
               "RAE", "RSE", "Poisson_Deviance", "Gamma_Deviance", "Pearson_Corr",
               "Spearman_Corr", "Kendall_Corr", "MGE", "MSLE", "RMSLE", "WAE", "Direction_Accuracy"]

    df = pd.DataFrame(metrics, columns=columns)
    df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

# ==============================
# Export per-omics node embeddings (only "h") at last epoch
# ==============================
# 如果基因名在图里用的是别的键名，把它加到下面列表
NAME_KEYS_CANDIDATES = ["gene_name", "name", "symbol", "gene", "id"]

def _extract_node_names(g, ntype):
    names = None
    for key in NAME_KEYS_CANDIDATES:
        if key in g.nodes[ntype].data:
            val = g.nodes[ntype].data[key]
            try:
                if torch.is_tensor(val):
                    if val.dtype in (torch.int32, torch.int64, torch.float32, torch.float64):
                        names = [f"{key}_{int(x)}" for x in val.view(-1).cpu().tolist()]
                    else:
                        names = [str(x) for x in val.view(-1).cpu().tolist()]
                else:
                    names = [str(x) for x in list(val)]
            except Exception:
                names = None

            if names is not None and len(names) == g.num_nodes(ntype):
                break
            else:
                names = None
    return names

def _ensure_2d(x: torch.Tensor) -> torch.Tensor:

    if x is None:
        return None
    if x.dim() == 3:
        return x.reshape(x.shape[0], -1)
    return x

def export_node_embeddings_by_omics(model, loaders, fold, out_root):

    os.makedirs(out_root, exist_ok=True)
    device = next(model.parameters()).device

    buckets = {"dna": [], "rna": [], "protein": []}

    model.eval()
    with torch.no_grad():
        for loader in loaders:
            for g, _, _, serial_ids in loader:
                g = g.to(device)

                _ = model(g)

                for ntype in g.ntypes:
                    key = ntype.lower()
                    if key not in buckets:
                        continue


                    emb = g.nodes[ntype].data.get("h", None)
                    emb = _ensure_2d(emb)
                    if emb is None:
                        continue


                    names_all = _extract_node_names(g, ntype)


                    counts = g.batch_num_nodes(ntype).cpu().tolist()
                    off = 0
                    for gi, cnt in enumerate(counts):
                        sid = serial_ids[gi]
                        for j in range(cnt):
                            row = {
                                "Serial_ID": sid,
                                "Gene": (
                                    names_all[off + j] if (names_all and len(names_all) > off + j)
                                    else f"{ntype}_{j}"
                                ),
                                "NodeType": ntype
                            }
                            ev = emb[off + j].detach().cpu().numpy().astype(float).tolist()
                            for k, v in enumerate(ev, 1):
                                row[f"Emb_{k}"] = v
                            buckets[key].append(row)
                        off += cnt


    for key, rows in buckets.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        save_path = fr"{out_root}\fold_{fold + 1}_{key.upper()}_emb.csv"
        df.to_csv(save_path, index=False)
        print(f"✅ Saved {key.upper()} embeddings to: {save_path}")

# ==============================
# Model Factory
# ==============================
def create_model(model_type, in_dims, hidden_dim=128, out_dim=1, dropout=0.5):
    """Create different types of graph models with deterministic initialization"""
    # Ensure deterministic model creation
    torch.manual_seed(torch.initial_seed())

    if model_type == "sage":
        model = ImprovedGraphEmbeddingMLP(in_dims, hidden_dim, out_dim, dropout, num_gnn_layers=1)
    elif model_type == "gat":
        model = GATGraphEmbeddingMLP(in_dims, hidden_dim, out_dim, dropout, num_gat_layers=1)
    elif model_type == "gcn":
        model = GCNGraphEmbeddingMLP(in_dims, hidden_dim, out_dim, dropout, num_gcn_layers=1)
    elif model_type == "transformer":
        model = TransformerGraphEmbeddingMLP(in_dims, hidden_dim, out_dim, dropout, num_layers=2)
    elif model_type == "ensemble":
        model = EnsembleGraphModel(in_dims, hidden_dim, out_dim, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Apply deterministic weight initialization
    model.apply(lambda m: _init_weights_deterministic(m))
    return model


def _init_weights_deterministic(module):
    """Deterministic weight initialization"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def load_best_hyperparams(json_file=r'outputs\metric\best_params_cls_Histopathology_fold_1.json'):
    with open(json_file, 'r') as f:
        results = json.load(f)
    return results['best_params']


# ===========================
# 3. 使用固定超参数训练模型
# ===========================

def train_with_fixed_hyperparams():

    try:
        best_params = load_best_hyperparams(
            r'outputs\metric\Histopathology\best_params_cls_Histopathology_fold_5.json'
        )
    except Exception as e:
        print(f"[WARN] Failed to load best hyperparameters: {e}. Using default values.")
        best_params = dict(hidden_dim=64, dropout=0.5, lr=1e-4,
                           num_gnn_layers=1, weight_decay=1e-3, l2_lambda=1e-4)

    fixed_hidden_dim   = best_params['hidden_dim']
    fixed_dropout      = best_params['dropout']
    fixed_lr           = best_params['lr']
    fixed_num_gnn      = best_params['num_gnn_layers']
    fixed_weight_decay = best_params['weight_decay']
    fixed_l2_lambda    = best_params['l2_lambda']


    in_dims = {
        ntype: all_graphs[0].nodes[ntype].data["feat"].shape[1]
        for ntype in all_graphs[0].ntypes
    }


    dataset = GraphDataset(all_graphs, all_labels, all_serial_ids)
    y_vec   = np.array(all_labels, dtype=int)
    kf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


    pos_rate   = float(y_vec.mean()) if len(y_vec) else 0.5
    pos_weight = torch.tensor((1 - pos_rate) / max(pos_rate, 1e-6), dtype=torch.float32)
    print(f"[Fixed Train] pos_rate={pos_rate:.3f}, pos_weight={float(pos_weight):.3f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_auc_list = []


    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)), y_vec)):
        print(f"\n===== Fold {fold + 1} (fixed params) =====")
        set_seed(42 + fold)

        train_graphs_raw = [all_graphs[i] for i in train_idx]
        val_graphs_raw = [all_graphs[i] for i in val_idx]

        train_serial_ids = [all_serial_ids[i] for i in train_idx]
        val_serial_ids = [all_serial_ids[i] for i in val_idx]

        output_dir = fr"outputs\Histopathology\Histopathology_5\fold_{fold + 1}_gene_matrix"

        export_gene_matrix_by_omics(
            graphs=train_graphs_raw,
            serial_ids=train_serial_ids,
            output_dir=output_dir,
            split_name="train"
        )

        export_gene_matrix_by_omics(
            graphs=val_graphs_raw,
            serial_ids=val_serial_ids,
            output_dir=output_dir,
            split_name="test"
        )


        train_graphs, val_graphs, train_stats = normalize_graphs_with_train_stats(
            train_graphs_raw,
            val_graphs_raw,
            node_types=list(in_dims.keys())
        )

        train_labels = [all_labels[i] for i in train_idx]
        val_labels = [all_labels[i] for i in val_idx]
        train_serial_ids = [all_serial_ids[i] for i in train_idx]
        val_serial_ids = [all_serial_ids[i] for i in val_idx]

        train_dataset = GraphDataset(train_graphs, train_labels, train_serial_ids)
        val_dataset = GraphDataset(val_graphs, val_labels, val_serial_ids)

        model = ImprovedGraphEmbeddingMLP(
            in_dims, hidden_dim=fixed_hidden_dim, out_dim=NUM_CLASSES,
            dropout=fixed_dropout, num_gnn_layers=fixed_num_gnn
        )

        classes = np.arange(NUM_CLASSES)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_vec[train_idx])
        class_w = torch.tensor(weights, dtype=torch.float32)

        train_loader = DataLoader(train_dataset,
                                  batch_size=16, shuffle=False, collate_fn=collate_fn_multicls, num_workers=0)
        val_loader = DataLoader(val_dataset,
                                batch_size=16, shuffle=False, collate_fn=collate_fn_multicls, num_workers=0)

        best_auc = train_model_with_hyperparams_multicls(
            model, train_loader, val_loader,
            epochs=150, lr=fixed_lr, device=device, fold=fold,
            weight_decay=fixed_weight_decay, l2_lambda=fixed_l2_lambda,
            train_idx=train_idx, val_idx=val_idx,
            num_classes=NUM_CLASSES, class_weights=class_w
        )
        val_auc_list.append(best_auc)

        train_serial_ids_fold = [all_serial_ids[i] for i in train_idx]
        test_serial_ids_fold = [all_serial_ids[i] for i in val_idx]


    avg_auc = float(np.mean(val_auc_list)) if val_auc_list else float('nan')
    print(f"\n✅ Average AUC across folds: {avg_auc:.4f}")

def run_optuna_5fold_outer_multicls(n_trials=100, seed=42):
    print("Starting OUTER 5-fold Optuna optimization (multi-class classification)...")

    y_vec = np.array(all_labels, dtype=int)
    outer_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    all_fold_results = []

    for outer_fold, (train_idx, test_idx) in enumerate(outer_kf.split(np.arange(len(all_graphs)), y_vec)):
        print(f"\n{'='*80}")
        print(f"OUTER FOLD {outer_fold + 1}/5")
        print(f"{'='*80}")

        train_graphs_all = [all_graphs[i] for i in train_idx]
        train_labels_all = [all_labels[i] for i in train_idx]
        train_serial_ids_all = [all_serial_ids[i] for i in train_idx]

        test_graphs_raw = [all_graphs[i] for i in test_idx]
        test_labels = [all_labels[i] for i in test_idx]
        test_serial_ids = [all_serial_ids[i] for i in test_idx]

        study = run_optuna_optimization_multicls(
            train_graphs_all,
            train_labels_all,
            train_serial_ids_all,
            n_trials=n_trials,
            seed=seed + outer_fold
        )

        best_params = study.best_params

        fold_result = {
            "outer_fold": outer_fold + 1,
            "best_value": study.best_value,
            "best_params": best_params,
            "n_trials": len(study.trials),
            "train_size": len(train_idx),
            "test_size": len(test_idx),
        }
        all_fold_results.append(fold_result)

        out_json = fr'outputs\metric\best_params_cls_Histopathology_fold_{outer_fold + 1}.json'
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(fold_result, f, indent=2, ensure_ascii=False)
        print(f"✅ Fold {outer_fold + 1} best params saved to: {out_json}")


        in_dims = {
            ntype: train_graphs_all[0].nodes[ntype].data["feat"].shape[1]
            for ntype in train_graphs_all[0].ntypes
        }


        train_graphs_norm, test_graphs_norm, _ = normalize_graphs_with_train_stats(
            train_graphs_all,
            test_graphs_raw,
            node_types=list(in_dims.keys())
        )

        train_dataset = GraphDataset(train_graphs_norm, train_labels_all, train_serial_ids_all)
        test_dataset  = GraphDataset(test_graphs_norm, test_labels, test_serial_ids)


        classes = np.arange(NUM_CLASSES)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=np.array(train_labels_all))
        class_w = torch.tensor(weights, dtype=torch.float32)

        model = ImprovedGraphEmbeddingMLP(
            in_dims,
            hidden_dim=best_params["hidden_dim"],
            out_dim=NUM_CLASSES,
            dropout=best_params["dropout"],
            num_gnn_layers=best_params["num_gnn_layers"]
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True,
            collate_fn=collate_fn_multicls, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=16, shuffle=False,
            collate_fn=collate_fn_multicls, num_workers=0
        )

        best_metric = train_model_with_hyperparams_multicls(
            model, train_loader, test_loader,
            epochs=150,
            lr=best_params["lr"],
            device=device,
            fold=outer_fold,
            weight_decay=best_params["weight_decay"],
            l2_lambda=best_params["l2_lambda"],
            num_classes=NUM_CLASSES,
            class_weights=class_w
        )

        print(f"✅ OUTER FOLD {outer_fold + 1} done, best metric = {best_metric:.4f}")

    summary_json = r'outputs\metric\best_params_cls_Histopathology_5fold_all.json'
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(all_fold_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ All 5-fold best params saved to: {summary_json}")

    return all_fold_results
def export_gene_matrix_by_omics(graphs, serial_ids, output_dir, split_name):

    os.makedirs(output_dir, exist_ok=True)

    omics_dict = {
        "DNA": {},
        "RNA": {},
        "PROTEIN": {}
    }

    node_order_dict = {
        "DNA": [],
        "RNA": [],
        "PROTEIN": []
    }

    for g, sid in zip(graphs, serial_ids):
        for ntype in g.ntypes:
            omics_name = ntype.upper()
            if omics_name not in omics_dict:
                continue

            if "h" not in g.nodes[ntype].data:
                continue

            feat = g.nodes[ntype].data["h"].detach().cpu().numpy()


            if feat.ndim == 1:
                feat = feat[:, None]


            if feat.shape[1] > 1:
                print(f"{ntype} has {feat.shape[1]} feature dimensions; only the first dimension will be exported.")
            values = feat[:, 0]

            if len(node_order_dict[omics_name]) == 0:
                node_order_dict[omics_name] = list(range(len(values)))

            for idx, val in enumerate(values):
                if idx not in omics_dict[omics_name]:
                    omics_dict[omics_name][idx] = {}

                omics_dict[omics_name][idx][sid] = float(val)


    for omics_name in ["DNA", "RNA", "PROTEIN"]:
        if len(omics_dict[omics_name]) == 0:
            print(f" {split_name}_{omics_name}.csv no data，skip")
            continue

        node_ids = node_order_dict[omics_name]
        rows = []

        for row_idx, node_id in enumerate(node_ids):
            row = {
                "serial_id": row_idx + 1,
                "node_id": node_id
            }

            for sid in serial_ids:
                row[sid] = omics_dict[omics_name].get(node_id, {}).get(sid, np.nan)

            rows.append(row)

        patient_cols = list(serial_ids)
        df_out = pd.DataFrame(rows)

        ordered_cols = ["serial_id"] + patient_cols + ["node_id"]
        df_out = df_out[ordered_cols]

        save_path = os.path.join(output_dir, f"{split_name}_{omics_name}.csv")
        df_out.to_csv(save_path, index=False)
        print(f"✅ files are saved: {save_path}")

def train_with_best_params_on_train_and_eval_test(
    train_graphs_raw, train_labels, train_serial_ids,
    test_graphs_raw, test_labels, test_serial_ids,
    best_params, seed=42
):
    in_dims = {
        ntype: train_graphs_raw[0].nodes[ntype].data["feat"].shape[1]
        for ntype in train_graphs_raw[0].ntypes
    }

    y_train = np.array(train_labels, dtype=int)
    idx_all = np.arange(len(train_graphs_raw))

    inner_train_idx, inner_val_idx = train_test_split(
        idx_all,
        test_size=0.2,
        stratify=y_train,
        random_state=seed
    )

    inner_train_raw = [train_graphs_raw[i] for i in inner_train_idx]
    inner_val_raw   = [train_graphs_raw[i] for i in inner_val_idx]

    inner_train_graphs, inner_val_graphs, train_stats = normalize_graphs_with_train_stats(
        inner_train_raw, inner_val_raw, node_types=list(in_dims.keys())
    )

    test_graphs = []
    for g in test_graphs_raw:
        g_new = copy.deepcopy(g)
        for ntype in g_new.ntypes:
            if ntype in train_stats and "feat" in g_new.nodes[ntype].data:
                feat = g_new.nodes[ntype].data["feat"]
                feat = (feat - train_stats[ntype]['mean']) / train_stats[ntype]['std']
                g_new.nodes[ntype].data["feat"] = feat
        test_graphs.append(g_new)

    inner_train_labels = [train_labels[i] for i in inner_train_idx]
    inner_val_labels   = [train_labels[i] for i in inner_val_idx]
    inner_train_sids   = [train_serial_ids[i] for i in inner_train_idx]
    inner_val_sids     = [train_serial_ids[i] for i in inner_val_idx]

    train_dataset = GraphDataset(inner_train_graphs, inner_train_labels, inner_train_sids)
    val_dataset   = GraphDataset(inner_val_graphs, inner_val_labels, inner_val_sids)
    test_dataset  = GraphDataset(test_graphs, test_labels, test_serial_ids)

    classes = np.arange(NUM_CLASSES)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=np.array(inner_train_labels))
    class_w = torch.tensor(weights, dtype=torch.float32)

    model = ImprovedGraphEmbeddingMLP(
        in_dims,
        hidden_dim=best_params["hidden_dim"],
        out_dim=NUM_CLASSES,
        dropout=best_params["dropout"],
        num_gnn_layers=best_params["num_gnn_layers"]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_multicls, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_multicls, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_multicls, num_workers=0)

    train_model_with_hyperparams_multicls(
        model, train_loader, val_loader,
        epochs=150,
        lr=best_params["lr"],
        device=device,
        fold=0,
        weight_decay=best_params["weight_decay"],
        l2_lambda=best_params["l2_lambda"],
        num_classes=NUM_CLASSES,
        class_weights=class_w
    )

    # 最终 test 评估（只评一次）
    model.eval()
    te_logits, te_y = [], []
    with torch.no_grad():
        for g, ys, serial_ids in test_loader:
            g = g.to(device)
            logits = model(g)
            te_logits.append(logits.cpu())
            te_y.append(ys)

    te_logits = torch.cat(te_logits)
    te_y = torch.cat(te_y)
    te_prob = torch.softmax(te_logits, dim=1)
    te_pred = te_prob.argmax(dim=1)

    test_acc = accuracy_score(te_y, te_pred)
    test_f1 = f1_score(te_y, te_pred, average="macro", zero_division=0)
    try:
        test_auc = roc_auc_score(te_y, te_prob, multi_class="ovr", average="macro")
    except Exception:
        test_auc = float("nan")

    print(f"\\n✅ Final TEST | AUC {test_auc:.3f} F1 {test_f1:.3f} Acc {test_acc:.3f}")
# ===========================
# Choose Execution Mode
# ===========================

def main():
    use_optuna = False
    seed = 42

    if use_optuna:
        run_optuna_5fold_outer_multicls(n_trials=100, seed=seed)
    else:
        train_with_fixed_hyperparams()
# ==============================
# 5-Fold Cross Validation with Optuna Optimization
# ==============================
'''if __name__ == "__main__":
    print("Starting Optuna hyperparameter optimization...")

   
    study = optuna.create_study(
        direction='maximize',  # 目标是最大化 C-index
        sampler=optuna.samplers.TPESampler(seed=42),  # 使用 TPE sampler for better performance
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)  # Prune unpromising trials
    )

    
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    
    print("\n" + "=" * 50)
    print("OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Best C-index: {study.best_value:.4f}")
    print(f"Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    
    import json

    results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials)
    }

    with open(r'E:\conference\LGG\risk_csv\optuna_optimization_results_DFS_single_DNA.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to 'optuna_optimization_results_DFS_single_DNA.json'")

    
    try:
        optuna.visualization.plot_optimization_history(study).show()
        optuna.visualization.plot_param_importances(study).show()
        optuna.visualization.plot_parallel_coordinate(study).show()
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("You can still view the results in the saved JSON file.")'''

if __name__ == "__main__":
    main()