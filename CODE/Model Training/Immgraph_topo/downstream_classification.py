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
    """
    使用训练集统计量归一化训练集和测试集

    Args:
        train_graphs: 训练集图列表
        test_graphs: 测试集图列表
        node_types: 所有节点类型

    Returns:
        normalized_train_graphs, normalized_test_graphs, train_stats
    """
    # Step 1: 计算训练集的均值和标准差
    train_stats = {}
    for ntype in node_types:
        train_feats = []
        for g in train_graphs:
            if ntype in g.ntypes and "feat" in g.nodes[ntype].data:
                train_feats.append(g.nodes[ntype].data["feat"])

        if train_feats:
            all_train_feat = torch.cat(train_feats, dim=0)
            mean = all_train_feat.mean(dim=0, keepdim=True)
            std = all_train_feat.std(dim=0, keepdim=True) + 1e-8
            train_stats[ntype] = {'mean': mean, 'std': std}
            print(f"  Train stats for {ntype}: mean shape {mean.shape}, std range [{std.min():.4f}, {std.max():.4f}]")

    # Step 2: 归一化训练集
    normalized_train = []
    for g in train_graphs:
        g_new = g.clone()
        for ntype in g_new.ntypes:
            if ntype in train_stats and "feat" in g_new.nodes[ntype].data:
                feat = g_new.nodes[ntype].data["feat"]
                mean = train_stats[ntype]['mean']
                std = train_stats[ntype]['std']

                # 归一化
                feat_norm = (feat - mean) / std
                g_new.nodes[ntype].data["h"] = feat_norm
                g_new.nodes[ntype].data["feat"] = feat_norm
        normalized_train.append(g_new)

    # Step 3: 用训练集统计量归一化测试集
    normalized_test = []
    for g in test_graphs:
        g_new = g.clone()
        for ntype in g_new.ntypes:
            if ntype in train_stats and "feat" in g_new.nodes[ntype].data:
                feat = g_new.nodes[ntype].data["feat"]
                mean = train_stats[ntype]['mean']
                std = train_stats[ntype]['std']

                # 使用训练集统计量归一化测试集
                feat_norm = (feat - mean) / std
                g_new.nodes[ntype].data["h"] = feat_norm
                g_new.nodes[ntype].data["feat"] = feat_norm
        normalized_test.append(g_new)

    return normalized_train, normalized_test, train_stats
# ==============================
# Load Labels
# ==============================
print("Loading Excel labels...")
cls_xlsx = r"E:\Multi-omic Immunity\GCN_immune\scripts\LGG\downstream_ZZU\patient_classification_290.xlsx"
assert os.path.exists(cls_xlsx), f"Excel not found: {cls_xlsx}"

df = pd.read_excel(cls_xlsx, dtype={"serial_id": str})

# 1) 规范化ID，确保和图文件里解析出来的一致（大小写/空格）
df["serial_id"] = df["serial_id"].astype(str).str.strip().str.upper()

# 2) 将 TERT_mutation 转成 0/1（兼容 0/1、Yes/No、WT/Mutant 等写法）
def to01(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"1","mut","mutant","m","yes","y","true","pos","positive"}:
        return 1
    if s in {"0","wt","wildtype","w","no","n","false","neg","negative"}:
        return 0
    # 数字字符串：'0','1','0.0','1.0'
    try:
        return int(float(s))
    except:
        return np.nan

df["label"] = df["TERT_mutation"].apply(to01).astype("Int64")
df = df.dropna(subset=["label"]).copy()
df["label"] = df["label"].astype(int)

# 3) 建立 {serial_id -> 0/1} 的字典
y_map = dict(zip(df["serial_id"], df["label"]))
print(f"✅ Loaded {len(y_map)} labels from Excel (serial_id & TERT_mutation).")

# ==============================
# Load Graphs
# ==============================
print("Loading heterograph files...")
graph_dir = r"F:\subgraph_ZZU_new"
graph_paths = [f for f in os.listdir(graph_dir) if f.endswith(".dgl")]

# Step 1: scan max dim for each node type and compute global statistics
max_dims = defaultdict(int)
all_feats = defaultdict(list)


print("Computing global feature statistics...")
for fname in graph_paths:
    g = dgl.load_graphs(os.path.join(graph_dir, fname))[0][0]
    for ntype in g.ntypes:
        if "feat" in g.nodes[ntype].data:
            feat = g.nodes[ntype].data["feat"]
            if not torch.isfinite(feat).all():
                bad_idx = torch.isnan(feat) | torch.isinf(feat)
                print(f"[!] {fname}, node type {ntype}, NaN/Inf at {bad_idx.nonzero(as_tuple=True)}")
                feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            all_feats[ntype].append(feat)

print("Max feature dims per node type:", max_dims)


# Step 2: process graphs
all_graphs = []
all_labels = []
graph_paths = sorted(graph_paths)
all_serial_ids = []
all_filenames = []

for fname in graph_paths:
    if not fname.lower().endswith(".dgl"):
        continue

    base = os.path.splitext(fname)[0]           # e.g. patient_P_1319772_subgraph
    m = re.match(r"^patient_(.+?)_subgraph$", base, flags=re.IGNORECASE)
    if not m:
        print(f"跳过（命名不匹配）: {fname}")
        continue

    sid = m.group(1).upper()                    # 和 y_map 的键保持一致（都转大写）
    if sid not in y_map:
        print(f"⚠ 没有该ID的标签: {sid} (file={fname})，已跳过")
        continue

    graphs, _ = dgl.load_graphs(os.path.join(graph_dir, fname))
    if len(graphs) == 0:
        print("⚠️ Empty graph file:", fname)
        continue
    g = graphs[0]

    # —— 节点特征处理（直接用 feat）——
    for ntype in g.ntypes:
        if "h" not in g.nodes[ntype].data:
            # 如果没有 feat，则提供一个最小兜底，避免 KeyError
            g.nodes[ntype].data["h"] = torch.zeros(
                (g.num_nodes(ntype), max_dims.get(ntype, 1)),
                dtype=torch.float32
            )

        # 2) 读取/标准化/对齐维度（全部转成 float32）
        feat = torch.nan_to_num(g.nodes[ntype].data["h"], nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)

        cur_dim = feat.shape[1]
        target_dim = max_dims.get(ntype, cur_dim)
        if cur_dim < target_dim:
            pad = torch.zeros(feat.shape[0], target_dim - cur_dim, dtype=torch.float32)
            feat = torch.cat([feat, pad], dim=1)
        elif cur_dim > target_dim:
            feat = feat[:, :target_dim]

        g.nodes[ntype].data["h"] = feat

        # 3) 关键：只保留 'h'，删除其它键（'logits'、'feat'、中间态等）
        for k in list(g.nodes[ntype].data.keys()):
            if k != "h":
                del g.nodes[ntype].data[k]

        # 更新节点特征
        g.nodes[ntype].data["feat"] = feat

    # —— 边特征处理（当前不使用边权重，故跳过）——
    for etype in g.canonical_etypes:
        for key in list(g.edges[etype].data.keys()):
            if key != "_ID":
                del g.edges[etype].data[key]

    # ✅ 追加到数据列表
    all_graphs.append(g)
    all_labels.append(int(y_map[sid]))      # 单个 0/1 标签
    all_serial_ids.append(sid)              # 用规范化后的 ID
    all_filenames.append(fname)

print("数据集样本:")
for i in range(min(10, len(all_labels))):
    print(f"{i}: 患者ID={all_serial_ids[i]}, 图文件={all_filenames[i]}, y={all_labels[i]}")

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
        self.serial_ids = serial_ids  # 每个图的 serial_id
        self.augment = augment

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        label = self.labels[idx]
        serial_id = self.serial_ids[idx]  # 获取 serial_id

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
    ys = torch.tensor(labels, dtype=torch.float32)  # [B]
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
    def __init__(self, in_dims, hidden_dim=128, out_dim=1, dropout=0.5, num_gnn_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout

        # Add GNN layers
        self.gnn = HeteroSAGEConv(in_dims, hidden_dim, num_gnn_layers, dropout)

        # MLP for final prediction - Simplified architecture to reduce overfitting
        self.mlp = None  # 延迟初始化

    def forward(self, g):
        # Apply GNN layers first
        g = self.gnn(g)

        feats = []
        batch_num_nodes = []

        # ➤ 对每个图中的每种节点类型，分别提取其 GNN 输出特征，并记录数量
        for ntype in g.ntypes:
            x = g.nodes[ntype].data["h"]  # Use GNN output
            batch_num_nodes.append(g.batch_num_nodes(ntype))
            feats.append(torch.split(x, tuple(g.batch_num_nodes(ntype).cpu().numpy())))

        # ➤ 每个图拼成一个长向量
        num_graphs = g.batch_size
        graph_feats = []
        for i in range(num_graphs):
            node_vecs = [f[i].flatten(start_dim=0) for f in feats]  # [各类型节点展平拼接]
            graph_feat = torch.cat(node_vecs, dim=0)
            graph_feats.append(graph_feat)

        flat_feat = torch.stack(graph_feats, dim=0)  # ➤ [batch_size, total_feat_dim]

        # ➤ 初始化 MLP（只在第一次 forward 调用时执行）- Simplified architecture
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
    def __init__(self, in_dims, hidden_dim=128, out_dim=1, dropout=0.5, num_gat_layers=1):
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

        # ➤ 对每个图中的每种节点类型，分别提取其 GAT 输出特征，并记录数量
        for ntype in g.ntypes:
            x = g.nodes[ntype].data["h"]  # Use GAT output
            batch_num_nodes.append(g.batch_num_nodes(ntype))
            feats.append(torch.split(x, tuple(g.batch_num_nodes(ntype).cpu().numpy())))

        # ➤ 每个图拼成一个长向量
        num_graphs = g.batch_size
        graph_feats = []
        for i in range(num_graphs):
            node_vecs = [f[i].flatten(start_dim=0) for f in feats]  # [各类型节点展平拼接]
            graph_feat = torch.cat(node_vecs, dim=0)
            graph_feats.append(graph_feat)

        flat_feat = torch.stack(graph_feats, dim=0)  # ➤ [batch_size, total_feat_dim]

        # ➤ 初始化 MLP（只在第一次 forward 调用时执行）
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

        # ➤ 对每个图中的每种节点类型，分别提取其 GCN 输出特征，并记录数量
        for ntype in g.ntypes:
            x = g.nodes[ntype].data["h"]  # Use GCN output
            batch_num_nodes.append(g.batch_num_nodes(ntype))
            feats.append(torch.split(x, tuple(g.batch_num_nodes(ntype).cpu().numpy())))

        # ➤ 每个图拼成一个长向量
        num_graphs = g.batch_size
        graph_feats = []
        for i in range(num_graphs):
            node_vecs = [f[i].flatten(start_dim=0) for f in feats]  # [各类型节点展平拼接]
            graph_feat = torch.cat(node_vecs, dim=0)
            graph_feats.append(graph_feat)

        flat_feat = torch.stack(graph_feats, dim=0)  # ➤ [batch_size, total_feat_dim]

        # ➤ 初始化 MLP（只在第一次 forward 调用时执行）
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
        self.mlp = None  # 延迟初始化

    def forward(self, g):
        # Apply Transformer layers first
        g = self.transformer(g)

        feats = []
        batch_num_nodes = []

        # ➤ 对每个图中的每种节点类型，分别提取其 Transformer 输出特征，并记录数量
        for ntype in g.ntypes:
            x = g.nodes[ntype].data["h"]  # Use Transformer output
            batch_num_nodes.append(g.batch_num_nodes(ntype))
            feats.append(torch.split(x, tuple(g.batch_num_nodes(ntype).cpu().numpy())))

        # ➤ 每个图拼成一个长向量
        num_graphs = g.batch_size
        graph_feats = []
        for i in range(num_graphs):
            node_vecs = [f[i].flatten(start_dim=0) for f in feats]  # [各类型节点展平拼接]
            graph_feat = torch.cat(node_vecs, dim=0)
            graph_feats.append(graph_feat)

        flat_feat = torch.stack(graph_feats, dim=0)  # ➤ [batch_size, total_feat_dim]

        # ➤ 初始化 MLP（只在第一次 forward 调用时执行）
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

    # 触发一次 forward 构建 MLP
    with torch.no_grad():
        g0, y0, _ = next(iter(train_loader))
        _ = model(g0.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce_loss  = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # ====== 早停 & 最佳轮缓存 ======
    best_val_auc   = -float("inf")
    best_state     = None          # 最佳轮的模型权重
    best_records   = None          # 最佳轮的逐样本记录(Serial_ID, y_true, y_prob)
    best_val_prob  = None          # 最佳轮概率（张量）
    best_val_y     = None          # 最佳轮标签（张量）
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

        # ====== 如果更好就缓存这一轮 ======
        if val_auc > best_val_auc + 1e-6:
            best_val_auc  = val_auc
            patience      = 0
            best_epoch    = epoch + 1
            # 深拷贝权重到 CPU，便于之后回滚
            best_state    = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_records  = list(cur_records)
            best_val_prob = val_prob.detach().cpu().clone()
            best_val_y    = val_y.detach().cpu().clone()
        else:
            patience += 1
            if patience >= early_stop:
                print("Early stopping!")
                break

    # ====== 回滚到最佳权重，并基于“最佳轮”写CSV ======
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        # 极端情况下没更新过 best_state，就用最后一轮
        best_records  = cur_records
        best_val_prob = val_prob.detach().cpu()
        best_val_y    = val_y.detach().cpu()

    model.eval()
    train_records = []
    with torch.no_grad():
        for g, ys, serial_ids in train_loader:
            g = g.to(device)
            logit = model(g).view(-1)
            prob = torch.sigmoid(logit).cpu().numpy().tolist()

            for sid, yy, pp in zip(serial_ids, ys.numpy().tolist(), prob):
                train_records.append({
                    "Serial_ID": sid,
                    "y_true": int(yy),
                    "y_prob": float(pp)
                })

    # 计算训练集的最佳F1阈值(可以用验证集的阈值,或单独计算)
    train_y_arr = np.array([r["y_true"] for r in train_records])
    train_prob_arr = np.array([r["y_prob"] for r in train_records])

    try:
        precision, recall, thresholds = precision_recall_curve(train_y_arr, train_prob_arr)
        f1s = 2 * precision * recall / (precision + recall + 1e-12)
        train_best_thr = thresholds[np.nanargmax(f1s)] if len(thresholds) > 0 else 0.5
    except:
        train_best_thr = best_thr  # 使用验证集阈值

    # 添加预测标签
    for r in train_records:
        r["pred_label@0.5"] = int(r["y_prob"] >= 0.5)
        r["pred_label@bestF1"] = int(r["y_prob"] >= float(train_best_thr))
        r["bestF1_threshold"] = float(train_best_thr)
        r["best_epoch"] = int(best_epoch)

    # 保存训练集预测到CSV
    train_csv = fr"E:\Multi-omic Immunity\GCN_immune\scripts\LGG\downstream_ZZU\outputs\fold_{fold + 1}_train_pred_TERT_mutation.csv"
    pd.DataFrame(train_records).to_csv(train_csv, index=False)
    print(f"✅ Saved BEST-epoch training predictions to {train_csv}")

    # 计算最佳轮的 bestF1 阈值
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

    os.makedirs(r"E:\Multi-omic Immunity\GCN_immune\scripts\LGG\downstream_ZZU\outputs", exist_ok=True)
    out_csv = fr"E:\Multi-omic Immunity\GCN_immune\scripts\LGG\downstream_ZZU\outputs\fold_{fold+1}_val_pred_TERT_mutation.csv"
    pd.DataFrame(best_records).to_csv(out_csv, index=False)
    print(f"✅ Saved BEST-epoch (epoch={best_epoch}) validation predictions to {out_csv}")

    return float(best_val_auc)

# ==============================
# Optuna Objective Function for Hyperparameter Optimization
# ==============================
def run_optuna_optimization_cls(n_trials=300, seed=42):
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
    out_json = r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\downstream_ZZU\outputs\best_params_cls_TERT_mutation_1.json'
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Optuna results saved to: {out_json}")
    return study

def calculate_metrics(true_values, predicted_values):
    # 将输入转换为numpy数组
    true_values = np.array(true_values).flatten()
    predicted_values = np.array(predicted_values).flatten()

    # 计算各项指标
    try:
        # MAPE (平均绝对百分比误差)
        mape = mean_absolute_percentage_error(true_values, predicted_values) * 100
    except:
        mape = np.nan

    # MedAE (中位数绝对误差)
    medae = median_absolute_error(true_values, predicted_values)

    # EVS (解释方差分数)
    evs = explained_variance_score(true_values, predicted_values)

    # ME (最大误差)
    me = max_error(true_values, predicted_values)

    try:
        # LogMSE (对数均方误差)
        log_mse = np.mean(np.square(np.log1p(true_values) - np.log1p(predicted_values)))
    except:
        log_mse = np.nan

    try:
        # SMAPE (对称平均绝对百分比误差)
        smape = 100 * np.mean(2 * np.abs(predicted_values - true_values) /
                              (np.abs(predicted_values) + np.abs(true_values)))
    except:
        smape = np.nan

    # MBE (平均偏差误差)
    mbe = np.mean(predicted_values - true_values)

    # NMSE (标准化均方误差)
    if np.var(true_values) != 0:
        nmse = np.mean(np.square(predicted_values - true_values)) / np.var(true_values)
    else:
        nmse = np.nan

    # RAE (相对绝对误差)
    denominator = np.sum(np.abs(true_values - np.mean(true_values)))
    if denominator != 0:
        rae = np.sum(np.abs(predicted_values - true_values)) / denominator
    else:
        rae = np.nan

    # RSE (相对平方误差)
    denominator = np.sum(np.square(true_values - np.mean(true_values)))
    if denominator != 0:
        rse = np.sum(np.square(predicted_values - true_values)) / denominator
    else:
        rse = np.nan

    try:
        # Poisson_Deviance (泊松偏差)
        safe_true = np.maximum(true_values, 1e-10)
        safe_pred = np.maximum(predicted_values, 1e-10)
        poisson_deviance = 2 * np.sum(safe_true * np.log(safe_true / safe_pred) - (safe_true - safe_pred))
    except:
        poisson_deviance = np.nan

    try:
        # Gamma_Deviance (伽马偏差)
        safe_true = np.maximum(true_values, 1e-10)
        safe_pred = np.maximum(predicted_values, 1e-10)
        gamma_deviance = 2 * np.sum(np.log(safe_pred / safe_true) + (safe_true / safe_pred) - 1)
    except:
        gamma_deviance = np.nan

    # 计算相关系数
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
        # MGE (平均几何误差)
        safe_true = np.maximum(true_values, 1e-10)
        safe_pred = np.maximum(predicted_values, 1e-10)
        mge = np.exp(np.mean(np.abs(np.log(safe_pred / safe_true))))
    except:
        mge = np.nan

    try:
        # MSLE (均方对数误差)
        msle = np.mean(np.square(np.log1p(predicted_values) - np.log1p(true_values)))
        # RMSLE (均方根对数误差)
        rmsle = np.sqrt(msle)
    except:
        msle = np.nan
        rmsle = np.nan

    # WAE (加权绝对误差)
    weights = np.ones_like(true_values)  # 使用均匀权重
    wae = np.average(np.abs(predicted_values - true_values), weights=weights)

    # Direction_Accuracy (方向准确率)
    if len(true_values) > 1:
        try:
            direction_accuracy = np.mean((np.diff(true_values) * np.diff(predicted_values)) > 0)
        except:
            direction_accuracy = np.nan
    else:
        direction_accuracy = np.nan

    return mape, medae, evs, me, log_mse, smape, mbe, nmse, rae, rse, poisson_deviance, gamma_deviance, pearson_corr, spearman_corr, kendall_corr, mge, msle, rmsle, wae, direction_accuracy


# 保存每个fold的结果
def save_metrics_to_csv(metrics, file_path):
    # 将指标列表转换为DataFrame，使用正确的列名
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
    """
    从 g.nodes[ntype].data[...] 里尽量提取基因名，失败时返回 None。
    支持 torch.Tensor/list/ndarray，尽量鲁棒转换为 str 列表。
    """
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
    """
    GAT 常见形状 (N, heads, dim)，这里统一展平为 (N, -1)；
    其他情况保持二维。
    """
    if x is None:
        return None
    if x.dim() == 3:
        return x.reshape(x.shape[0], -1)
    return x

def export_node_embeddings_by_omics(model, loaders, fold, out_root):
    """
    在最后一个 epoch 结束后调用本函数：
    - 遍历传入的 DataLoader（可传 [train_loader, val_loader]）
    - 对每个 batch 做一次前向（eval/no_grad），读取 g.nodes[ntype].data["h"]
    - 仅导出 "h"（即经过 SAGE/GAT/GCN/Transformer 后的节点嵌入）
    - 分别按 DNA/RNA/PROTEIN 三类写成三个 CSV（若某类为空则跳过）
    CSV 列：
      Serial_ID, Gene, NodeType, Emb_1...Emb_D
    """
    os.makedirs(out_root, exist_ok=True)
    device = next(model.parameters()).device

    buckets = {"dna": [], "rna": [], "protein": []}

    model.eval()
    with torch.no_grad():
        for loader in loaders:
            for g, _, _, serial_ids in loader:
                g = g.to(device)
                # 前向一次，确保 "h" 是最后一轮的表示
                _ = model(g)

                for ntype in g.ntypes:
                    key = ntype.lower()
                    if key not in buckets:
                        continue  # 只导 DNA/RNA/PROTEIN

                    # 取嵌入 h
                    emb = g.nodes[ntype].data.get("h", None)
                    emb = _ensure_2d(emb)
                    if emb is None:
                        continue

                    # 基因名
                    names_all = _extract_node_names(g, ntype)

                    # 将本 batch 的该 ntype，按图拆分
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

    # 写出 3 份 CSV
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


def load_best_hyperparams(json_file=r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\downstream_ZZU\outputs\best_params_cls_TERT_mutation_1.json'):
    """从JSON文件中加载最佳超参数"""
    with open(json_file, 'r') as f:
        results = json.load(f)
    return results['best_params']


# ===========================
# 3. 使用固定超参数训练模型
# ===========================

def train_with_fixed_hyperparams():
    # 1) 读取最佳超参 JSON（路径要和你保存的一致）
    try:
        best_params = load_best_hyperparams(
            r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\downstream_ZZU\outputs\best_params_cls_TERT_mutation.json'
        )
    except Exception as e:
        print(f"[WARN] 读取最佳超参失败：{e}，将使用默认值")
        best_params = dict(hidden_dim=64, dropout=0.5, lr=1e-4,
                           num_gnn_layers=1, weight_decay=1e-3, l2_lambda=1e-4)

    fixed_hidden_dim   = best_params['hidden_dim']
    fixed_dropout      = best_params['dropout']
    fixed_lr           = best_params['lr']
    fixed_num_gnn      = best_params['num_gnn_layers']
    fixed_weight_decay = best_params['weight_decay']
    fixed_l2_lambda    = best_params['l2_lambda']

    # 2) 输入维度（注意：不要用未定义的 g；直接从 all_graphs[0] 读）
    in_dims = {
        ntype: all_graphs[0].nodes[ntype].data["feat"].shape[1]
        for ntype in all_graphs[0].ntypes
    }

    # 3) 数据与分层K折
    dataset = GraphDataset(all_graphs, all_labels, all_serial_ids)
    y_vec   = np.array(all_labels, dtype=int)
    kf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 4) 类别不平衡权重（全局一次）
    pos_rate   = float(y_vec.mean()) if len(y_vec) else 0.5
    pos_weight = torch.tensor((1 - pos_rate) / max(pos_rate, 1e-6), dtype=torch.float32)
    print(f"[Fixed Train] pos_rate={pos_rate:.3f}, pos_weight={float(pos_weight):.3f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_auc_list = []

    # 5) 5折训练与验证
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)), y_vec)):
        print(f"\n===== Fold {fold+1} (fixed params) =====")
        set_seed(42 + fold)

        # 划分训练集和测试集（原始特征）
        train_graphs_raw = [all_graphs[i] for i in train_idx]
        test_graphs_raw = [all_graphs[i] for i in val_idx]
        train_labels = [all_labels[i] for i in train_idx]
        test_labels = [all_labels[i] for i in val_idx]

        # **关键：使用训练集统计量归一化**
        print("Normalizing with train statistics...")
        train_graphs, test_graphs, train_stats = normalize_graphs_with_train_stats(
            train_graphs_raw,
            test_graphs_raw,
            node_types=list(in_dims.keys())  # 使用 in_dims 的键
        )

        # 计算类别权重（基于训练集）
        pos_rate = np.mean(train_labels)
        pos_weight_fold = None
        if 0 < pos_rate < 1:
            pos_weight_fold = torch.tensor((1 - pos_rate) / max(pos_rate, 1e-6), dtype=torch.float32).to(device)
        print(
            f"Train pos_rate={pos_rate:.3f}, pos_weight={float(pos_weight_fold) if pos_weight_fold is not None else None}")
        # ====== 归一化代码结束 ======

        model = ImprovedGraphEmbeddingMLP(
            in_dims, hidden_dim=fixed_hidden_dim, out_dim=1,
            dropout=fixed_dropout, num_gnn_layers=fixed_num_gnn
        )

        train_loader = DataLoader(list(zip(train_graphs, train_labels, [all_serial_ids[i] for i in train_idx])),
                                  batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_loader   = DataLoader(list(zip(test_graphs, test_labels, [all_serial_ids[i] for i in val_idx])),
                                  batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0)

        best_auc = train_model_with_hyperparams_cls(
            model, train_loader, val_loader,
            epochs=150, lr=fixed_lr, device=device, fold=fold,
            weight_decay=fixed_weight_decay, l2_lambda=fixed_l2_lambda,
            train_idx=train_idx, val_idx=val_idx, pos_weight=pos_weight_fold
        )
        val_auc_list.append(best_auc)

    # 6) 汇总
    avg_auc = float(np.mean(val_auc_list)) if val_auc_list else float('nan')
    print(f"\n✅ Average AUC across folds: {avg_auc:.4f}")


# ===========================
# 4. 主程序部分，选择运行模式
# ===========================

def main():
    use_optuna = False  # 设置为True以运行贝叶斯调参，False以加载调优后的超参数进行训练

    if use_optuna:
        # 运行Optuna进行贝叶斯调参
        run_optuna_optimization_cls()
    else:
        # 使用之前调优后的超参数进行训练
        train_with_fixed_hyperparams()
# ==============================
# 5-Fold Cross Validation with Optuna Optimization
# ==============================
'''if __name__ == "__main__":
    print("Starting Optuna hyperparameter optimization...")

    # 使用 Optuna 进行贝叶斯优化
    study = optuna.create_study(
        direction='maximize',  # 目标是最大化 C-index
        sampler=optuna.samplers.TPESampler(seed=42),  # 使用 TPE sampler for better performance
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)  # Prune unpromising trials
    )

    # 进行超参数优化
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # 输出最佳的超参数和结果
    print("\n" + "=" * 50)
    print("OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Best C-index: {study.best_value:.4f}")
    print(f"Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # 保存优化结果
    import json

    results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials)
    }

    with open(r'E:\conference\LGG\risk_csv\optuna_optimization_results_DFS_single_DNA.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to 'optuna_optimization_results_DFS_single_DNA.json'")

    # 可视化优化过程
    try:
        optuna.visualization.plot_optimization_history(study).show()
        optuna.visualization.plot_param_importances(study).show()
        optuna.visualization.plot_parallel_coordinate(study).show()
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("You can still view the results in the saved JSON file.")'''

if __name__ == "__main__":
    main()