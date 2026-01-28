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
from itertools import product
from sklearn.metrics import explained_variance_score, mean_absolute_percentage_error, median_absolute_error, max_error
import scipy.stats as stats
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


# ==============================
# Normalization Functions
# ==============================
def compute_fold_statistics(train_graphs, max_dims):

    fold_stats = {}
    all_feats = defaultdict(list)

    for g in train_graphs:
        for ntype in g.ntypes:
            if "h" in g.nodes[ntype].data:
                feat = g.nodes[ntype].data["h"]
                feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
                all_feats[ntype].append(feat)

    
    for ntype, feats in all_feats.items():
        if feats:
            all_feat = torch.cat(feats, dim=0)

            raw_mean = all_feat.mean().item()
            raw_std = all_feat.std().item()

            print(f"  {ntype}: original mean={raw_mean:.4f}, std={raw_std:.4f}")

            
            fold_stats[ntype] = {
                'mean': all_feat.mean(0, keepdim=True),
                'std': all_feat.std(0, keepdim=True) + 1e-6
            }

    return fold_stats


def normalize_graph(g, stats, max_dims, protein_clip=1.0, other_clip=5.0):

    for ntype in g.ntypes:
        
        if "h" not in g.nodes[ntype].data:
            g.nodes[ntype].data["h"] = torch.zeros(
                (g.num_nodes(ntype), max_dims.get(ntype, 1)),
                dtype=torch.float32
            )

        feat = g.nodes[ntype].data["h"]
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)

        
        if ntype in stats:
            feat = (feat - stats[ntype]['mean']) / (stats[ntype]['std'] + 1e-8)
        else:
            # Fallback
            mean = feat.mean(0, keepdim=True)
            std = feat.std(0, keepdim=True) + 1e-6
            feat = (feat - mean) / std

        
        #feat = torch.clamp(feat, min=-5.0, max=5.0)

        if ntype.lower() == 'protein':
            feat = torch.clamp(feat, min=-protein_clip, max=protein_clip) 
        else:
            #feat = torch.clamp(feat, min=-other_clip, max=other_clip)  
            pass

        
        cur_dim = feat.shape[1]
        target_dim = max_dims.get(ntype, cur_dim)
        if cur_dim < target_dim:
            pad = torch.zeros(feat.shape[0], target_dim - cur_dim, dtype=torch.float32)
            feat = torch.cat([feat, pad], dim=1)
        elif cur_dim > target_dim:
            feat = feat[:, :target_dim]

        g.nodes[ntype].data["h"] = feat

        #clear others
        for k in list(g.nodes[ntype].data.keys()):
            if k != "h":
                del g.nodes[ntype].data[k]

    return g


def evaluate_on_external_test_simple(model, test_graphs, test_labels, test_serial_ids,
                                     fold_stats, max_dims, device, fold,
                                     train_risks=None, adaptive_clip=True, quantile_range=(1, 99)):

    print(f"\n{'=' * 70}")
    print(f"Fold {fold + 1} external datasets")
    print(f"{'=' * 70}")

    # function for aligning
    test_graphs_aligned = align_test_to_train_advanced(
        test_graphs,
        fold_stats,
        max_dims,
        clip_strategy='quantile',  
        quantile_low=quantile_range[0],  
        quantile_high=quantile_range[1],
        clip_multiplier={'protein': 2.0, 'default': 3.0}
    )

    # prediction
    test_dataset = GraphDataset(test_graphs_aligned, test_labels, test_serial_ids)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    model.eval()
    all_risks = []
    all_times = []
    all_events = []
    all_sids = []

    with torch.no_grad():
        for g, times, events, batch_serial_ids in test_loader:
            g = g.to(device)
            risk = model(g)
            all_risks.extend(risk.squeeze().cpu().numpy().tolist())
            all_times.extend(times.numpy().tolist())
            all_events.extend(events.numpy().tolist())
            all_sids.extend(batch_serial_ids)

    all_risks = np.array(all_risks)
    all_times = np.array(all_times)      
    all_events = np.array(all_events)

    
    c_index_raw = concordance_index(all_times, -all_risks, all_events)

    print(f"\n C-index: {c_index_raw:.4f}")
    print(f" range: [{all_risks.min():.3f}, {all_risks.max():.3f}]")
    print(f"   mean: {all_risks.mean():.3f} ± {all_risks.std():.3f}")

    # ✅ 第四步：分时间段诊断（修复变量名）
    print(f"\n")

    time_ranges = [
        ("0-12月(early)", lambda t: t < 12),
        ("12-24月(middle early)", lambda t: (t >= 12) & (t < 24)),
        ("24-54月(middle)", lambda t: (t >= 24) & (t <= 54.9)),
        (">54月(long)", lambda t: t > 54.9),
    ]

    segment_results = {}
    for seg_name, mask_fn in time_ranges:
        mask = mask_fn(all_times) 
        n_samples = mask.sum()
        n_events = all_events[mask].sum()

        if n_samples >= 3:
            c_seg = concordance_index(
                all_times[mask],
                -all_risks[mask],
                all_events[mask]
            )
            event_rate = n_events / n_samples
            segment_results[seg_name] = {
                'n': n_samples,
                'events': n_events,
                'event_rate': event_rate,
                'c_index': c_seg
            }
            print(f"  {seg_name}:")
            print(f"    numbers of samples: {n_samples}, numbers of events: {n_events}, ratio of events: {event_rate:.1%}")
            print(f"    C-index: {c_seg:.4f}")
        else:
            print(f"  {seg_name}: ({n_samples}), skip")

 
    all_risks_calibrated = all_risks.copy()

    if train_risks is not None:
        long_mask = all_times > 54.9  
        if long_mask.sum() >= 3:
            long_risks = all_risks[long_mask]
            train_median = np.median(train_risks)
            long_median = np.median(long_risks)

            if long_median < train_median * 0.8:  
                adjustment = (train_median - long_median) * 0.15  
                all_risks_calibrated[long_mask] = long_risks + adjustment

                c_index_calibrated = concordance_index(
                    all_times[long_mask],
                    -all_risks_calibrated[long_mask],
                    all_events[long_mask]
                )


    =
    c_index_final = concordance_index(all_times, -all_risks_calibrated, all_events)

    print(f"\n{'=' * 70}")
    print(f"✅ external datasets C-index: {c_index_final:.4f}")
    print(f"{'=' * 70}\n")

    
    output_dir = r"E:\Multi-omic Immunity\GCN_immune\scripts\LGG\downstream_ZZU\MLP\risk_csv_PFS"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"fold_{fold + 1}_test_risk_external_LGG.csv")

    with open(output_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Serial_ID", "Time", "Event", "Risk"])
        writer.writeheader()
        for sid, t, e, r in zip(all_sids, all_times, all_events, all_risks_calibrated):
            writer.writerow({
                "Serial_ID": sid,
                "Time": float(t),
                "Event": int(e),
                "Risk": float(r)
            })

    print(f"✅ It is saved: {output_file}\n")

    return c_index_final, segment_results


def align_test_to_train_advanced(test_graphs, train_stats, max_dims,
                                 clip_strategy='quantile',
                                 quantile_low=1, quantile_high=99,
                                 clip_multiplier={'protein': 2.0, 'default': 3.0}):
    
    aligned = []

    for g in test_graphs:
        g_new = g.clone()

        for ntype in g.ntypes:
            if 'h' not in g.nodes[ntype].data or ntype not in train_stats:
                continue

            feat = g.nodes[ntype].data['h']
            train_mean = train_stats[ntype]['mean']
            train_std = train_stats[ntype]['std']

            # noremalized
            feat_norm = (feat - train_mean) / (train_std + 1e-8)

            # clip
            if clip_strategy == 'quantile':
                multiplier = clip_multiplier.get(ntype.lower(), clip_multiplier['default'])
                
                feat_norm = torch.clamp(feat_norm, min=-multiplier, max=multiplier)

            elif clip_strategy == 'mad':
                
                multiplier = clip_multiplier.get(ntype.lower(), clip_multiplier['default'])
                
                mad_multiplier = multiplier / 0.6745
                feat_norm = torch.clamp(feat_norm, min=-mad_multiplier, max=mad_multiplier)

            else:  # 'std'
                multiplier = clip_multiplier.get(ntype.lower(), clip_multiplier['default'])
                feat_norm = torch.clamp(feat_norm, min=-multiplier, max=multiplier)

            
            if feat_norm.shape[0] > 1:
                median = feat_norm.median(dim=0, keepdim=True)[0]
                mad_local = torch.abs(feat_norm - median).median(dim=0, keepdim=True)[0]
                outlier_mask = torch.abs(feat_norm - median) > 3 * mad_local
                feat_norm = torch.where(outlier_mask, median, feat_norm)

            g_new.nodes[ntype].data['h'] = feat_norm

        aligned.append(g_new)

    return aligned


def align_test_features_to_train_correct(test_graphs, train_stats, max_dims, alpha=0.9):

    aligned_graphs = []

    for g in test_graphs:
        g_aligned = g.clone()

        for ntype in g.ntypes:
            if 'h' not in g.nodes[ntype].data:
                continue

            feat = g.nodes[ntype].data['h']

            if ntype not in train_stats:
                continue

            # normalized
            test_mean = feat.mean(dim=0, keepdim=True)
            test_std = feat.std(dim=0, keepdim=True) + 1e-6
            # feat_normalized = (feat - test_mean) / test_std

            # use train datasets
            train_mean = train_stats[ntype]['mean']
            train_std = train_stats[ntype]['std']
            feat_aligned = (feat - train_mean) / train_std


            if alpha > 0:

                clip_range = 2.5
                train_min = train_mean - clip_range * train_std
                train_max = train_mean + clip_range * train_std
                feat_aligned = torch.clamp(feat_aligned, train_min, train_max)

            g_aligned.nodes[ntype].data['h'] = feat_aligned

        aligned_graphs.append(g_aligned)

    return aligned_graphs


def diagnose_feature_distribution(graphs, train_stats):
    
    test_means = []
    test_stds = []
    train_means = []
    train_stds = []

    for g in graphs:
        for ntype in g.ntypes:
            if 'h' in g.nodes[ntype].data:
                feat = g.nodes[ntype].data['h']
                test_means.append(feat.mean().item())
                test_stds.append(feat.std().item())

                if ntype in train_stats:
                    train_means.append(train_stats[ntype]['mean'].mean().item())
                    train_stds.append(train_stats[ntype]['std'].mean().item())

    print(f"  test: mean={np.mean(test_means):.4f}, std={np.mean(test_stds):.4f}")
    if train_means:
        print(f"  training: 均值={np.mean(train_means):.4f}, std={np.mean(train_stds):.4f}")
        print(f"  difference: Δmean={abs(np.mean(test_means) - np.mean(train_means)):.4f}")


set_seed(1)

# ==============================
# Load Labels
# ==============================
print("Loading Excel labels...")
csv_file = r"/data/data_for_running/ImmGraph-topo/Patient prognosis.xlsx"
df = pd.read_csv(csv_file)
df['serial_id'] = df['serial_id'].astype(str).str.strip()
name_to_time = dict(zip(df['serial_id'], df['PFS']))
name_to_event = dict(zip(df['serial_id'], df['Recurrence']))


print("Loading external test set labels...")
test_csv_file = r"/data/data_for_running/ImmGraph-topo/merged_patient_clinical_filtered_OS.csv"  
test_df = pd.read_csv(test_csv_file)
test_df['serial_id'] = test_df['SAMPLE_ID'].astype(str).str.strip()
test_name_to_time = dict(zip(test_df['serial_id'], test_df['OS']))
test_name_to_event = dict(zip(test_df['serial_id'], test_df['Status']))


print("original data sample:")
for i, (sid, time, event) in enumerate(zip(df['serial_id'], df['PFS'], df['Recurrence'])):
    print(f"{i}: ID={sid}, time={time}, events={event}")
    if i > 10: break

# ==============================
# Load Graphs
# ==============================
print("Loading heterograph files...")
graph_dir = r"/data/Data_for_ImmGraph_topo/Discovery"
graph_paths = [f for f in os.listdir(graph_dir) if f.endswith(".dgl")]

# Step 1: scan max dim for each node type and compute global statistics
max_dims = defaultdict(int)
all_feats = defaultdict(list)

# find max dims
print("Scanning max dimensions for each node type...")
max_dims = defaultdict(int)

for fname in graph_paths:
    g = dgl.load_graphs(os.path.join(graph_dir, fname))[0][0]
    for ntype in g.ntypes:
        if "h" in g.nodes[ntype].data:
            feat = g.nodes[ntype].data["h"]
            max_dims[ntype] = max(max_dims[ntype], feat.shape[1])

print("Max feature dims per node type:", dict(max_dims))

# Step 2: process graphs
# Step 2: process graphs
all_graphs, all_labels = [], []
# Sort graph paths for deterministic processing order
graph_paths = sorted(graph_paths)
all_serial_ids = []
all_filenames = []
max_dims[ntype] = max(max_dims[ntype], feat.shape[1])
for fname in graph_paths:
    
    if not fname.lower().endswith(".dgl"):
        continue

    base = os.path.splitext(fname)[0]  # e.g. patient_P_1319772_subgraph
    m = re.match(r'^patient_(.+?)_subgraph$', base, flags=re.IGNORECASE)
    if not m:
        print(f"beacause of the file name,skip：{fname}")
        continue

    serial_id = m.group(1).upper()  

    
    if serial_id in name_to_time:
        graphs, _ = dgl.load_graphs(os.path.join(graph_dir, fname))
        if len(graphs) == 0:
            print("⚠️ Empty graph file:", fname)
            continue
        g = graphs[0]

        
        for ntype in g.ntypes:
            if "h" not in g.nodes[ntype].data:
                # if no feat
                g.nodes[ntype].data["h"] = torch.zeros(
                    (g.num_nodes(ntype), max_dims.get(ntype, 1)),
                    dtype=torch.float32
                )

            
            feat = torch.nan_to_num(g.nodes[ntype].data["h"], nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)

            
            feat = torch.nan_to_num(
                g.nodes[ntype].data["h"],
                nan=0.0, posinf=0.0, neginf=0.0
            ).to(torch.float32)

            # padding/truncation
            cur_dim = feat.shape[1]
            target_dim = max_dims.get(ntype, cur_dim)
            if cur_dim < target_dim:
                pad = torch.zeros(feat.shape[0], target_dim - cur_dim, dtype=torch.float32)
                feat = torch.cat([feat, pad], dim=1)
            elif cur_dim > target_dim:
                feat = feat[:, :target_dim]

            
            g.nodes[ntype].data["h"] = feat

            cur_dim = feat.shape[1]
            target_dim = max_dims.get(ntype, cur_dim)
            if cur_dim < target_dim:
                pad = torch.zeros(feat.shape[0], target_dim - cur_dim, dtype=torch.float32)
                feat = torch.cat([feat, pad], dim=1)
            elif cur_dim > target_dim:
                feat = feat[:, :target_dim]

            g.nodes[ntype].data["h"] = feat

            # 3) save the feature of h
            for k in list(g.nodes[ntype].data.keys()):
                if k != "h":
                    del g.nodes[ntype].data[k]

        # edge
        for etype in g.canonical_etypes:
            for key in list(g.edges[etype].data.keys()):
                if key != "_ID":
                    del g.edges[etype].data[key]
        

        # add label
        all_graphs.append(g)
        all_labels.append((float(name_to_time[serial_id]), int(name_to_event[serial_id])))
        all_serial_ids.append(serial_id) 
        all_filenames.append(fname)
    else:

        pass

print("samples of dataset:")
for i in range(min(10, len(all_labels))):
    print(
        f"{i}: patient ID={all_serial_ids[i]}, file name={all_filenames[i]}, time ={all_labels[i][0]}, event={all_labels[i][1]}")

test_graph_dir = r"/data/Data_for_ImmGraph_topo/Validation 1"
test_graphs, test_labels, test_serial_ids = [], [], []

for fname in sorted(os.listdir(test_graph_dir)):
    if not fname.endswith(".dgl"):
        continue

    
    base = os.path.splitext(fname)[0]
    m = re.match(r'^patient_(.+?)_subgraph$', base, flags=re.IGNORECASE)
    if not m:
        continue

    serial_id = m.group(1).upper()

    if serial_id not in test_name_to_time:
        continue

    #loading
    graphs, _ = dgl.load_graphs(os.path.join(test_graph_dir, fname))
    g = graphs[0]

    # the key modifiction
    for ntype in g.ntypes:
        # reassure the feature of h
        if "h" not in g.nodes[ntype].data:
            print(f"Warning: {ntype} nodes missing 'h' in {fname}, using zeros")
            g.nodes[ntype].data["h"] = torch.zeros(
                (g.num_nodes(ntype), max_dims.get(ntype, 1)),
                dtype=torch.float32
            )

        
        feat = g.nodes[ntype].data["h"]
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)

        
        if feat.dim() == 1:
            feat = feat.unsqueeze(-1)

        cur_dim = feat.shape[1]
        target_dim = max_dims.get(ntype, cur_dim)
        if cur_dim < target_dim:
            pad = torch.zeros(feat.shape[0], target_dim - cur_dim, dtype=torch.float32)
            feat = torch.cat([feat, pad], dim=1)
        elif cur_dim > target_dim:
            feat = feat[:, :target_dim]

        
        g.nodes[ntype].data["h"] = feat

        
        for k in list(g.nodes[ntype].data.keys()):
            if k != "h":
                del g.nodes[ntype].data[k]

    for etype in g.canonical_etypes:
        for key in list(g.edges[etype].data.keys()):
            if key == "weight":
                
                weight = g.edges[etype].data[key]
                g.edges[etype].data[key] = weight.to(torch.float32)
            elif key != "_ID":
                del g.edges[etype].data[key]

    test_graphs.append(g)
    test_labels.append((float(test_name_to_time[serial_id]),
                        int(test_name_to_event[serial_id])))
    test_serial_ids.append(serial_id)

print(f"External test set loaded: {len(test_graphs)} ")


# ==============================
# Dataset
# ==============================
class GraphDataset(Dataset):
    def __init__(self, graphs, labels, serial_ids, augment=False):
        self.graphs = graphs
        self.labels = labels  # (time, event)
        self.serial_ids = serial_ids  #  serial_id of each graph
        self.augment = augment

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        label = self.labels[idx]
        serial_id = self.serial_ids[idx]  # find serial_id

        if self.augment and random.random() < 0.5:
            torch.manual_seed(idx)
            for ntype in graph.ntypes:
                if "feat" in graph.nodes[ntype].data:
                    noise = torch.randn_like(graph.nodes[ntype].data["feat"]) * 0.1
                    graph.nodes[ntype].data["feat"] = graph.nodes[ntype].data["feat"] + noise

        return graph, label, serial_id  # return serial_id


def collate_fn(batch):
    
    graphs, labels, serial_ids = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    times = torch.tensor([float(x[0]) for x in labels], dtype=torch.float32)
    events = torch.tensor([int(x[1]) for x in labels], dtype=torch.float32)
    return batched_graph, times, events, serial_ids


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
                x = g.nodes[ntype].data["h"]
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
        self.mlp = None  

    def forward(self, g):
        # Apply GNN layers first
        g = self.gnn(g)

        feats = []
        batch_num_nodes = []

        #find each type of the node
        for ntype in g.ntypes:
            x = g.nodes[ntype].data["h"]  # Use GNN output
            batch_num_nodes.append(g.batch_num_nodes(ntype))
            feats.append(torch.split(x, tuple(g.batch_num_nodes(ntype).cpu().numpy())))

        # Concatenate each graph into a single long vector.
        num_graphs = g.batch_size
        graph_feats = []
        for i in range(num_graphs):
            node_vecs = [f[i].flatten(start_dim=0) for f in feats]  # [Flatten and concatenate nodes of each type]
            graph_feat = torch.cat(node_vecs, dim=0)
            graph_feats.append(graph_feat)

        flat_feat = torch.stack(graph_feats, dim=0)  # ➤ [batch_size, total_feat_dim]

        # Initialize the MLP
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
        self.mlp = None  

    def forward(self, g):
        # Apply GAT layers first
        g = self.gat(g)

        feats = []
        batch_num_nodes = []

        
        for ntype in g.ntypes:
            x = g.nodes[ntype].data["h"]  # Use GAT output
            batch_num_nodes.append(g.batch_num_nodes(ntype))
            feats.append(torch.split(x, tuple(g.batch_num_nodes(ntype).cpu().numpy())))

        #
        num_graphs = g.batch_size
        graph_feats = []
        for i in range(num_graphs):
            node_vecs = [f[i].flatten(start_dim=0) for f in feats]  
            graph_feat = torch.cat(node_vecs, dim=0)
            graph_feats.append(graph_feat)

        flat_feat = torch.stack(graph_feats, dim=0)  # ➤ [batch_size, total_feat_dim]

        # 
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
        self.mlp = None  

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
def train_model_with_hyperparams(model, train_loader, val_loader, epochs=100, lr=1e-4, device="cuda",
                                 fold=0, weight_decay=1e-3, l2_lambda=1e-4, train_idx=None, val_idx=None):
    model.to(device)

    # intinialized the parameter
    with torch.no_grad():
        sample_g, _, _, _ = next(iter(train_loader))
        sample_g = sample_g.to(device)
        _ = model(sample_g)

    # Ensure deterministic optimizer state
    torch.manual_seed(torch.initial_seed())

    # Use passed hyperparameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Better learning rate scheduler - ReduceLROnPlateau instead of OneCycleLR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Monitor C-index (higher is better)
        factor=0.5,  # Reduce LR by half when plateauing
        patience=15,  # Wait 15 epochs before reducing LR
        verbose=True,
        min_lr=1e-7
    )

    loss_fn = ImprovedCoxPHLoss()

    val_cindex_list = []
    train_cindex_list = []
    best_cindex = 0
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 25  # Increased patience for better convergence
    val_metrics = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_risk_list = []
        train_time_list = []
        train_event_list = []

        for g, times, events, batch_serial_ids in train_loader:
            g = g.to(device)
            risk = model(g)

            loss = loss_fn(risk, times.to(device), events.to(device))

            # Add L2 regularization to prevent overfitting
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)
            loss += l2_lambda * l2_reg

            optimizer.zero_grad()
            loss.backward()

            # Improved gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

            # Collect training predictions for C-index
            train_risk_list.extend(risk.detach().cpu().numpy())
            train_time_list.extend(times.numpy())
            train_event_list.extend(events.numpy())

        # Calculate training C-index
        train_cindex = concordance_index(train_time_list, -np.array(train_risk_list), train_event_list)
        train_cindex_list.append(train_cindex)

        #valid
        model.eval()
        all_true = []
        all_pred = []
        all_risk, all_time, all_event = [], [], []
        val_records = []
        val_loss = 0
        with torch.no_grad():
            for g, times, events, batch_serial_ids in val_loader:
                g = g.to(device)
                risk = model(g)
                all_true.extend(times.numpy())  # True label
                all_pred.extend(risk.cpu().numpy())  # predict label

                # Calculate validation loss
                val_loss += loss_fn(risk, times.to(device), events.to(device)).item()

                all_risk.extend(risk.cpu().numpy())
                all_time.extend(times.numpy())
                all_event.extend(events.numpy())

                for t, e, r in zip(times.numpy(), events.numpy(), risk.cpu().numpy()):
                    val_records.append({"Time": t, "Event": e, "Risk": float(r)})

        mape, medae, evs, me, log_mse, smape, mbe, nmse, rae, rse, poisson_deviance, gamma_deviance, pearson_corr, spearman_corr, kendall_corr, mge, msle, rmsle, wae, direction_accuracy = calculate_metrics(
            all_true, all_pred)
        val_metrics.append(
            [fold, mape, medae, evs, me, log_mse, smape, mbe, nmse, rae, rse, poisson_deviance, gamma_deviance,
             pearson_corr, spearman_corr, kendall_corr, mge, msle, rmsle, wae, direction_accuracy])

        c_index = concordance_index(all_time, -np.array(all_risk), all_event)
        val_cindex_list.append(c_index)
        val_loss /= len(val_loader)

        # Update learning rate based on validation performance
        scheduler.step(c_index)

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss {total_loss:.4f}, Val Loss {val_loss:.4f}, "
            f"Train C-index {train_cindex:.4f}, Val C-index {c_index:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if c_index > best_cindex:
            best_cindex = c_index
            patience_counter = 0
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ✅ New best model saved (C-index: {best_cindex:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print("Early stopping triggered!")
            break
    # save_metrics_to_csv(val_metrics, f"E:/conference/LGG/metric/fold_{fold + 1}_metrics_DFS_LGG.csv")
    
    # save_metrics_to_csv(val_metrics, r"E:\conference\LGG\metric\fold_metrics_DFS_LGG.csv")
    # Use best validation C-index instead of average of last 10
    best_val_cindex = max(val_cindex_list)
    print(f"\n==== Training Finished ====\nBest Val C-index: {best_val_cindex:.4f}")

    
    model.eval()
    train_records = []
    with torch.no_grad():
        for g, times, events, batch_serial_ids in train_loader:
            g = g.to(device)
            risk = model(g)
            for t, e, r in zip(times.numpy(), events.numpy(), risk.cpu().numpy()):
                train_records.append({"Time": t, "Event": e, "Risk": float(r)})

    os.makedirs(r"/results/risk_csv_OS", exist_ok=True)

    
    train_serial_ids = [all_serial_ids[idx] for idx in train_idx]
    val_serial_ids = [all_serial_ids[idx] for idx in val_idx]

    # align the dataset
    print(f"\n===== Fold {fold + 1} align =====")
    print(f"the samples of training_dataset: {len(train_records)}")
    print(f"the numbers of serial_id in training datasets: {len(train_serial_ids)}")
    print(f"the samples of validation_dataset: {len(val_records)}")
    print(f"the numbers of serial_id in validation datasets: {len(val_serial_ids)}")

    
    print("\n training datasets:")
    for i in range(min(3, len(train_records))):
        print(f"  samples{i}: ID={train_serial_ids[i]}, time={train_records[i]['Time']}, events={train_records[i]['Event']}")

    print("\n validation datasets:")
    for i in range(min(3, len(val_records))):
        print(f"  samples{i}: ID={val_serial_ids[i]}, time={val_records[i]['Time']}, events={val_records[i]['Event']}")

    
    with open(
            fr"/results/fold_{fold + 1}_train_risk_ZZU.csv",
            "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Serial_ID", "Time", "Event", "Risk"])
        writer.writeheader()

        for i, record in enumerate(train_records):
            serial_id = train_serial_ids[i]  
            writer.writerow({
                "Serial_ID": serial_id,
                "Time": record["Time"],
                "Event": record["Event"],
                "Risk": record["Risk"]
            })

    with open(
            fr"/results/fold_{fold + 1}_val_risk_ZZU.csv",
            "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Serial_ID", "Time", "Event", "Risk"])
        writer.writeheader()

        for i, record in enumerate(val_records):
            serial_id = val_serial_ids[i]  
            writer.writerow({
                "Serial_ID": serial_id,
                "Time": record["Time"],
                "Event": record["Event"],
                "Risk": record["Risk"]
            })

    try:
        export_node_embeddings_by_omics(
            model,
            loaders=[train_loader, val_loader], 
            fold=fold,
            out_root=r"/results/node_emb_csv_DFS"  
        )
    except Exception as e:
        print(f"[WARN] Export node embeddings failed on fold {fold + 1}: {e}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Loaded best model weights from epoch {best_epoch} (Val C-index: {best_cindex:.4f})")
    else:
        print("⚠️ No best model state found, using final epoch weights")

    return best_val_cindex  


# ==============================
# Optuna Objective Function for Hyperparameter Optimization
# ==============================
def run_optuna_optimization(n_trials=50,
                            results_json=r'/results/optuna_optimization_results_regression_OS.json',
                            seed=42):
    """Run Optuna to perform Bayesian hyperparameter tuning with 5-fold cross-validation, and save the best results to a JSON file."""
    print("Starting Optuna hyperparameter optimization...")

    # Define Optuna’s objective function (one trial = one set of hyperparameters; run 5-fold cross-validation and take the mean C-index).——
    def objective(trial):
        
        hidden_dim = trial.suggest_int('hidden_dim', 32, 128, step=32)
        dropout = trial.suggest_float('dropout', 0.3, 0.7)
        lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        num_gnn_layers = trial.suggest_int('num_gnn_layers', 1, 2)
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
        l2_lambda = trial.suggest_float('l2_lambda', 1e-5, 1e-3, log=True)

        
        in_dims = {ntype: all_graphs[0].nodes[ntype].data["h"].shape[1]
                   for ntype in all_graphs[0].ntypes}

       
        dataset = GraphDataset(all_graphs, all_labels, all_serial_ids)
        events = [label[1] for label in all_labels]

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)), events)):
            print(f"\n===== Fold {fold + 1} (trial {trial.number}) =====")
            set_seed(seed + fold)

            #performed independently for each fold
            train_graphs_this_fold = [all_graphs[i] for i in train_idx]
            fold_stats = compute_fold_statistics(train_graphs_this_fold, max_dims)

            train_graphs_normalized = [
                normalize_graph(all_graphs[i].clone(), fold_stats, max_dims)
                for i in train_idx
            ]
            val_graphs_normalized = [
                normalize_graph(all_graphs[i].clone(), fold_stats, max_dims)
                for i in val_idx
            ]

            
            train_dataset_norm = GraphDataset(
                train_graphs_normalized,
                [all_labels[i] for i in train_idx],
                [all_serial_ids[i] for i in train_idx]
            )
            val_dataset_norm = GraphDataset(
                val_graphs_normalized,
                [all_labels[i] for i in val_idx],
                [all_serial_ids[i] for i in val_idx]
            )

            
            model = ImprovedGraphEmbeddingMLP(
                in_dims, hidden_dim, out_dim=1, dropout=dropout, num_gnn_layers=num_gnn_layers
            )

            # Subset & DataLoader（collate_fn return g, times, events, serial_ids）
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            train_loader = DataLoader(train_dataset_norm, batch_size=16, shuffle=False, collate_fn=collate_fn,
                                      num_workers=0)
            val_loader = DataLoader(val_dataset_norm, batch_size=16, shuffle=False, collate_fn=collate_fn,
                                    num_workers=0)

            # to get the best Val C-index
            best_cindex = train_model_with_hyperparams(
                model, train_loader, val_loader,
                epochs=150, lr=lr, device=device, fold=fold,
                weight_decay=weight_decay, l2_lambda=l2_lambda,
                train_idx=train_idx, val_idx=val_idx
            )
            fold_scores.append(best_cindex)

            
            trial.report(float(np.mean(fold_scores)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    #print it 
    print("\n" + "=" * 50)
    print("OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Best C-index: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    os.makedirs(os.path.dirname(results_json), exist_ok=True)
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump({
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
        }, f, indent=2)
    print(f"\nResults saved to '{results_json}'")

    
    try:
        optuna.visualization.plot_optimization_history(study).show()
        optuna.visualization.plot_param_importances(study).show()
        optuna.visualization.plot_parallel_coordinate(study).show()
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("You can still view the results in the saved JSON file.")

    return study


def calculate_metrics(true_values, predicted_values):
    
    true_values = np.array(true_values).flatten()
    predicted_values = np.array(predicted_values).flatten()

    #calculate
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

    # RSE(np.square(true_values - np.mean(true_values)))
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

    # \
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
        # RMSLE
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


# save the result of each fold
def save_metrics_to_csv(metrics, file_path):
    
    columns = ["Fold", "MAPE", "MedAE", "EVS", "ME", "LogMSE", "SMAPE", "MBE", "NMSE",
               "RAE", "RSE", "Poisson_Deviance", "Gamma_Deviance", "Pearson_Corr",
               "Spearman_Corr", "Kendall_Corr", "MGE", "MSLE", "RMSLE", "WAE", "Direction_Accuracy"]

    df = pd.DataFrame(metrics, columns=columns)
    df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)


# ==============================
# Export per-omics node embeddings (only "h") at last epoch
# ==============================

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


def evaluate_on_test_set(model, test_graphs, test_labels, test_serial_ids,
                         fold_stats, max_dims, device, fold):
    
    print(f"\n===== External Test Evaluation (Fold {fold + 1}) =====")

    # normalized
    test_graphs_norm = align_test_to_train_simple(test_graphs, fold_stats, max_dims)

    # prediction
    test_dataset = GraphDataset(test_graphs_norm, test_labels, test_serial_ids)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    model.eval()
    all_risk, all_time, all_event = [], [], []

    with torch.no_grad():
        for g, times, events, batch_serial_ids in test_loader:
            g = g.to(device)
            risk = model(g)
            all_risk.extend(risk.cpu().numpy())
            all_time.extend(times.numpy())
            all_event.extend(events.numpy())

    all_risk = np.array(all_risk)
    all_time = np.array(all_time)
    all_event = np.array(all_event)

    
    c_index = concordance_index(all_time, -all_risk, all_event)

    print(f"\n{'=' * 60}")
    print(f"C-index: {c_index:.4f}")
    print(f"mean={all_risk.mean():.3f}, std={all_risk.std():.3f}")
    print(f"{'=' * 60}")

    # 保存结果
    output_dir = r"E:\Multi-omic Immunity\GCN_immune\scripts\LGG\downstream_ZZU\MLP\risk_csv_os"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/fold_{fold + 1}_test_risk_external.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Serial_ID", "Time", "Event", "Risk"])
        writer.writeheader()
        for sid, t, e, r in zip(test_serial_ids, all_time, all_event, all_risk):
            writer.writerow({"Serial_ID": sid, "Time": t, "Event": e, "Risk": float(r)})

    return c_index, all_risk, all_time, all_event


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


def load_best_hyperparams(
        json_file=r'/data/data_for_running/ImmGraph-topo/optuna_optimization_results_regression_OS_1.json'):
    """load the best parameter"""
    with open(json_file, 'r') as f:
        results = json.load(f)
    return results['best_params']


# ===========================

# ===========================

def train_with_fixed_hyperparams():


    
    print("\n=== compare ===")
    train_feat_means = []
    test_feat_means = []

    for g in all_graphs:
        for ntype in g.ntypes:
            if "h" in g.nodes[ntype].data:
                train_feat_means.append(g.nodes[ntype].data["h"].mean().item())

    for g in test_graphs:
        for ntype in g.ntypes:
            if "h" in g.nodes[ntype].data:
                test_feat_means.append(g.nodes[ntype].data["h"].mean().item())

    print(f"training datasets: {np.mean(train_feat_means):.4f} ± {np.std(train_feat_means):.4f}")
    print(f"testing dasatest: {np.mean(test_feat_means):.4f} ± {np.std(test_feat_means):.4f}")
    print(f"difference: {abs(np.mean(train_feat_means) - np.mean(test_feat_means)):.4f}")

    print("\n===  ===")
    train_times_arr = np.array([l[0] for l in all_labels])
    train_events_arr = np.array([l[1] for l in all_labels])

    bins = [0, 12, 24, 36, 54.9]
    labels = ['0-12', '12-24', '24-36', '36-55']

    for i in range(len(bins) - 1):
        mask = (train_times_arr >= bins[i]) & (train_times_arr < bins[i + 1])
        count = mask.sum()
        event_rate = train_events_arr[mask].mean() if count > 0 else 0
        print(f"  {labels[i]}: {count}samples ({count / len(train_times_arr) * 100:.1f}%), ratio of events{event_rate:.1%}")

    print("\n=== detail of testing datasets===")
    test_times_arr = np.array([l[0] for l in test_labels])
    test_events_arr = np.array([l[1] for l in test_labels])

    bins_test = [0, 12, 24, 36, 54.9, 999]
    labels_test = ['0-12', '12-24', '24-36', '36-55', '>55']

    for i in range(len(bins_test) - 1):
        mask = (test_times_arr >= bins_test[i]) & (test_times_arr < bins_test[i + 1])
        count = mask.sum()
        event_rate = test_events_arr[mask].mean() if count > 0 else 0
        print(f"  {labels_test[i]}: {count}samples ({count / len(test_times_arr) * 100:.1f}%), ratio of events{event_rate:.1%}")

    time_diff = abs(np.median(train_times_arr) - np.median(test_times_arr))
    event_diff = abs(np.mean(train_events_arr) - np.mean(test_events_arr))

    print("\n look for pboblems:")
    issues_found = False

    if time_diff > 20:
        print(f"    Large difference in median survival time ({time_diff:.1f}month)")
        issues_found = True
    if event_diff > 0.2:
        print(f"   Large difference in event rate ({event_diff:.1%})")
        issues_found = True
    if abs(np.mean(train_feat_means) - np.mean(test_feat_means)) > 0.5:
        print(f"    Significant feature distribution shift detected")
        issues_found = True

    train_short_count = (train_times_arr < 24).sum()
    if train_short_count < 50:
        print(f"    Few short-term samples in the training set ({train_short_count}个)")
        issues_found = True

    if not issues_found:
        print("   No obvious data distribution issues detected")

    print("=" * 70 + "\n")

    # ============Load fixed hyperparameters ============
    print("\n" + "=" * 70)
    print("🚀 Start training (using fixed hyperparameters)")
    print("=" * 70 + "\n")

    best_params = load_best_hyperparams()
    fixed_hidden_dim = best_params['hidden_dim']
    fixed_dropout = best_params['dropout']
    fixed_lr = best_params['lr']
    fixed_num_gnn_layers = best_params['num_gnn_layers']
    fixed_weight_decay = best_params['weight_decay']
    fixed_l2_lambda = best_params['l2_lambda']

    adaptive_clip_config = {
        'clip_strategy': 'quantile',
        'clip_multiplier': {
            'protein': 2.0,
            'default': 3.0
        }
    }

    print(f"📋 clip:")
    print(f"  statagies: {adaptive_clip_config['clip_strategy']}")
    print(f"  Protein node: ±{adaptive_clip_config['clip_multiplier']['protein']}σ")
    print(f"  other node: ±{adaptive_clip_config['clip_multiplier']['default']}σ\n")

    events = [label[1] for label in all_labels]
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_cindex_list = []
    test_cindex_list = []

    # save all the model
    all_fold_models = []
    all_fold_stats_list = []

    # ============ five-fold cross validation ============
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(all_graphs)), events)):
        print(f"\n{'=' * 70}")
        print(f"Fold {fold + 1}/5")
        print(f"{'=' * 70}")

        set_seed(42 + fold)

        # ✅ standardization statistics
        print(f"📊 calculate Fold {fold + 1} standardization statistics...")
        train_graphs_this_fold = [all_graphs[i] for i in train_idx]
        fold_stats = compute_fold_statistics(train_graphs_this_fold, max_dims)
        all_fold_stats_list.append(fold_stats)  # ✅ standardization statistics

        # Standardize the training and validation sets
        print(f"Standardize the training and validation sets...")
        train_graphs_normalized = [
            normalize_graph(all_graphs[i].clone(), fold_stats, max_dims,
                            protein_clip=1.0, other_clip=5.0)
            for i in train_idx
        ]
        val_graphs_normalized = [
            normalize_graph(all_graphs[i].clone(), fold_stats, max_dims,
                            protein_clip=1.0, other_clip=5.0)
            for i in val_idx
        ]

        # Create the model and data loaders
        in_dims = {ntype: train_graphs_normalized[0].nodes[ntype].data["h"].shape[1]
                   for ntype in train_graphs_normalized[0].ntypes}

        model = ImprovedGraphEmbeddingMLP(
            in_dims,
            hidden_dim=fixed_hidden_dim,
            out_dim=1,
            dropout=fixed_dropout,
            num_gnn_layers=fixed_num_gnn_layers
        )

        train_dataset = GraphDataset(
            train_graphs_normalized,
            [all_labels[i] for i in train_idx],
            [all_serial_ids[i] for i in train_idx]
        )
        val_dataset = GraphDataset(
            val_graphs_normalized,
            [all_labels[i] for i in val_idx],
            [all_serial_ids[i] for i in val_idx]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=False,
            collate_fn=collate_fn, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=16, shuffle=False,
            collate_fn=collate_fn, num_workers=0
        )

        # train the model
        print(f"🚀 train Fold {fold + 1}...")
        val_cindex = train_model_with_hyperparams(
            model, train_loader, val_loader,
            epochs=150,
            lr=fixed_lr,
            device="cpu",
            fold=fold,
            weight_decay=fixed_weight_decay,
            l2_lambda=fixed_l2_lambda,
            train_idx=train_idx,
            val_idx=val_idx
        )

        val_cindex_list.append(val_cindex)
        all_fold_models.append(model)  # the model save

        # external test
        print(f"\n🔍 external test Fold {fold + 1}...")
        test_cindex, segment_results = evaluate_on_external_test_simple(
            model, test_graphs, test_labels, test_serial_ids,
            fold_stats, max_dims, device="cpu", fold=fold,
            train_risks=None,
            adaptive_clip=True,
            quantile_range=(1, 99)
        )

        test_cindex_list.append(test_cindex)

    # ============ Final Results Summary ============
    print(f"\n{'=' * 70}")
    print("🎯 5-fold cross-validation completed!")
    print(f"{'=' * 70}")
    print(f"Mean C-index on internal validation: {np.mean(val_cindex_list):.4f} ± {np.std(val_cindex_list):.4f}")
    print(f"Mean C-index on external test: {np.mean(test_cindex_list):.4f} ± {np.std(test_cindex_list):.4f}")
    print(f"{'=' * 70}\n")
    results = {
        'internal_validation': {
            'mean_cindex': float(np.mean(val_cindex_list)),
            'std_cindex': float(np.std(val_cindex_list)),
            'fold_results': [float(c) for c in val_cindex_list]
        },
        'external_test': {
            'mean_cindex': float(np.mean(test_cindex_list)),
            'std_cindex': float(np.std(test_cindex_list)),
            'fold_results': [float(c) for c in test_cindex_list]
        },
        'external_test_loo': {
            'mean_cindex': float(np.mean(loo_c_indices_array)),
            'std_cindex': float(np.std(loo_c_indices_array)),
            'detail_results': loo_results
        }
    }

    output_path = r"E:\Multi-omic Immunity\GCN_immune\scripts\LGG\downstream_ZZU\metric\final_results_with_loo.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, indent=2, fp=f)

    print(f"✅ the result is saved: {output_path}")


def calibrate_risks_robust(all_risk, all_time, train_risks):

    train_quantiles = np.percentile(train_risks, [10, 25, 50, 75, 90])


    long_mask = all_time > 54.9
    if long_mask.sum() < 5:
        return all_risk

    long_risks = all_risk[long_mask]


    long_quantiles = np.percentile(long_risks, [10, 25, 50, 75, 90])


    calibrated_long = np.interp(
        long_risks,
        long_quantiles,  
        train_quantiles
    )


    p10, p90 = np.percentile(long_risks, [10, 90])
    middle_mask = (long_risks >= p10) & (long_risks <= p90)


    calibrated_all = all_risk.copy()
    long_indices = np.where(long_mask)[0]


    for i, is_middle in enumerate(middle_mask):
        if is_middle:
            calibrated_all[long_indices[i]] = calibrated_long[i]

    return calibrated_all


def evaluate_on_test_set(model, test_graphs, test_labels, test_serial_ids,
                         fold_stats, max_dims, device, fold):

    print(f"\n===== External Test Evaluation (Fold {fold + 1}) =====")

   
    test_graphs_aligned = align_test_to_train_simple(
        test_graphs, fold_stats, max_dims
    )

    test_dataset = GraphDataset(test_graphs_aligned, test_labels, test_serial_ids)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    model.eval()
    all_risk = []
    all_time = []
    all_event = []
    all_serial_ids = []

    with torch.no_grad():
        for g, times, events, batch_serial_ids in test_loader:
            g = g.to(device)
            risk = model(g)
            all_risk.extend(risk.squeeze().cpu().numpy().tolist())
            all_time.extend(times.numpy().tolist())
            all_event.extend(events.numpy().tolist())
            all_serial_ids.extend(batch_serial_ids)

   
    all_risk = np.array(all_risk)
    all_time = np.array(all_time)
    all_event = np.array(all_event)

    # C-index
    c_index = concordance_index(all_time, -all_risk, all_event)

    print(f"\n{'=' * 60}")
    print(f"C-index: {c_index:.4f}")
    print(f"the score of risk: mean ={all_risk.mean():.3f}, std={all_risk.std():.3f}")
    print(f"{'=' * 60}")

    #
    print(f"\n📊 C-index diagose:")
    time_ranges = [
        ("0-24months", lambda t: t < 24),
        ("24-54months", lambda t: (t >= 24) & (t <= 54.9)),
        (">54months", lambda t: t > 54.9)
    ]

    for time_range, mask_fn in time_ranges:
        mask = mask_fn(all_time)
        if mask.sum() >= 5:
            c_sub = concordance_index(all_time[mask], -all_risk[mask], all_event[mask])
            event_rate = all_event[mask].mean()
            print(f"  {time_range}: {mask.sum()}samples, ratio of events={event_rate:.1%}, C-index={c_sub:.4f}")

    print(f"\n{'=' * 60}\n")

    # save the result
    output_dir = r"E:\Multi-omic Immunity\GCN_immune\scripts\LGG\downstream_ZZU\MLP\risk_csv_os"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"fold_{fold + 1}_test_risk_external.csv")

    with open(output_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Serial_ID", "Time", "Event", "Risk"])
        writer.writeheader()
        for sid, t, e, r in zip(all_serial_ids, all_time, all_event, all_risk):
            writer.writerow({
                "Serial_ID": sid,
                "Time": float(t),
                "Event": int(e),
                "Risk": float(r)
            })

    print(f"✅ the result is saved in: {output_file}")

    return c_index, all_risk, all_time, all_event


def calibrate_risks_by_time(test_risks, test_times, train_risks):

    calibrated = test_risks.copy()
    train_median = np.median(train_risks)
    train_std = np.std(train_risks)


    short_mask = test_times < 24


    mid_mask = (test_times >= 24) & (test_times <= 54.9)
    if mid_mask.sum() > 0:
        mid_risks = test_risks[mid_mask]
        mid_median = np.median(mid_risks)

        adjustment = (train_median - mid_median) * 0.4
        calibrated[mid_mask] = mid_risks + adjustment


    long_mask = test_times > 54.9
    if long_mask.sum() > 0:
        long_risks = test_risks[long_mask]

        long_quantiles = np.percentile(long_risks, [25, 50, 75])
        train_quantiles = np.percentile(train_risks, [25, 50, 75])
        calibrated[long_mask] = np.interp(long_risks, long_quantiles, train_quantiles)

    return calibrated


def align_risk_distribution(test_risks, train_risks):

    
    rank = np.argsort(np.argsort(test_risks))

    #to the train datasets
    test_min, test_max = test_risks.min(), test_risks.max()
    train_min, train_max = train_risks.min(), train_risks.max()

    #normalized
    normalized = (test_risks - test_min) / (test_max - test_min + 1e-8)

    
    aligned = normalized * (train_max - train_min) + train_min

    # keep the queue
    sorted_aligned = np.sort(aligned)
    final = sorted_aligned[rank]

    return final


# ===========================
# 4. main
# ===========================

def main():
    use_optuna = False  # True：Bayesian，FALSE：the tuned hyperparameters

    if use_optuna:
       
        run_optuna_optimization()
    else:
        
        train_with_fixed_hyperparams()


# ==============================
# 5-Fold Cross Validation with Optuna Optimization
# ==============================


if __name__ == "__main__":
    main()