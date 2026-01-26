# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')

import torch, dgl, numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from scipy.spatial.distance import cdist

print("Loading graph and node embeddings...")
graph_list, _ = dgl.load_graphs(
    r"data/ImmGraph_results/heterographs_ImmGraph/train_graph_epoch_160f4s.dgl")
g = graph_list[0]

# === Load node embeddings ===
dna_h = g.nodes['dna'].data['h'].cpu().numpy()
rna_h = g.nodes['rna'].data['h'].cpu().numpy()
protein_h = g.nodes['protein'].data['h'].cpu().numpy()

print(f"DNA shape: {dna_h.shape} | RNA shape: {rna_h.shape} | Protein shape: {protein_h.shape}")

# === Load true node IDs ===
dna_ids = pd.read_csv(r"data/ImmGraph_results/intermediate_product/nodes_dna.csv")['serial_id'].values
rna_ids = pd.read_csv(r"data/ImmGraph_results/intermediate_product/nodes_rna.csv")['serial_id'].values
protein_ids = pd.read_csv(r"data/ImmGraph_results/intermediate_product/ZZU_CSV_nodes/nodes_protein.csv")['serial_id'].values

print(f"DNA ID range: {dna_ids.min()} ~ {dna_ids.max()}")
print(f"RNA ID range: {rna_ids.min()} ~ {rna_ids.max()}")
print(f"Protein ID range: {protein_ids.min()} ~ {protein_ids.max()}")

# === Intra-layer K-NN ===
print("Building intra-layer k-NN edges...")
k = 10
graphs = {}

for name, X, IDs in zip(['dna', 'rna', 'protein'],
                         [dna_h, rna_h, protein_h],
                         [dna_ids, rna_ids, protein_ids]):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, indices = nbrs.kneighbors(X)


    all_distances_flat = distances[:, 1:].flatten()  
    epsilon = np.percentile(all_distances_flat, 10)  

    edges = []
    for idx, (dist_row, nbr_row) in enumerate(zip(distances, indices)):
        for d, j in zip(dist_row[1:], nbr_row[1:]):  
            if d <= epsilon:  
                edges.append((name, IDs[idx], name, IDs[j]))
    graphs[name] = edges
    print(f"{name} intra-layer candidate edges: {len(edges)}")

with open("intra_edges.txt", "w") as f:
    for name in graphs:
        for u in graphs[name]:
            f.write(f"{u[0]}_{u[1]}\t{u[2]}_{u[3]}\n")

# === Inter-layer edges ===
print("Building cross-layer edges using top 20% closest pairs (instead of fixed threshold)...")
edges_cross = []

pairs = [
    ('dna', dna_h, dna_ids, 'rna', rna_h, rna_ids),
    ('dna', dna_h, dna_ids, 'protein', protein_h, protein_ids),
    ('rna', rna_h, rna_ids, 'protein', protein_h, protein_ids)
]

for name1, X1, IDs1, name2, X2, IDs2 in pairs:
    
    D = cdist(X1, X2)
    flat_D = D.flatten()
    
    
    epsilon = np.percentile(flat_D, 10)  

    idx1, idx2 = np.where(D <= epsilon)
    
    for i, j in zip(idx1, idx2):
        edges_cross.append((name1, IDs1[i], name2, IDs2[j]))


print(f"Cross-layer candidate edges: {len(edges_cross)}")


with open("cross_edges.txt", "w") as f:
    for u in edges_cross:
        f.write(f"{u[0]}_{u[1]}\t{u[2]}_{u[3]}\n")

# === Combine candidate edges as set ===
print("Combining candidate edges...")
candidate_edges = set()
for edge_list in graphs.values():
    candidate_edges.update(edge_list)
candidate_edges.update(edges_cross)

print(f"Total candidate edges before filtering: {len(candidate_edges)}")

# === Filter: only edges that exist in original DGL ===
print("Filtering only edges that exist in the original DGL graph...")
final_edges = []

for etype in g.canonical_etypes:
    src_type, edge_type, dst_type = etype
    src, dst = g.edges(etype=etype)
    src = src.cpu().numpy()
    dst = dst.cpu().numpy()
    for s_id, d_id in zip(src, dst):
        pair_1 = (src_type, s_id, dst_type, d_id)
        pair_2 = (dst_type, d_id, src_type, s_id)
        if pair_1 in candidate_edges or pair_2 in candidate_edges:
            final_edges.append(pair_1)

print(f"Total edges kept after filtering: {len(final_edges)}")

# === Save final edges ===
with open("filtered_graph_visual_train.txt", "w") as f:
    for src_type, s_id, dst_type, d_id in final_edges:
        f.write(f"{src_type}_{s_id}\t{dst_type}_{d_id}\n")

print("Filtered edges saved to filtered_graph_visual_train.txt")

# === Build final graph ===
print("Building final NetworkX graph and writing enhanced_node_info_train.txt ...")
G = nx.Graph()
for src_type, s_id, dst_type, d_id in final_edges:
    src_label = f"{src_type}_{s_id}"
    dst_label = f"{dst_type}_{d_id}"
    G.add_edge(src_label, dst_label)

print(f"Final filtered graph nodes: {G.number_of_nodes()}")
print(f"Final filtered graph edges: {G.number_of_edges()}")

with open("enhanced_node_info_train.txt", "w", encoding="utf-8") as f:
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if neighbors:
            line = f"{node} connects to: {', '.join(neighbors)}\n"
        else:
            line = f"{node} has no connections\n"
        f.write(line)

print("Done! Final enhanced_node_info_train.txt includes only filtered connections.")
