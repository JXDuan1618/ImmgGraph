import os
import dgl
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from community import community_louvain
import matplotlib.pyplot as plt


# Step 1: load Graph
def load_graph(graph_path):
    graph, _ = dgl.load_graphs(graph_path)
    return graph[0]


# Step 2: extract node feature
def extract_node_features(graph):
    node_features = {}

    for ntype in graph.ntypes:
        if 'feat' in graph.nodes[ntype].data:
            node_features[ntype] = graph.nodes[ntype].data['feat']
        elif 'h' in graph.nodes[ntype].data:
            node_features[ntype] = graph.nodes[ntype].data['h']
        else:
            raise KeyError(
                f"Node type '{ntype}' has no feature field named 'feat' or 'h'. "
                f"Available fields: {list(graph.nodes[ntype].data.keys())}"
            )

    return node_features


# Step 3: calculate similarity
def compute_similarity(node_features):

    all_node_features = np.concatenate(
        [node_features[ntype].detach().cpu().numpy() for ntype in node_features],
        axis=0
    )

    similarity_matrix = cosine_similarity(all_node_features)
    return similarity_matrix


def heterograph_to_networkx(graph):

    nx_graph = nx.Graph()

    # 1) 添加所有节点
    for ntype in graph.ntypes:
        for nid in range(graph.num_nodes(ntype)):
            node_name = f"{ntype}_{nid}"
            nx_graph.add_node(
                node_name,
                ntype=ntype,
                local_id=nid
            )

    # 2) 添加所有边
    for canonical_etype in graph.canonical_etypes:
        src_type, edge_type, dst_type = canonical_etype
        src_nodes, dst_nodes = graph.edges(etype=canonical_etype)

        src_nodes = src_nodes.detach().cpu().numpy()
        dst_nodes = dst_nodes.detach().cpu().numpy()

        for s, d in zip(src_nodes, dst_nodes):
            src_name = f"{src_type}_{int(s)}"
            dst_name = f"{dst_type}_{int(d)}"

            nx_graph.add_edge(
                src_name,
                dst_name,
                etype=edge_type,
                canonical_etype=str(canonical_etype)
            )

    return nx_graph


# Step 4: detect community
def community_detection(graph):
    print("Converting DGL heterograph to NetworkX graph...")
    nx_graph = heterograph_to_networkx(graph)

    print(f"NetworkX graph nodes: {nx_graph.number_of_nodes()}")
    print(f"NetworkX graph edges: {nx_graph.number_of_edges()}")

    print("Running Louvain community detection...")
    partition = community_louvain.best_partition(nx_graph)

    communities = {}
    for node_name, community_id in partition.items():
        communities.setdefault(community_id, [])

        node_attr = nx_graph.nodes[node_name]
        node_type = node_attr.get("ntype", "unknown")
        local_id = node_attr.get("local_id", "unknown")

        communities[community_id].append({
            "node": node_name,
            "id": local_id,
            "type": node_type
        })


    print(f"Detected communities: {len(communities)}")
    for community_id, nodes in communities.items():
        type_counts = {}
        for node_info in nodes:
            ntype = node_info["type"]
            type_counts[ntype] = type_counts.get(ntype, 0) + 1

        print(
            f"Community {community_id}: "
            f"total={len(nodes)}, "
            f"type_counts={type_counts}"
        )

    return communities, nx_graph


# Step 5: print
def output_community_nodes(communities, output_file="community_nodes.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        for community_id, nodes in communities.items():
            f.write(f"Community {community_id}\n")
            for node_info in nodes:
                f.write(
                    f"  node={node_info['node']}\t"
                    f"type={node_info['type']}\t"
                    f"local_id={node_info['id']}\n"
                )
            f.write("\n")

    print(f"Community node list saved to: {output_file}")


# Step 6: visualization
def visualize_community(nx_graph, communities, output_png="community_visualization.png"):
    plt.figure(figsize=(12, 12))

    print("Computing graph layout...")
    pos = nx.spring_layout(nx_graph, seed=42)

    community_ids = sorted(communities.keys())
    cmap = plt.colormaps.get_cmap("tab20")

    # 绘制节点
    for idx, community_id in enumerate(community_ids):
        nodes = communities[community_id]
        node_ids = [node_info["node"] for node_info in nodes]

        color = cmap(idx % 20)

        nx.draw_networkx_nodes(
            nx_graph,
            pos,
            nodelist=node_ids,
            node_size=5,
            node_color=[color],
            label=f"Community {community_id}"
        )


    nx.draw_networkx_edges(nx_graph, pos, alpha=0.2, width=0.3)


    if len(community_ids) <= 20:
        plt.legend(title="Communities", loc="best", markerscale=4)

    plt.title("Community Detection using Louvain")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"Community visualization saved to: {output_png}")


def main():
    print("Current working directory:", os.getcwd())


    graph_path = r"multiomics_all.dgl"


    print("Loading graph...")
    graph = load_graph(graph_path)

    print("Node types:", graph.ntypes)
    print("Canonical edge types:", graph.canonical_etypes)


    print("Extracting node features...")
    node_features = extract_node_features(graph)

    for ntype, feat in node_features.items():
        print(f"{ntype} feature shape: {tuple(feat.shape)}")


    print("Computing node similarity matrix...")
    similarity_matrix = compute_similarity(node_features)

    with open(r"node_feature_f0s52train.txt", "w", encoding="utf-8") as log_file:
        log_file.write("similarity_matrix:\n")
        log_file.write(str(similarity_matrix))

    print("similarity matrix shape:", similarity_matrix.shape)
    print("Similarity matrix saved to: node_feature_f0s52train.txt")

    # Step 4: 社区检测
    communities, nx_graph = community_detection(graph)

    with open(r"node_feature_f999s52train.txt", "w", encoding="utf-8") as log_file:
        log_file.write("The communities are:\n")
        log_file.write(str(communities))

    print("Community dict saved to: node_feature_f999s52train.txt")


    output_community_nodes(communities, output_file=r"community_nodes.txt")


    visualize_community(
        nx_graph,
        communities,
        output_png=r"community_visualization.png"
    )

    print("Done!")


if __name__ == "__main__":
    main()
