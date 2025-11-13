import dgl
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from community import community_louvain
from mlxtend.frequent_patterns import fpgrowth, apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd
import matplotlib.pyplot as plt


# Step 1: load the graph
def load_graph(graph_path):
    graph, _ = dgl.load_graphs(graph_path)
    return graph[0]


# Step 2: Extract node features
def extract_node_features(graph):
    node_features = {ntype: graph.nodes[ntype].data['feat'] for ntype in graph.ntypes}
    return node_features


# Step 3: Compute the similarity between nodes
def compute_similarity(node_features):
    #Concatenate all node features into a matrix
    all_node_features = np.concatenate([node_features[ntype].cpu().numpy() for ntype in node_features], axis=0)

    #Compute cosine similarity
    similarity_matrix = cosine_similarity(all_node_features)
    return similarity_matrix


# Step 4: Convert node features into a format suitable for FP-Growth
def prepare_data_for_fpgrowth(node_features):

    feature_list = []
    for ntype, features in node_features.items():
        features_np = features.cpu().numpy()
        for feature in features_np:
            binarized_feature = (feature > 0.5).astype(int)
            feature_list.append(binarized_feature)

    df = pd.DataFrame(feature_list)
    return df


# Step 5: Extract frequent itemsets using the FP-Growth algorithm
def apply_fpgrowth(df, min_support=0.5):
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    return frequent_itemsets


# Step 6: Generate association rules
def generate_association_rules(frequent_itemsets, min_threshold=0.7):
    #Generate association rules using frequent_itemsets
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
    return rules


# Step 6:Perform community detection
def community_detection(graph):
    nx_graph = graph.to_networkx().to_undirected()
    rna_features = graph.nodes['rna'].data['feat'].cpu().numpy()
    dna_features = graph.nodes['dna'].data['feat'].cpu().numpy()
    protein_features = graph.nodes['protein'].data['feat'].cpu().numpy()

    dna_node_ids = list(range(len(dna_features)))  # DNA ID
    protein_node_ids = list(
        range(len(dna_features), len(dna_features) + len(protein_features)))  # PROTEIN ID
    rna_node_ids = list(range(len(dna_features) + len(protein_features), len(rna_features) + len(dna_features) + len(
        protein_features)))  # Rna ID

    print(f"RNA node IDs: {rna_node_ids}")
    print(f"DNA node IDs: {dna_node_ids}")
    print(f"Protein node IDs: {protein_node_ids}")

    node_type_mapping = {}
    for rna_id in rna_node_ids:
        node_type_mapping[rna_id] = 'rna'
    for dna_id in dna_node_ids:
        node_type_mapping[dna_id] = 'dna'
    for protein_id in protein_node_ids:
        node_type_mapping[protein_id] = 'protein'

    #Obtain the community assignment of nodes
    partition = community_louvain.best_partition(nx_graph)

    communities = {}
    for node, community_id in partition.items():
        if community_id not in communities:
            communities[community_id] = []

        node_type = node_type_mapping.get(node, 'unknown')

        communities[community_id].append({
            'id': node,
            'type': node_type
        })

    return communities


# Step 7: Combine association rules with communities
def combine_rules_and_communities(communities, node_features, min_support=0.3, min_threshold=0.7):
    community_rules = {}

    for community_id, nodes in communities.items():
        community_node_features = {}

        for node in nodes:
            node_id = node['id']
            if node_id in node_features:
                community_node_features[node_id] = node_features[node_id]
            else:
                print(f"Warning: Unable to find features for node {node_id}")

        # Generate frequent itemsets and compute association rules
        if community_node_features:
            df = prepare_data_for_fpgrowth(community_node_features)
            frequent_itemsets = apply_fpgrowth(df, min_support)
            rules = generate_association_rules(frequent_itemsets, min_threshold)
            community_rules[community_id] = rules
        else:
            print(f"Community {community_id} has no valid node features.")

    return community_rules


# Step 8: Output the association rules for each community
def output_community_rules(community_rules):
    with open('community_association_rules.txt', 'a', encoding='utf-8') as log_file:
        for community_id, rules in community_rules.items():
            log_file.write(f"{community_id} rules:\n")
            for idx, row in rules.iterrows():
                antecedents = ', '.join(str(i) for i in row['antecedents'])
                consequents = ', '.join(str(i) for i in row['consequents'])
                support = row['support']
                confidence = row['confidence']
                lift = row['lift']
                conviction = row.get('conviction', 'N/A')
                kulczynski = row.get('kulczynski', 'N/A')
                certainty = row.get('certainty', 'N/A')

                log_file.write(f"rule {idx + 1}:\n")
                log_file.write(f"If the following occurs: [{antecedents}]\n")
                log_file.write(f"Then the following may also occur: [{consequents}]\n")
                log_file.write(f"Support: {support:.4f}\n")
                log_file.write(f"Confidence: {confidence:.4f}\n")
                log_file.write(f"Lift: {lift:.4f}\n")
                log_file.write(f"Certainty:{certainty}\n")
                log_file.write(f"Kulczynski：{kulczynski}\n")
                log_file.write(f"Conviction：{conviction}\n")
                log_file.write("-" * 40 + "\n")


def main():

    graph_path = r"data/ImmGraph_results/heterographs_ImmGraph/train_graph_epoch_160.dgl"

    graph = load_graph(graph_path)

    node_features = extract_node_features(graph)

    communities = community_detection(graph)

    community_rules = combine_rules_and_communities(communities, node_features)

    output_community_rules(community_rules)


if __name__ == "__main__":
    main()
