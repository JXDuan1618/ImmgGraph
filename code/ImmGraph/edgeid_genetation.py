# -*- coding: utf-8 -*-
"""
Created on Fri May 19 17:34:25 2023
@author: D
"""

import pandas as pd
import numpy as np
from pandas import DataFrame
from scipy import stats

# === Load data ===
rna = pd.read_csv(r"data/raw_data_discovery/05_FPKM_339_logtransformed.csv", sep=',', header=0, index_col=0)
dna = pd.read_csv(r"data/raw_data_discovery/09_DNA_all_306_new.csv", sep=',', header=0, index_col=0)
protein = pd.read_csv(r"data/raw_data_discovery/06_Protein_304_new.csv", sep=',', header=0, index_col=0)
PPIpro = pd.read_csv(r"data/raw_data_discovery/string_linkpro_202507.csv", sep=',', header=0)
im = pd.read_csv(r"data/processed_data_discovery/im.csv", sep=',', header=0, index_col=0)

# === Preprocess ===
cleanpro = protein.dropna(thresh=len(protein.columns) * 0.8)
cleanrna = rna.dropna(thresh=len(rna.columns) * 0.8)
if not cleanrna.index.is_unique:
    cleanrna = cleanrna.loc[~cleanrna.index.duplicated(keep='first')]
if not cleanpro.index.is_unique:
    cleanpro = cleanpro.loc[~cleanpro.index.duplicated(keep='first')]

patient = im.index
proteinsel = cleanpro.loc[cleanpro.index, list(patient)]
rnasel = cleanrna.loc[cleanrna.index, list(patient)]
dnasel = dna.loc[dna.index, list(patient)]

rnaZ = pd.DataFrame(stats.zscore(rnasel.values, axis=None), index=rnasel.index, columns=rnasel.columns)
proZ = pd.DataFrame(stats.zscore(proteinsel.values, axis=None), index=proteinsel.index, columns=proteinsel.columns)
dnaZ = pd.DataFrame(stats.zscore(dnasel.values, axis=None), index=dnasel.index, columns=dnasel.columns)

rnaZ.insert(0, "node_id", np.arange(rnasel.shape[0]))
proZ.insert(0, "node_id", np.arange(proteinsel.shape[0]))
dnaZ.insert(0, "node_id", np.arange(dnasel.shape[0]))
rnasel = rnaZ
proteinsel = proZ
dnasel = dnaZ

rnasel.to_csv(r"data/processed_data_discovery/nodes_rna.csv")
proteinsel.to_csv(r"data/processed_data_discovery/nodes_protein.csv")
dnasel.to_csv(r"data/processed_data_discovery/nodes_dna.csv")


imgeneset1 = pd.read_csv(r"data/processed_data_discovery/imgeneset.csv", sep=',', header=0)
imgeneset = imgeneset1[imgeneset1['gene'].isin(proteinsel.index)]

DF = rnasel.reset_index()
DFpro = proteinsel.reset_index()
DFdna = dnasel.reset_index()

iddict = DF[["Hugo_Symbol", "node_id"]].set_index("Hugo_Symbol").to_dict(orient="dict")["node_id"]
iddictpro = DFpro[["Genes", "node_id"]].set_index("Genes").to_dict(orient="dict")["node_id"]
iddictdna = DFdna[["Genes", "node_id"]].set_index("Genes").to_dict(orient="dict")["node_id"]

# --- RNA KEGG edges ---
group = imgeneset.groupby("term")
path = list(group.groups.keys())
valid_rna_genes = set(DF["Hugo_Symbol"])

src_id, dst_id = [], []
for pa in path:
    pagenes = set(group.get_group(pa)["gene"]).intersection(valid_rna_genes)
    pagenes = list(pagenes)
    for i in range(len(pagenes)):
        for j in range(i + 1, len(pagenes)):
            src_id.append(iddict[pagenes[i]])
            dst_id.append(iddict[pagenes[j]])
edges_rna = DataFrame({"srcrna_id": src_id, "dstrna_id": dst_id}).drop_duplicates()
edges_rna.to_csv(r"data/processed_data_discovery/edges_rna.csv", index=None)

# --- Protein PPI edges ---
condition = PPIpro['node1'].isin(proteinsel.index) & PPIpro['node2'].isin(proteinsel.index)
PPIpro1 = PPIpro[condition]
proid1 = [iddictpro[n1] for n1 in PPIpro1["node1"]]
proid2 = [iddictpro[n2] for n2 in PPIpro1["node2"]]
edgepro = DataFrame({"srcpro_id": proid1, "dstpro_id": proid2})
edgepro.to_csv(r"data/processed_data_discovery/edges_protein.csv", index=None)

# --- DNA KEGG edges ---
valid_dna_genes = set(DFdna["Genes"])
srcdna_id, dstdna_id = [], []
for pa in path:
    pagenes = set(group.get_group(pa)["gene"]).intersection(valid_dna_genes)
    pagenes = list(pagenes)
    for i in range(len(pagenes)):
        for j in range(i + 1, len(pagenes)):
            srcdna_id.append(iddictdna[pagenes[i]])
            dstdna_id.append(iddictdna[pagenes[j]])
edges_dna = DataFrame({"srcdna_id": srcdna_id, "dstdna_id": dstdna_id}).drop_duplicates()
edges_dna.to_csv(r"data/processed_data_discovery/edges_dna.csv", index=None)


valid_pro_genes = set(DFpro["Genes"])
commongene_rnapro = valid_rna_genes.intersection(valid_pro_genes)
edge2 = DataFrame({
    "rna_id": [iddict[g] for g in commongene_rnapro],
    "pro_id": [iddictpro[g] for g in commongene_rnapro]
})
edge2.to_csv(r"data/processed_data_discovery/edges_rnapro.csv", index=None)

# --- DNA-RNA edges ---
valid_dna_genes = set(DFdna["Genes"])
commongene_dnarna = valid_dna_genes.intersection(valid_rna_genes)
edge3 = DataFrame({
    "dna_id": [iddictdna[g] for g in commongene_dnarna],
    "rna1_id": [iddict[g] for g in commongene_dnarna]
})
edge3.to_csv(r"data/processed_data_discovery/edges_dnarna.csv", index=None)

# === 只保留有边节点 + 连续编号 ===

# === RNA ===
rna_nodes_with_edge = set(edges_rna['srcrna_id']).union(
    edges_rna['dstrna_id'],
    edge2['rna_id'],
    edge3['rna1_id']
)
rna_used = rnasel[rnasel['node_id'].isin(rna_nodes_with_edge)].copy().reset_index()
rna_used['new_node_id'] = range(len(rna_used))
rna_id_map = dict(zip(rna_used['node_id'], rna_used['new_node_id']))

edges_rna = edges_rna.copy()
edges_rna['srcrna_id'] = edges_rna['srcrna_id'].map(rna_id_map)
edges_rna['dstrna_id'] = edges_rna['dstrna_id'].map(rna_id_map)
edge2['rna_id'] = edge2['rna_id'].map(rna_id_map)
edge3['rna1_id'] = edge3['rna1_id'].map(rna_id_map)

# 插入 serial_id：与 node_id 对齐
rna_used['serial_id'] = rna_used['new_node_id'] + 1
rna_used.insert(
    loc=rna_used.columns.get_loc("Hugo_Symbol") + 1,
    column="serial_id",
    value=rna_used.pop('serial_id')
)

rna_used = rna_used.drop(columns=['node_id']).rename(columns={'new_node_id': 'node_id'})
rna_used.to_csv(r"data/processed_data_discovery/nodes_rna.csv", index=False)


# === Protein ===
protein_nodes_with_edge = set(edgepro['srcpro_id']).union(
    edgepro['dstpro_id'],
    edge2['pro_id']
)
protein_used = proteinsel[proteinsel['node_id'].isin(protein_nodes_with_edge)].copy().reset_index()
protein_used['new_node_id'] = range(len(protein_used))
protein_id_map = dict(zip(protein_used['node_id'], protein_used['new_node_id']))

edgepro = edgepro.copy()
edgepro['srcpro_id'] = edgepro['srcpro_id'].map(protein_id_map)
edgepro['dstpro_id'] = edgepro['dstpro_id'].map(protein_id_map)
edge2['pro_id'] = edge2['pro_id'].map(protein_id_map)

# 插入 serial_id
protein_used['serial_id'] = protein_used['new_node_id'] + 1
protein_used.insert(
    loc=protein_used.columns.get_loc("Genes") + 1,
    column="serial_id",
    value=protein_used.pop('serial_id')
)

protein_used = protein_used.drop(columns=['node_id']).rename(columns={'new_node_id': 'node_id'})
protein_used.to_csv(r"data/processed_data_discovery/nodes_protein.csv", index=False)


# === DNA ===
dna_nodes_with_edge = set(edges_dna['srcdna_id']).union(
    edges_dna['dstdna_id'],
    edge3['dna_id']
)
dna_used = dnasel[dnasel['node_id'].isin(dna_nodes_with_edge)].copy().reset_index()
dna_used['new_node_id'] = range(len(dna_used))
dna_id_map = dict(zip(dna_used['node_id'], dna_used['new_node_id']))

edges_dna = edges_dna.copy()
edges_dna['srcdna_id'] = edges_dna['srcdna_id'].map(dna_id_map)
edges_dna['dstdna_id'] = edges_dna['dstdna_id'].map(dna_id_map)
edge3['dna_id'] = edge3['dna_id'].map(dna_id_map)

# 插入 serial_id
dna_used['serial_id'] = dna_used['new_node_id'] + 1
dna_used.insert(
    loc=dna_used.columns.get_loc("Genes") + 1,
    column="serial_id",
    value=dna_used.pop('serial_id')
)

dna_used = dna_used.drop(columns=['node_id']).rename(columns={'new_node_id': 'node_id'})
dna_used.to_csv(r"data/processed_data_discovery/nodes_dna.csv", index=False)

# --- 保存所有更新后的 edges ---
edges_rna.to_csv(r"data/processed_data_discovery/edges_rna.csv", index=False)
edgepro.to_csv(r"data/processed_data_discovery/edges_protein.csv", index=False)
edges_dna.to_csv(r"data/processed_data_discovery/edges_dna.csv", index=False)
edge2.to_csv(r"data/processed_data_discovery/edges_rnapro.csv", index=False)
edge3.to_csv(r"data/processed_data_discovery/edges_dnarna.csv", index=False)




