# ImmGraph

This repository provides the code and processed data associated with the manuscript "ImmGraph learns immune interaction topology linking cross-omic regulation to glioma outcome".

ImmGraph is an immunologically informed heterogeneous graph learning framework for integrating genomics, transcriptomics and proteomics to characterize the cross-omic immune interaction network in glioma. The model derives supervision from curated immune marker genes, obviating the need for matched experimental immune phenotyping, and distils the trained graph into a compact core network termed ImmGraph-topo. This refined network achieves prognostic concordance indices up to 0.67, a histopathological subtype classification AUC of 0.93 and biomarker prediction AUCs above 0.85, and identifies a web of candidate modulator genes, including TIMM13, EXOSC8 and GLUL, that were subsequently validated in syngeneic orthotopic mouse models.

## Repository structure

    CODE/Data Preprocess/edgeid_genetation.py      construction of node and edge tables for the heterogeneous graph
    CODE/Data Preprocess/label_generation.py       generation of immune-marker-derived PCA labels for each omic layer
    CODE/Model Training/Immgraph/main.py           training script for ImmGraph (multi-task, five-fold cross-validation)
    CODE/Model Training/Immgraph/model.py          definition of the heterogeneous graph neural network
    CODE/Model Training/Immgraph/best_model.*.rar  trained model weights, distributed across two archives
    CODE/Model Training/Immgraph_topo/             downstream classification and Cox regression models
    CODE/Topology Analysis/                        embedding-based topology refinement and Louvain community detection
    CODE/Visualization/Figure2/Pearson_Figure2_b.R          correlation between predicted and observed RNA marker levels
    CODE/Visualization/Figure2/boxplot.py_Figure2_c.py      immune cell infiltration boxplots across the four subtypes
    CODE/Visualization/Figure3/ROC_draw_Figure3c.py         receiver operating characteristic curves
    CODE/Visualization/Figure3/PR_draw_Figure3d.py          precision-recall curves
    CODE/Visualization/Figure3/km_cutoff_Figure3b.py        risk score cutoff optimization
    CODE/Visualization/Figure3/KM_curve_new_Figure3b.py     Kaplan-Meier survival curves
    CODE/Visualization/figure4/picture_enrich.R    KEGG and Reactome enrichment bubble plot
    CODE/Visualization/Figure7/KM-curve_picture.py          survival stratification under the majority-voting rule
    CODE/Visualization/Figure7/mantal_v5.R                  Mantel test and Pearson correlation heatmap
    DATA/data_for_immgraph/                        node features, edge lists and labels used for training
    DATA/data_for_immgraph_topo/                   per-patient subgraphs for the downstream tasks
    DATA/data_for_running/                         tabular inputs consumed by the visualization scripts
    Environment/enviroment_for_needed.txt          a complete conda environment export
    requirements.txt

## Installation

The code was developed for Python 3.11, PyTorch 2.1.2 and DGL 2.0.0 with CUDA 12.1.

    conda create -n immgraph python=3.11
    conda activate immgraph
    conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
    conda install -c dglteam/label/cu121 dgl
    pip install -r requirements.txt

A fully pinned environment is provided in Environment/enviroment_for_needed.txt, exported from the conda environment named pytorch.

Three visualization scripts are written in R. Figure2/Pearson_Figure2_b.R requires ggplot2; figure4/picture_enrich_Figure4c.R requires clusterProfiler, org.Hs.eg.db, enrichplot and ggnewscale; and Figure7/mantal_v5.R requires ggplot2, vegan, linkET, dplyr and reshape2.

## Usage

Data preprocessing and training:

    cd "CODE/Data Preprocess"
    python edgeid_genetation.py
    python label_generation.py
    cd "../Model Training/Immgraph"
    python main.py

Topology refinement:

    cd "CODE/Topology Analysis"
    python Topological_Data_Anaysis.py
    python Louvain.py

Downstream prediction:

    cd "CODE/Model Training/Immgraph_topo"
    python downstream_regression.py
    python downstream_classification.py

Figure generation:

    cd "CODE/Visualization"
    python Figure3/ROC_draw_Figure3c.py
    python Figure3/PR_draw_Figure3d.py
    python Figure3/km_cutoff_Figure3b.py
    python Figure3/KM_curve_new_Figure3b.py
    python Figure2/boxplot.py_Figure2_c.py
    python Figure7/KM-curve_picture.py
    Rscript Figure2/Pearson_Figure2_b.R
    Rscript figure4/picture_enrich_Figure4c.R
    Rscript Figure7/mantal_v5.R

## Data

data_for_immgraph contains the node tables (nodes_dna.csv, nodes_rna.csv, nodes_protein.csv), the five edge tables (edges_dna.csv, edges_rna.csv, edges_protein.csv, edges_dnarna.csv, edges_rnapro.csv) and the label file Label.csv.

data_for_immgraph_topo contains one DGL subgraph per patient, organized into Discovery, Validation 1 and Validation 2.

data_for_running contains the CSV and XLSX files read by the visualization scripts.

## Data and code availability

Sequencing data are deposited under GSA accession HRA006184 (WES and RNA-seq) and iProX accession PXD062023 (proteomics). The code is available at https://github.com/JXDuan1618/ImmGraph. Correspondence should be addressed to Jingxian Duan at jx.duan@siat.ac.cn.
