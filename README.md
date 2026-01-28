# ImmGraph learns immune interaction topology linking cross-omic regulation to glioma outcome

This repository contains the code for a research paper on multi-omic immunology analysis using heterogeneous graph neural networks. 

## Project Structure

```
capsule/
├── CODE/
│   ├── Data Preprocess/          # Data preprocessing scripts
│   │   ├── edgeid_genetation.py  # Edge generation for graph construction
│   │   └── label_generation.py   # Label generation using PCA
│   ├── Model Training/
│   │   ├── Immgraph/             # Main heterogeneous GNN model
│   │   │   ├── main.py           # Training script for ImmGraph
│   │   │   └── model.py          # Model architecture definition
│   │   └── Immgraph_topo/        # Downstream tasks with topological features
│   │       ├── downstream_classification.py  # Classification tasks
│   │       └── downstream_regression.py       # Regression tasks
│   ├── Topology Analysis/        # Topological data analysis
│   │   ├── Topological_Data_Anaysis.py      # TDA implementation
│   │   └── accosiation_community.py          # Community detection and association rules
│   └── Visualization/           # Visualization scripts for paper figures
│       ├── Figure2/              # Boxplots and correlation plots
│       ├── Figure3/              # ROC, PR, and KM curves
│       ├── figure4/              # Heatmaps
│       └── Figure5/              # Association analysis visualizations
├── DATA/
│   ├── data_for_immgraph/        # Processed data for ImmGraph training
│   └── data_for_immgraph_topo/   # Processed graphs for downstream tasks
└── Environment/
    └── enviroment_for_needed.txt # Environment configuration
```

## Overview

This project implements a heterogeneous graph neural network (ImmGraph) that integrates multi-omic data (DNA mutations, RNA expression, and protein abundance) to predict immunological signatures. The framework includes:

1. **Data Preprocessing**: Normalization, edge construction based on biological pathways (immune pathways) and protein-protein interactions (PPI)
2. **Graph Neural Network Training**: Heterogeneous GNN that learns representations across DNA, RNA, and protein nodes
3. **Downstream Tasks**: Classification and regression tasks using learned graph embeddings
4. **Topological Analysis**: Topological data analysis and community detection
5. **Visualization**: Comprehensive plotting scripts for publication-quality figures

## Key Features

- **Multi-omic Integration**: Simultaneously processes DNA, RNA, and protein data
- **Heterogeneous Graph Neural Networks**: Uses DGL to model complex relationships between different omic layers
- **Cross-validation**: 5-fold cross-validation for robust model evaluation
- **Hyperparameter Optimization**: Optuna-based Bayesian optimization
- **Topological Analysis**: Integration of topological data analysis for feature extraction

## Requirements

### Python Dependencies

See `requirements.txt` for the complete list of Python packages. Key dependencies include:

- **PyTorch** (>=2.1.2) with CUDA support
- **DGL** (>=2.0.0) for graph neural networks
- **NumPy**, **Pandas** for data manipulation
- **Scikit-learn** for machine learning utilities
- **Lifelines** for survival analysis
- **Optuna** for hyperparameter optimization
- **NetworkX** for graph analysis
- **Matplotlib**, **Seaborn** for visualization
- **Giotto-tda** for topological data analysis

### R Dependencies (for some visualizations)

Some visualization scripts require R with the following packages:
- `ggplot2`
- `survival`
- `survminer`

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd capsule
```

2. **Create a conda environment** (recommended):
```bash
conda create -n immgraph python=3.11
conda activate immgraph
```

3. **Install PyTorch with CUDA support**:
```bash
# For CUDA 12.1 (adjust based on your system)
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

4. **Install DGL**:
```bash
# For CUDA 12.1
conda install -c dglteam/label/cu121 dgl
```

5. **Install remaining Python packages**:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing

First, prepare your multi-omic data and run preprocessing scripts:

```bash
cd "CODE/Data Preprocess"
python edgeid_genetation.py
python label_generation.py
```

This will generate:
- Node files: `nodes_dna.csv`, `nodes_rna.csv`, `nodes_protein.csv`
- Edge files: `edges_dna.csv`, `edges_rna.csv`, `edges_protein.csv`, `edges_dnarna.csv`, `edges_rnapro.csv`
- Label file: `Label.csv` with PCA-derived immunological signatures

### 2. Train ImmGraph Model

Train the main heterogeneous graph neural network:

```bash
cd "CODE/Model Training/Immgraph"
python main.py
```

The script performs 5-fold cross-validation and saves:
- Trained models
- Node embeddings
- Patient predictions
- Training metrics


### 3. Topological Analysis

Perform topological data analysis:

```bash
cd "CODE/Topology Analysis"
python Topological_Data_Anaysis.py
python accosiation_community.py
```
### 4. Downstream Tasks

Run classification or regression tasks using learned embeddings:

```bash
cd "CODE/Model Training/Immgraph_topo"
# For classification
python downstream_classification.py

# For regression
python downstream_regression.py
```



### 5. Generate Visualizations

Create publication-quality figures:

```bash
cd "CODE/Visualization"
# ROC curves
python Figure3/ROC_draw.py

# Precision-Recall curves
python Figure3/PR_draw.py

# Kaplan-Meier curves
python Figure3/KM_curve_new.py

# Heatmaps
python figure4/heatmap_original.py
python Figure5/heatmap_star.py
```

## Data Format

### Input Data

- **DNA data**: CSV file with genes as rows and patients as columns
- **RNA data**: CSV file with genes as rows and patients as columns (log-transformed FPKM values)
- **Protein data**: CSV file with proteins as rows and patients as columns
- **PPI network**: CSV file with protein-protein interactions
- **KEGG pathways**: CSV file with pathway-gene associations

### Graph Structure

The heterogeneous graph contains:
- **Node types**: `dna`, `rna`, `protein`
- **Edge types**:
  - `dna_interact`: DNA-DNA interactions (immune pathways)
  - `rna_interact`: RNA-RNA interactions (immune pathways)
  - `pro_interact`: Protein-protein interactions (PPI)
  - `transcribe`: DNA to RNA connections (same genes)
  - `translate`: RNA to protein connections (same genes)

## Model Architecture

The ImmGraph model consists of:

1. **Heterogeneous RGCN Layers**: Learn node representations by aggregating information from neighbors
2. **Multi-branch Classifier**: Separate branches for DNA, RNA, and protein with residual connections
3. **Attention Mechanism**: Edge weight learning for different edge types

## Output Files

The training process generates:

- **Model checkpoints**: Best models for each fold
- **Node embeddings**: Learned representations for each node type
- **Patient predictions**: Immunological signature predictions
- **Metrics**: CSV files with performance metrics (MAE, RMSE, R², etc.)
- **Graphs**: Saved DGL graphs with learned edge weights

## Configuration

Key hyperparameters can be adjusted in the training scripts:

- Learning rate: `1e-4`
- Batch size: `200` (for ImmGraph), `16` (for downstream tasks)
- Number of epochs: `400` (ImmGraph), `150` (downstream)
- Hidden dimensions: `1024` → `512` → `256`
- Dropout: `0.3-0.45`

## Citation

If you use this code in your research, please cite:

```
Jingxian Duan (jx.duan@siat.ac.cn)
```

## License

[License information to be added]

## Contact

[Contact information to be added]

## Acknowledgments

This project uses the following open-source libraries:
- DGL (Deep Graph Library)
- PyTorch
- Giotto-tda
- NetworkX
- And other libraries listed in requirements.txt
