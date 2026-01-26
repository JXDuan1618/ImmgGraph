# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:17:19 2024

@author: D
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import matplotlib.pyplot as plt

# Step 1: Load the CSV files into DataFrames
immune_marker_df = pd.read_csv('data/raw_data_discovery/immune_marker.csv')
rna_df = pd.read_csv(r'data/raw_data_discovery/05_FPKM_339_new_normalized.csv')
protein_df = pd.read_csv(r'data/raw_data_discovery/06_Protein_304_new_normalized.csv')
mutation_df = pd.read_csv(r'data/raw_data_discovery/09_DNA_all_306_new_normalized.csv')
# Step 2: Extract the list of gene names from the immune_marker DataFrame
gene_names = immune_marker_df['Gene Names'].tolist()

# Step 3: Filter the rna DataFrame to keep only the rows with matching gene names
filtered_rna_df = rna_df[rna_df['Hugo_Symbol'].isin(gene_names)]
filtered_protein_df = protein_df[protein_df['Genes'].isin(gene_names)]
filtered_mutation_df = mutation_df[mutation_df['Genes'].isin(gene_names)]

# Rename the column 'Hugo_Symbol' to 'Genes' in filtered_rna_df
filtered_rna_df = filtered_rna_df.rename(columns={'Hugo_Symbol': 'Genes'})
# Assume the filtered DataFrames have been loaded or created as filtered_rna_df, filtered_protein_df, filtered_mutation_df

def perform_pca_on_patients(df):
    # Exclude non-numeric columns (like gene names)
    numeric_df = df.drop(columns=['Genes'])  # Change to 'Genes' if already renamed
    
    # Transpose the DataFrame so that rows become patients and columns become genes
    transposed_df = numeric_df.transpose()
    
    # Get the original column names (patient identifiers)
    patient_names = transposed_df.index.tolist()
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(transposed_df)
    
    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)
    
    # Create a DataFrame for the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=patient_names)
    
    return pca_df

# Perform PCA on the filtered RNA, Protein, and Mutation DataFrames
pca_rna_df = perform_pca_on_patients(filtered_rna_df)
pca_protein_df = perform_pca_on_patients(filtered_protein_df)
pca_mutation_df = perform_pca_on_patients(filtered_mutation_df)

# Cross-reference patients (keep only patients that exist in all three files)
common_patients = pca_rna_df.index.intersection(pca_protein_df.index).intersection(pca_mutation_df.index)

filtered_rna_df = pca_rna_df.loc[common_patients]
filtered_protein_df = pca_protein_df.loc[common_patients]
filtered_mutation_df = pca_mutation_df.loc[common_patients]

# Rename columns to specify their origin
filtered_rna_df.columns = ['RNA_PC1', 'RNA_PC2']
filtered_protein_df.columns = ['Protein_PC1', 'Protein_PC2']
filtered_mutation_df.columns = ['Mutation_PC1', 'Mutation_PC2']

# Calculate Z-scores for the three files
zscored_rna_df = filtered_rna_df.apply(zscore)
zscored_protein_df = filtered_protein_df.apply(zscore)
zscored_mutation_df = filtered_mutation_df.apply(zscore)

# Concatenate the columns from the three DataFrames into one
final_concatenated_df = pd.concat([zscored_rna_df, zscored_protein_df, zscored_mutation_df], axis=1)
# Save the final concatenated DataFrame to a CSV file
final_concatenated_df.to_csv(r'data/processed_data_discovery/im.csv')