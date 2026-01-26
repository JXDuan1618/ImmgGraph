# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:20:35 2023

@author: D
"""

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

# Load the CSV files containing patient data into Pandas dataframes
####################################################################################
train_df = pd.read_csv(r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\figure_3_A\prognosis\KM_curve\correct_Discovery\fold_5_train_risk_ZZU.csv')
train_df["split"]="train"
train_df = train_df.rename(columns={'Risk': 'risk_score', 'Time': 'overall_survival', "Event": "vital_status"})
###############################################################################
val_df = pd.read_csv(r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\figure_3_A\prognosis\KM_curve\correct_Discovery\fold_5_val_risk_ZZU.csv')
val_df["split"]="valid"
val_df = val_df.rename(columns={'Risk': 'risk_score', 'Time': 'overall_survival', "Event": "vital_status"})
# Merge the training and validation dataframes into one dataframe
merged_df = pd.concat([train_df, val_df])

# Sort the merged dataframe by the patient risk scores in ascending order
merged_df = merged_df.sort_values('risk_score')

# Create an empty list to store all the possible cutoffs
cutoffs = []

# Use a for loop to iterate through all the unique patient risk scores in the merged dataframe
for score in merged_df['risk_score'].unique():
    # Append each unique risk score to the cutoff list
    cutoffs.append(score)

# Create a function to assign each patient to the high or low risk group based on the cutoff
def assign_risk_group(score, cutoff):
    if score >= cutoff:
        return 'high risk'
    else:
        return 'low risk'

# Create a new column in the merged dataframe to indicate whether each patient is in the high risk or low risk group based on the cutoff
for cutoff in cutoffs:
    merged_df[f'risk_group_{cutoff}'] = merged_df['risk_score'].apply(assign_risk_group, cutoff=cutoff)

# Split the merged dataframe back into the training and validation datasets
train_df = merged_df[merged_df['split'] == 'train']
val_df = merged_df[merged_df['split'] == 'valid']

# Create an empty dictionary to store the log-rank test results
logrank_resultt = {}
logrank_resultv = {}
# Define the columns in the dataframes that contain patient risk group, overall survival, and vital status
risk_group_col = 'risk_group'
survival_col = 'overall_survival'
vital_status_col = 'vital_status'

cutvalue=[]
p_valuet=[]
# Use the Kaplan-Meier estimator to estimate the survival curves for each risk group in the training dataset
# kmf_train = KaplanMeierFitter()
for cutoff in cutoffs:
    risk_group = train_df[f'risk_group_{cutoff}']
    # kmf_train.fit(train_df[survival_col][risk_group == 'high risk'], train_df[vital_status_col][risk_group == 'high risk'], label=f'High Risk (Cutoff={cutoff:.2f})')
    # kmf_train.fit(train_df[survival_col][risk_group == 'low risk'], train_df[vital_status_col][risk_group == 'low risk'], label=f'Low Risk (Cutoff={cutoff:.2f})')
    # # Use the log-rank test to compare the survival curves of the high and low risk groups in the training dataset
    results = logrank_test(train_df[survival_col][risk_group == 'high risk'], train_df[survival_col][risk_group == 'low risk'], train_df[vital_status_col][risk_group == 'high risk'], train_df[vital_status_col][risk_group == 'low risk'])
    p_valuett = results.p_value

    logrank_resultt[f'training_{cutoff:.7f}'] = p_valuett

# Use the Kaplan-Meier estimator to estimate the survival curves for each risk group in the validation dataset
#kmf_val = KaplanMeierFitter()
p_valuev=[]
for cutoff in cutoffs:
    risk_group = val_df[f'risk_group_{cutoff}']
    # kmf_val.fit(val_df[survival_col][risk_group == 'high risk'], val_df[vital_status_col][risk_group == 'high risk'], label=f'High Risk (Cutoff={cutoff:.2f})')
    # kmf_val.fit(val_df[survival_col][risk_group == 'low risk'], val_df[vital_status_col][risk_group == 'low risk'], label=f'Low Risk (Cutoff={cutoff:.2f})')
    
    # Use the log-rank test to compare the survival curves of the high and low risk groups in the validation dataset
    results = logrank_test(val_df[survival_col][risk_group == 'high risk'], val_df[survival_col][risk_group == 'low risk'], val_df[vital_status_col][risk_group == 'high risk'], val_df[vital_status_col][risk_group == 'low risk'])
    p_valuevv = results.p_value

    logrank_resultv[f'validation_{cutoff:.7f}'] = p_valuevv

# Save the log-rank test results for both the training and validation datasets to a CSV file

logrank_dft = pd.DataFrame.from_dict(logrank_resultt, orient='index', columns=['p-value'])
logrank_dfv = pd.DataFrame.from_dict(logrank_resultv, orient='index', columns=['p-value'])
#########################################################################
logrank_df=pd.concat([logrank_dft, logrank_dfv], axis=1)
logrank_df.to_csv(r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\figure_3_A\prognosis\KM_curve\correct_Discovery\P_OS_ZZU_fold_5.csv')


'''train_risk_scores = train_df['risk_score']
train_overall_survival = train_df['overall_survival']
train_vital_status = train_df['vital_status']

val_risk_scores = val_df['risk_score']
val_overall_survival = val_df['overall_survival']
val_vital_status = val_df['vital_status']

# Determine the cutoff for high and low risk groups
#######################################################
cutoff = 0.2063524
train_high_risk = train_risk_scores >= cutoff
train_low_risk = train_risk_scores < cutoff
val_high_risk = val_risk_scores >= cutoff
val_low_risk = val_risk_scores < cutoff

# Create Kaplan-Meier fitter objects for the high and low risk groups
kmf_train_high = KaplanMeierFitter()
kmf_train_low = KaplanMeierFitter()
kmf_val_high = KaplanMeierFitter()
kmf_val_low = KaplanMeierFitter()

# Fit the KM models to the high and low risk groups in the training and validation data
kmf_train_high.fit(train_overall_survival[train_high_risk], train_vital_status[train_high_risk], timeline=train_overall_survival)
kmf_train_low.fit(train_overall_survival[train_low_risk], train_vital_status[train_low_risk], timeline=train_overall_survival)
kmf_val_high.fit(val_overall_survival[val_high_risk], val_vital_status[val_high_risk], timeline=val_overall_survival)
kmf_val_low.fit(val_overall_survival[val_low_risk], val_vital_status[val_low_risk], timeline=val_overall_survival)

# Plot the KM curves for the high and low risk groups in the training and validation data
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True)
kmf_train_high.plot(ax=ax1, label='Training high risk')
kmf_train_low.plot(ax=ax1, label='Training low risk')
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Survival probability')
ax1.set_title('Training data')
ax1.legend()
kmf_val_high.plot(ax=ax2, label='Validation high risk')
kmf_val_low.plot(ax=ax2, label='Validation low risk')
ax2.set_xlabel('Time (days)')
ax2.set_title('Validation data')
ax2.legend()


####################################################
plt.savefig('F:/GCN_multiomic/output_lgg/km_plot_f2S140.pdf')
plt.show()'''
