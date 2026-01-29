# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 09:56:02 2023

@author: D
"""

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
from lifelines.plotting import add_at_risk_counts
# from sklearn.metrics import brier_score_loss
import matplotlib as mpl

# Set the font to Arial
mpl.rcParams['font.family'] = 'Arial'
# Load the CSV files containing patient data into Pandas dataframes
####################################################################################
train_df = pd.read_csv(r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\figure_5\KM_curve\PFS\TIMM13\TIMM13_train.csv')
train_df["split"] = "train"
train_df = train_df.rename(
    columns={'Risk': 'risk_score', 'Time': 'overall_survival', "Event": "vital_status"})
###############################################################################
val_df = pd.read_csv(r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\figure_5\KM_curve\PFS\TIMM13\TIMM13_valid.csv')
val_df["split"] = "valid"
val_df = val_df.rename(
    columns={'Risk': 'risk_score', 'Time': 'overall_survival', "Event": "vital_status"})
# Merge the training and validation dataframes into one dataframe
# merged_df = pd.concat([train_df, val_df])

# Sort the merged dataframe by the patient risk scores in ascending order
# merged_df = merged_df.sort_values('risk_score')

# Create an empty list to store all the possible cutoffs
# cutoffs = []

# Use a for loop to iterate through all the unique patient risk scores in the merged dataframe
# for score in merged_df['risk_score'].unique():
#     # Append each unique risk score to the cutoff list
#     cutoffs.append(score)

# # Create a function to assign each patient to the high or low risk group based on the cutoff
# def assign_risk_group(score, cutoff):
#     if score >= cutoff:
#         return 'high risk'
#     else:
#         return 'low risk'

# # Create a new column in the merged dataframe to indicate whether each patient is in the high risk or low risk group based on the cutoff
# for cutoff in cutoffs:
#     merged_df[f'risk_group_{cutoff}'] = merged_df['risk_score'].apply(assign_risk_group, cutoff=cutoff)

# # Split the merged dataframe back into the training and validation datasets
# train_df = merged_df[merged_df['split'] == 'train']
# val_df = merged_df[merged_df['split'] == 'valid']

# # Create an empty dictionary to store the log-rank test results
# logrank_resultt = {}
# logrank_resultv = {}
# # Define the columns in the dataframes that contain patient risk group, overall survival, and vital status
# risk_group_col = 'risk_group'
# survival_col = 'overall_survival'
# vital_status_col = 'vital_status'

# cutvalue=[]
# p_valuet=[]
# # Use the Kaplan-Meier estimator to estimate the survival curves for each risk group in the training dataset
# # kmf_train = KaplanMeierFitter()
# for cutoff in cutoffs:
#     risk_group = train_df[f'risk_group_{cutoff}']
#     # kmf_train.fit(train_df[survival_col][risk_group == 'high risk'], train_df[vital_status_col][risk_group == 'high risk'], label=f'High Risk (Cutoff={cutoff:.2f})')
#     # kmf_train.fit(train_df[survival_col][risk_group == 'low risk'], train_df[vital_status_col][risk_group == 'low risk'], label=f'Low Risk (Cutoff={cutoff:.2f})')
#     # # Use the log-rank test to compare the survival curves of the high and low risk groups in the training dataset
#     results = logrank_test(train_df[survival_col][risk_group == 'high risk'], train_df[survival_col][risk_group == 'low risk'], train_df[vital_status_col][risk_group == 'high risk'], train_df[vital_status_col][risk_group == 'low risk'])
#     p_valuett = results.p_value

#     logrank_resultt[f'training_{cutoff:.7f}'] = p_valuett

# # Use the Kaplan-Meier estimator to estimate the survival curves for each risk group in the validation dataset
# #kmf_val = KaplanMeierFitter()
# p_valuev=[]
# for cutoff in cutoffs:
#     risk_group = val_df[f'risk_group_{cutoff}']
#     # kmf_val.fit(val_df[survival_col][risk_group == 'high risk'], val_df[vital_status_col][risk_group == 'high risk'], label=f'High Risk (Cutoff={cutoff:.2f})')
#     # kmf_val.fit(val_df[survival_col][risk_group == 'low risk'], val_df[vital_status_col][risk_group == 'low risk'], label=f'Low Risk (Cutoff={cutoff:.2f})')

#     # Use the log-rank test to compare the survival curves of the high and low risk groups in the validation dataset
#     results = logrank_test(val_df[survival_col][risk_group == 'high risk'], val_df[survival_col][risk_group == 'low risk'], val_df[vital_status_col][risk_group == 'high risk'], val_df[vital_status_col][risk_group == 'low risk'])
#     p_valuevv = results.p_value

#     logrank_resultv[f'validation_{cutoff:.7f}'] = p_valuevv

# # Save the log-rank test results for both the training and validation datasets to a CSV file

# logrank_dft = pd.DataFrame.from_dict(logrank_resultt, orient='index', columns=['p-value'])
# logrank_dfv = pd.DataFrame.from_dict(logrank_resultv, orient='index', columns=['p-value'])
# #########################################################################
# logrank_df=pd.concat([logrank_dft, logrank_dfv], axis=1)
# logrank_df.to_csv('F:/GCN_multiomic/output_pcc/logrank_resultf1s78.csv')


train_risk_scores = train_df['risk_score']
train_overall_survival = train_df['overall_survival']
train_vital_status = train_df['vital_status']

# y_true =train_vital_status
# y_prob =train_risk_scores
# BS=brier_score_loss(y_true, y_prob)
# print("BS",BS)

val_risk_scores = val_df['risk_score']
val_overall_survival = val_df['overall_survival']
val_vital_status = val_df['vital_status']

# Determine the cutoff for high and low risk groups
#######################################################
cutoff_os = -0.7290039
cutoff_dfs = -0.7290039
train_high_risk = train_risk_scores >= cutoff_os
train_low_risk = train_risk_scores < cutoff_os
val_high_risk = val_risk_scores >= cutoff_dfs
val_low_risk = val_risk_scores < cutoff_dfs

train_df["risk_group"] = np.where(train_df["risk_score"] >= cutoff_os, "High-risk", "Low-risk")
val_df["risk_group"] = np.where(val_df["risk_score"] >= cutoff_dfs, "High-risk", "Low-risk")

# ----------------------------------------------------------
# 导出结果到 CSV 文件，方便查看每个病人的分组情况
# ----------------------------------------------------------
train_output_path = r"E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\figure_5\KM_curve\PFS\TIMM13\train_with_risk_group.csv"
val_output_path   = r"E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\figure_5\KM_curve\PFS\TIMM13\val_with_risk_group.csv"

train_df.to_csv(train_output_path, index=False)
val_df.to_csv(val_output_path, index=False)

print("训练集与验证集的分组结果已导出：")
print(train_output_path)
print(val_output_path)

# Create Kaplan-Meier fitter objects for the high and low risk groups
kmf_train_high = KaplanMeierFitter()
kmf_train_low = KaplanMeierFitter()
kmf_val_high = KaplanMeierFitter()
kmf_val_low = KaplanMeierFitter()

# Fit the KM models to the high and low risk groups in the training and validation data

kmf_train_high.fit(train_overall_survival[train_high_risk], train_vital_status[train_high_risk])
kmf_train_low.fit(train_overall_survival[train_low_risk], train_vital_status[train_low_risk])
kmf_val_high.fit(val_overall_survival[val_high_risk], val_vital_status[val_high_risk])
kmf_val_low.fit(val_overall_survival[val_low_risk], val_vital_status[val_low_risk])

resultt = logrank_test(train_overall_survival[train_high_risk], train_overall_survival[train_low_risk],
                       train_vital_status[train_high_risk], train_vital_status[train_low_risk])
p_valuett = resultt.p_value
if p_valuett < 0.0001:
    p_valuett = 0.0001
results = logrank_test(val_overall_survival[val_high_risk], val_overall_survival[val_low_risk],
                       val_vital_status[val_high_risk], val_vital_status[val_low_risk])
p_valuets = results.p_value
if p_valuets < 0.0001:
    p_valuets = 0.0001
# Force the curves to start at 1 (if necessary)
kmf_val_high.survival_function_.iloc[0, 0] = 1.0
kmf_val_low.survival_function_.iloc[0, 0] = 1.0
# Plot the KM curves for the high and low risk groups in the training and validation data
# Plot the KM curves for the high and low risk groups in the training and validation data
fig, ax1 = plt.subplots(figsize=(12, 8))

kmf_train_high.plot_survival_function(ax=ax1, label='High-risk Group', ci_show=True, ci_alpha=0.2, show_censors=True,
                                    censor_styles={'marker': '|', 'ms': 12, 'mew': 3},
                                    linewidth=6, color="#4d92c3", fontsize=55)

kmf_train_low.plot_survival_function(ax=ax1, label='Low-risk Group', ci_show=True, ci_alpha=0.2, show_censors=True,
                                   censor_styles={'marker': '|', 'ms': 12, 'mew': 3},
                                   linewidth=6, color="#ff9e4a", fontsize=55)

# Add "at risk" counts manually
add_at_risk_counts(kmf_train_high, kmf_train_low, ax=ax1, rows_to_show=['At risk'], labels=['High Risk', 'Low Risk'],
                   fontsize=60)

ax1.set_xlabel('Time (Months)', fontsize=60, labelpad=150)
ax1.set_ylabel('Survival Probability', fontsize=60, labelpad=30)
ax1.set_title('Training ', fontsize=60, pad=30)
ax1.annotate(f'p= {p_valuett:.4f}', xy=(0.5, 0.8), xycoords='axes fraction', fontsize=60,bbox=dict(facecolor='white', alpha=0.7, edgecolor='white', boxstyle='round,pad=0.3'))

# Customize the border of the plot
for spine in ax1.spines.values():
    spine.set_linewidth(6)

# Increase tick font size and make the ticks larger
ax1.tick_params(axis='both', which='major', labelsize=60, length=15, width=6)

# Remove legend
ax1.legend().set_visible(False)
plt.subplots_adjust(bottom=0.1)
plt.savefig(r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\figure_5\KM_curve\PFS\TIMM13\kaplan_meier_train.png', bbox_inches='tight')
#plt.show()

fig, ax2 = plt.subplots(figsize=(12, 8))
kmf_val_high.plot_survival_function(ax=ax2, label='High-risk Group', ci_show=True, ci_alpha=0.2, show_censors=True,
                                    censor_styles={'marker': '|', 'ms': 12, 'mew': 3},
                                    linewidth=6, color="#4d92c3", fontsize=55)
kmf_val_low.plot_survival_function(ax=ax2, label='Low-risk Group', ci_show=True, ci_alpha=0.2, show_censors=True,
                                   censor_styles={'marker': '|', 'ms': 12, 'mew': 3},
                                   linewidth=6, color="#ff9e4a", fontsize=55)

# Add "at risk" counts manually
add_at_risk_counts(kmf_val_high, kmf_val_low, ax=ax2, rows_to_show=['At risk'], labels=['High Risk', 'Low Risk'],
                   fontsize=60)

ax2.set_xlabel('Time (Months)', fontsize=60, labelpad=150)
ax2.set_ylabel('Survival Probability', fontsize=60, labelpad=30)
ax2.set_title('Validation', fontsize=60, pad=30)
ax2.annotate(f'p= {p_valuets:.4f}', xy=(0.5, 0.8), xycoords='axes fraction', fontsize=60,bbox=dict(facecolor='white', alpha=0.7, edgecolor='white', boxstyle='round,pad=0.3'))

# Customize the border of the plot
for spine in ax2.spines.values():
    spine.set_linewidth(6)

# Increase tick font size and make the ticks larger
ax2.tick_params(axis='both', which='major', labelsize=60, length=15, width=6)

# Remove legend
ax2.legend().set_visible(False)
#ax2.legend(loc='upper left', fontsize=60, bbox_to_anchor=(1, 1))

plt.subplots_adjust(bottom=0.1)
plt.savefig(r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\figure_5\KM_curve\PFS\TIMM13\kaplan_meier_valid.png', bbox_inches='tight')
#plt.show()
