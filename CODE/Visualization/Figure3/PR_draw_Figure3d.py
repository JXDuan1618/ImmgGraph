# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import pandas as pd
import numpy as np
import matplotlib as mpl

# Global font: Arial
mpl.rcParams['font.family'] = 'Arial'

# Output path (change to the PR filename)
output_path = r"/results/TERT_mutation\pr_TERT_mutation_new.png"

# Load data for each fold (columns: y_true, y_prob)
fold_1 = pd.read_csv(r'/data/data_for_running/Figure3/fold_1_val_pred_TERT_mutation.csv')
fold_2 = pd.read_csv(r'/data/data_for_running/Figure3/fold_2_val_pred_TERT_mutation.csv')
fold_3 = pd.read_csv(r'/data/data_for_running/Figure3/fold_3_val_pred_TERT_mutation.csv')
fold_4 = pd.read_csv(r'/data/data_for_running/Figure3/fold_4_val_pred_TERT_mutation.csv')
fold_5 = pd.read_csv(r'/data/data_for_running/Figure3/fold_5_val_pred_TERT_mutation.csv')

def compute_pr_ap(y_true, y_prob):
    """Return (recall, precision, AP), where x=recall and y=precision"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    return recall, precision, ap

# Average AUPRC (simple average across folds)
r1, p1, ap1 = compute_pr_ap(fold_1['y_true'], fold_1['y_prob'])
r2, p2, ap2 = compute_pr_ap(fold_2['y_true'], fold_2['y_prob'])
r3, p3, ap3 = compute_pr_ap(fold_3['y_true'], fold_3['y_prob'])
r4, p4, ap4 = compute_pr_ap(fold_4['y_true'], fold_4['y_prob'])
r5, p5, ap5 = compute_pr_ap(fold_5['y_true'], fold_5['y_prob'])

#  Average AUPRC (simple average across folds)
average_ap = (ap1 + ap2 + ap3 + ap4 + ap5) / 5.0

# Compute the positive baseline (class prevalence), the reference horizontal line on the PR plot
all_y = pd.concat([fold_1['y_true'], fold_2['y_true'], fold_3['y_true'], fold_4['y_true'], fold_5['y_true']], axis=0)
pos_rate = float(np.mean(all_y))  # 0~1

# Plot
fig, ax = plt.subplots(figsize=(8, 8))

# Per-fold curves (consistent colors and line widths)
ax.plot(r1, p1, color='#247aaf', lw=10)
ax.plot(r2, p2, color='#ff7a2f', lw=10)
ax.plot(r3, p3, color='#389d40', lw=10)
ax.plot(r4, p4, color='#d92c34', lw=10)
ax.plot(r5, p5, color='#8666b2', lw=10)

# Baseline (positive rate)
ax.hlines(pos_rate, 0, 1, color='gray', linestyle='--', lw=2)

# Center annotation: AUPRC
plt.text(0.5, 0.08, f'AUPRC = {average_ap:.3f}', horizontalalignment='center', fontsize=55)

#Thicken spines & set tick style
for side in ['left', 'bottom', 'right', 'top']:
    ax.spines[side].set_linewidth(3)
ax.tick_params(width=3, length=14, labelsize=55, pad=6)

# Axis limits
ax.set_xlim(-0.01, 1.03)
ax.set_ylim(-0.01, 1.03)

# Axis labels (keep your current style: no xlabel; keep the y-axis title as IDH_Mutation)
# If you want standard axis labels, you can change the line below to: ax.set_ylabel('Precision', fontsize=55, labelpad=80)
ax.set_ylabel('TERT_mutation', fontsize=55, labelpad=30)

# Save
fig.savefig(output_path, dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"PR plot saved: {output_path}")
