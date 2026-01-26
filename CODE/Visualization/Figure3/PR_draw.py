# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import pandas as pd
import numpy as np
import matplotlib as mpl

# 全局字体：Arial
mpl.rcParams['font.family'] = 'Arial'

# 输出路径（改成 PR 文件名）
output_path = r"E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\figure_3_A\TERT_mutation\pr_TERT_mutation_new.png"

# 载入每一折的数据（列：y_true, y_prob）
fold_1 = pd.read_csv(r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\downstream_ZZU\prior_knowledge\outputs\fold_1_val_pred_TERT_mutation.csv')
fold_2 = pd.read_csv(r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\downstream_ZZU\prior_knowledge\outputs\fold_2_val_pred_TERT_mutation.csv')
fold_3 = pd.read_csv(r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\downstream_ZZU\prior_knowledge\outputs\fold_3_val_pred_TERT_mutation.csv')
fold_4 = pd.read_csv(r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\downstream_ZZU\prior_knowledge\outputs\fold_4_val_pred_TERT_mutation.csv')
fold_5 = pd.read_csv(r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\downstream_ZZU\prior_knowledge\outputs\fold_5_val_pred_TERT_mutation.csv')

def compute_pr_ap(y_true, y_prob):
    """返回 (recall, precision, AP)，其中 x=recall, y=precision"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    return recall, precision, ap

# 各折 PR 曲线与 AUPRC
r1, p1, ap1 = compute_pr_ap(fold_1['y_true'], fold_1['y_prob'])
r2, p2, ap2 = compute_pr_ap(fold_2['y_true'], fold_2['y_prob'])
r3, p3, ap3 = compute_pr_ap(fold_3['y_true'], fold_3['y_prob'])
r4, p4, ap4 = compute_pr_ap(fold_4['y_true'], fold_4['y_prob'])
r5, p5, ap5 = compute_pr_ap(fold_5['y_true'], fold_5['y_prob'])

# 平均 AUPRC（简单平均各折）
average_ap = (ap1 + ap2 + ap3 + ap4 + ap5) / 5.0

# 计算阳性基线（类别占比），PR 图的参考水平线
all_y = pd.concat([fold_1['y_true'], fold_2['y_true'], fold_3['y_true'], fold_4['y_true'], fold_5['y_true']], axis=0)
pos_rate = float(np.mean(all_y))  # 0~1

# 画图（沿用你的风格）
fig, ax = plt.subplots(figsize=(8, 8))

# 每折曲线（配色与线宽一致）
ax.plot(r1, p1, color='#247aaf', lw=10)
ax.plot(r2, p2, color='#ff7a2f', lw=10)
ax.plot(r3, p3, color='#389d40', lw=10)
ax.plot(r4, p4, color='#d92c34', lw=10)
ax.plot(r5, p5, color='#8666b2', lw=10)

# 基线（正例率）
ax.hlines(pos_rate, 0, 1, color='gray', linestyle='--', lw=2)

# 中央标注：AUPRC
plt.text(0.5, 0.08, f'AUPRC = {average_ap:.3f}', horizontalalignment='center', fontsize=55)

# 边框加粗 & 刻度样式
for side in ['left', 'bottom', 'right', 'top']:
    ax.spines[side].set_linewidth(3)
ax.tick_params(width=3, length=14, labelsize=55, pad=6)

# 坐标范围
ax.set_xlim(-0.01, 1.03)
ax.set_ylim(-0.01, 1.03)

# 轴标签（保持你现在的风格：不画 xlabel，y 标题继续写 IDH_Mutation）
# 如需标准坐标轴，可把下面这行改成：ax.set_ylabel('Precision', fontsize=55, labelpad=80)
ax.set_ylabel('TERT_mutation', fontsize=55, labelpad=30)

# 保存
fig.savefig(output_path, dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"PR 图已保存：{output_path}")
