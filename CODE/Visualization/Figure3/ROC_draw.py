import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'

output_path = r"E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\figure_3_A\TERT_mutation\roc_TERT_mutation_new.png"

# 载入每一折的数据
fold_1 = pd.read_csv(r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\downstream_ZZU\prior_knowledge\outputs\fold_1_val_pred_TERT_mutation.csv')
fold_2 = pd.read_csv(r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\downstream_ZZU\prior_knowledge\outputs\fold_2_val_pred_TERT_mutation.csv')
fold_3 = pd.read_csv(r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\downstream_ZZU\prior_knowledge\outputs\fold_3_val_pred_TERT_mutation.csv')
fold_4 = pd.read_csv(r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\downstream_ZZU\prior_knowledge\outputs\fold_4_val_pred_TERT_mutation.csv')
fold_5 = pd.read_csv(r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\downstream_ZZU\prior_knowledge\outputs\fold_5_val_pred_TERT_mutation.csv')

# 假设文件里有两列: 'true_labels' 和 'pred_probs'，分别是真实标签和预测概率
# 计算每一折的fpr, tpr和AUC
def compute_roc_auc(true_labels, pred_probs):
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    return fpr, tpr, auc(fpr, tpr)

# 计算各折的ROC和AUC
fpr1, tpr1, auc1 = compute_roc_auc(fold_1['y_true'], fold_1['y_prob'])
fpr2, tpr2, auc2 = compute_roc_auc(fold_2['y_true'], fold_2['y_prob'])
fpr3, tpr3, auc3 = compute_roc_auc(fold_3['y_true'], fold_3['y_prob'])
fpr4, tpr4, auc4 = compute_roc_auc(fold_4['y_true'], fold_4['y_prob'])
fpr5, tpr5, auc5 = compute_roc_auc(fold_5['y_true'], fold_5['y_prob'])

# 计算平均AUC
average_auc = (auc1 + auc2 + auc3 + auc4 + auc5) / 5

# 绘制ROC曲线
plt.figure(figsize=(8, 8))
'''plt.plot(fpr1, tpr1, color='#247aaf', lw=10, label=f'Fold 1 (AUC = {auc1:.3f})')
plt.plot(fpr2, tpr2, color='#ff7a2f', lw=10, label=f'Fold 2 (AUC = {auc2:.3f})')
plt.plot(fpr3, tpr3, color='#389d40', lw=10, label=f'Fold 3 (AUC = {auc3:.3f})')
plt.plot(fpr4, tpr4, color='#d92c34', lw=10, label=f'Fold 4 (AUC = {auc4:.3f})')
plt.plot(fpr5, tpr5, color='#8666b2', lw=10, label=f'Fold 5 (AUC = {auc5:.3f})')'''

plt.plot(fpr1, tpr1, color='#247aaf', lw=10)
plt.plot(fpr2, tpr2, color='#ff7a2f', lw=10)
plt.plot(fpr3, tpr3, color='#389d40', lw=10)
plt.plot(fpr4, tpr4, color='#d92c34', lw=10)
plt.plot(fpr5, tpr5, color='#8666b2', lw=10)

# 绘制平均AUC
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.text(0.5, 0.1, f'AUC = {average_auc:.3f}', horizontalalignment='center', fontsize=55)

ax = plt.gca()  # 取得当前Axes
# 加粗四条边框
for side in ['left', 'bottom', 'right', 'top']:
    ax.spines[side].set_linewidth(3)

plt.ylim(-0.01, 1.01)
plt.xlim(-0.01, 1.01)

# 刻度线也加粗（可选）
ax.tick_params(width=3, length=14)

plt.tick_params(axis="both", labelsize=55)

# 设置图形标签
#plt.xlabel('False Positive Rate')
plt.ylabel('TERT_mutation', fontsize=55, labelpad=30)
#plt.title('ROC Curve for Cross-Validation Folds')
#plt.legend(loc='lower right', fontsize=20)

plt.savefig(output_path, dpi=600, bbox_inches="tight")
plt.close()

print(f"ROC 图已保存：{output_path}")
