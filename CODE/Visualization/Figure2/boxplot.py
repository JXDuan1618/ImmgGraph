 import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
file_path = r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\attached_figure\attached1\fold_metrics_with_train.csv'
df = pd.read_csv(file_path)

# 提取四列数据
columns_to_plot = ['train_dna_RMSE', 'train_rna_RMSE', 'train_pro_RMSE', 'train_mean_RMSE']
data = df[columns_to_plot]

# 配色
colors = ['#6697cc', '#df6029', '#4cad47', '#a00000']

# ——布局参数——
width = 1.0                  # 箱宽 = 1
gap   = 0.5                  # 箱与箱之间的空隙 = 0.5 个箱宽
positions = [1 + i*(width+gap) for i in range(len(columns_to_plot))]  # [1, 2.5, 4, 5.5]

plt.figure(figsize=(8, 8))
box = plt.boxplot(
    [data[c] for c in columns_to_plot],
    positions=positions,
    widths=width,
    patch_artist=True,
    flierprops=dict(marker='*', color='red', markersize=1),
    showmeans=False,        # 不显示均值
    meanline=False,
    medianprops=dict(color='blue', linewidth=3),
    whiskerprops=dict(linewidth=5),
    capprops=dict(linewidth=5),
    whis=100
)

# 箱体样式
for i, (patch, color) in enumerate(zip(box['boxes'], colors)):
    patch.set_edgecolor('black')
    patch.set_facecolor(color)
    patch.set_linewidth(5)
    box['medians'][i].set_color('black')
    box['medians'][i].set_linewidth(5)

for whisker in box['whiskers']:
    whisker.set_linewidth(5)

# ——散点与箱体对齐——
for i, col in enumerate(columns_to_plot):
    y = data[col].to_numpy()
    x = np.full_like(y, fill_value=positions[i], dtype=float)
    plt.plot(x, y, '.', color='black', markersize=20)

# 轴与网格
plt.ylabel("RMSE", fontsize=55, family='Arial', labelpad=30)
plt.grid(False)             # 不要水平虚线
plt.xticks([])              # 不显示X轴刻度
yticks_values = [0.10, 0.34, 0.57]
plt.yticks(yticks_values, fontsize=55, family='Arial')
#plt.yticks(fontsize=40, family='Arial')
plt.tick_params(axis='y', which='both', direction='out', length=10, width=3)

# 边框线宽
for spine in plt.gca().spines.values():
    spine.set_linewidth(4)

# ——关键：边框与箱子留 0.5 个箱宽距离——
plt.xlim(positions[0] - width, positions[-1] + width)

# 红色参考虚线
plt.axhline(y=0.5560, color='#283363', linestyle='--', linewidth=5)
plt.gca().set_facecolor('white')

# 保存
out_path = r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\attached_figure\attached1\boxplot_RMSE_train.png'
plt.savefig(out_path, dpi=600, bbox_inches="tight")
print("Saved to:", out_path)
