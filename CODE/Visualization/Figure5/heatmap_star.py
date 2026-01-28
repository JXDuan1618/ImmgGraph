# -*- coding: utf-8 -*-
# heatmap_no_interaction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'Arial'

# === 1) 读取 CSV（使用你上传的文件路径）===
csv_path = r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\figure_5\heatmap\Genes_summary_heatmap.csv'
df = pd.read_csv(csv_path)

# === 2) 生成行索引（尽量可读）===
edge_labels = None
if {'src_gene', 'dst_gene'}.issubset(df.columns):
    edge_labels = df['src_gene'].astype(str) + ' -> ' + df['dst_gene'].astype(str)
elif 'Gene' in df.columns:
    edge_labels = df['Gene'].astype(str)
else:
    # 若没有基因列，就用行号
    edge_labels = pd.Series([f'row_{i}' for i in range(len(df))])

# === 3) 选择用于热图的数据列：保留除 Gene 以外的所有数值列 ===
meta_cols = {'Gene', 'src_gene', 'dst_gene', 'interaction', 'src', 'dst'}
candidate_cols = [c for c in df.columns if c not in meta_cols]
data_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
data = df[data_cols].copy()

# 颜色范围
vmin, vmax = 0.0, 1.0

# === 4) 轴互换：X=基因，Y=cluster ===
# clusters：原 data 的列名
clusters = list(data.columns)
# genes：来自行索引（如果有 Gene 列则是基因名），X 轴仍不显示标签
genes = edge_labels.astype(str).tolist()
# 转置后：行=cluster，列=gene
data_T = data.T

# === 5) 行颜色条（按 cluster0~3 上色）===
row_colors_map = {
    'cluster0': '#8583a9',
    'cluster1': '#ca8ba8',
    'cluster2': '#a0bdd5',
    'cluster3': '#efc57f',
}
def pick_row_color(name):
    s = str(name).lower()
    for key, color in row_colors_map.items():
        if key in s:
            return color
    return '#BDC3C7'
row_color_list = [pick_row_color(c) for c in clusters]
row_cmap = ListedColormap(row_color_list)

# === 6) 绘图布局：左侧行颜色条 + 主热图 + 右侧色标 ===
fig = plt.figure(figsize=(100, 20))
gs = gridspec.GridSpec(
    nrows=1, ncols=3,
    width_ratios=[2, 50, 3],
    #width_ratios=[2, 50],
    wspace=0.06
)

ax_rowbar = fig.add_subplot(gs[0, 0])  # 左侧行颜色条
ax_main   = fig.add_subplot(gs[0, 1])  # 主热力图
#ax_cbar   = fig.add_subplot(gs[0, 2])  # 数值色标

# —— 左侧行颜色条 ——（每个 cluster 一条色块）
if data_T.shape[0] > 0:
    row_idx = np.arange(data_T.shape[0]).reshape(-1, 1)   # n_rows x 1
    row_img = np.tile(row_idx, (1, 30))                   # 做成竖条（加宽显示）
    ax_rowbar.imshow(row_img, aspect='auto', cmap=row_cmap, interpolation='nearest')
else:
    ax_rowbar.text(0.5, 0.5, 'No rows', ha='center', va='center', fontsize=12)
ax_rowbar.set_xticks([])
ax_rowbar.set_yticks([])

# —— 主热力图 ——（行=cluster，列=gene）
im = ax_main.imshow(
    data_T.values, aspect='auto', cmap='Reds',
    vmin=vmin, vmax=vmax, interpolation='nearest'
)

# X 轴（基因）——不显示标签
#ax_main.set_xticks([])
ax_main.set_xticks(np.arange(data_T.shape[1])+0.5)  # 设置 X 轴的刻度
ax_main.set_xticklabels(genes, rotation=45, fontsize=100)  # 设置 X 轴的标签（基因）
ax_main.tick_params(axis='x', which='both', bottom=False, top=True, labeltop=True, labelbottom=False)  # 移动标签到顶部



# Y 轴（cluster）——按你的要求：不显示 y 轴标签
ax_main.set_yticks([])
ax_main.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

# —— 可选网格：如不需要可删除这四行 ——
ax_main.set_xticks(np.arange(-0.5, data_T.shape[1], 1), minor=True)
ax_main.set_yticks(np.arange(-0.5, data_T.shape[0], 1), minor=True)

ax_main.tick_params(which='minor', bottom=False, left=False)


# 自己画线
for x in np.arange(0, data_T.shape[1]+1):
    ax_main.vlines(x-0.5, -0.5, data_T.shape[0]-0.5, color='black', linewidth=5)
for y in np.arange(0, data_T.shape[0]+1):
    ax_main.hlines(y-0.5, -0.5, data_T.shape[1]-0.5, color='black', linewidth=5)

# —— 右侧颜色条 ——
show_colorbar = False  # 改成 True 就能重新显示
if show_colorbar:
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.ax.tick_params(labelsize=100)


color_for_0 = plt.cm.Reds(0.0)
color_for_1 = plt.cm.Reds(1.0)

legend_handles = [
    mpatches.Patch(
        facecolor=color_for_0, edgecolor='black', linewidth=5, label='0'
    ),
    mpatches.Patch(
        facecolor=color_for_1, edgecolor='black', linewidth=5, label='1'
    )
]
ax_main.legend(
    handles=legend_handles,
    title='Legend',
    loc='upper right',                # 图例锚点位置
    bbox_to_anchor=(1.1, 1.0),      # (x, y)：往右移一点
    fontsize=100,
    title_fontsize=100,
    frameon=False,
    handlelength=2,
    handleheight=2
)


plt.tight_layout()

# === 7) 保存 ===
out_path = r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\figure_5\heatmap\heatmap_no_interaction.png'
plt.savefig(out_path, dpi=600, bbox_inches='tight', pad_inches=0.3)
print(f'Saved to: {out_path}')
