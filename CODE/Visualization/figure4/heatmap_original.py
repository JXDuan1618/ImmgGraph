import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
# 可选：添加图例用
from matplotlib.patches import Patch

plt.rcParams['font.family'] = 'Arial'

# === 1. 读取 CSV 文件 ===
df = pd.read_csv(r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\figure_4\figure_4_B_heatmap\heatmap\3\summary_cluster3_scaled.csv')

# === 2. 创建边标签列 ===
df['edge'] = df['src_gene'].astype(str) + ' -> ' + df['dst_gene'].astype(str)
df.set_index('edge', inplace=True)
df.index.name = None

# === 3. 分离数据与 interaction ===
# 这里保持你原来的列剔除方式
data = df.drop(columns=['src_gene', 'dst_gene', 'interaction', 'src', 'dst'])  #
interaction = df['interaction']

# === 4. 定义 interaction 颜色映射（可按需调整/增减） ===
interaction_colors_map = {
    'dna_interact': '#6697cc',
    'transcribe':   '#ac5aa1',
    'rna_interact': '#df6029',
    'translate':    '#dbd2ea',
    'protein_interact': '#4cad47',
}
# 未定义类型默认灰色
color_series = interaction.map(lambda x: interaction_colors_map.get(x, '#BDC3C7'))

# === 5. 准备右侧“交互颜色条”数据 ===
n_rows = data.shape[0]
colors = color_series.to_numpy()                     # 直接拿到颜色数组，避免 DataFrame 列名问题
color_index_img = np.arange(n_rows).reshape(n_rows, 1)  # 0..n_rows-1 的索引矩阵

# === 6. 绘图：主热力图 + 交互条 + 原始 colorbar ===
fig = plt.figure(figsize=(20, 30))
# 三列：主图、交互条、colorbar
gs = gridspec.GridSpec(ncols=3, nrows=1, width_ratios=[50, 1, 1.8], wspace=0.08)

ax_main = fig.add_subplot(gs[0, 0])
ax_strip = fig.add_subplot(gs[0, 1])
ax_cbar  = fig.add_subplot(gs[0, 2])

# 主热力图（保留你的 RdBu/vmin/vmax）
hm = sns.heatmap(
    data,
    ax=ax_main,
    cmap='RdBu',
    annot=False,
    linewidths=0.5,
    vmin=0,
    vmax=1,
    cbar=True,          # 开启色条
    cbar_ax=ax_cbar     # 把色条画在第三列
)

# 坐标外观与原来一致
ax_main.set_xticklabels(ax_main.get_xticklabels(), rotation=90)
ax_main.set_yticklabels(ax_main.get_yticklabels(), rotation=0, fontsize=38)
ax_main.set_xticks([])

# 右侧“交互颜色条”：逐行一个色块，与主图行对齐
ax_strip.imshow(
    color_index_img,
    aspect='auto',
    cmap=ListedColormap(colors),
    interpolation='nearest'
)
ax_strip.set_xticks([])
ax_strip.set_yticks([])

# 可选：在 colorbar 顶部加标题
#ax_cbar.set_title('Expression', fontsize=12, pad=6)

# 可选：添加 interaction 图例（放在主图外，以免挡住图形）
'''legend_handles = []
for k, v in interaction_colors_map.items():
    legend_handles.append(Patch(facecolor=v, edgecolor='none', label=k))
legend_handles.append(Patch(facecolor='#BDC3C7', edgecolor='none', label='other'))
# 把图例放在主图右上角外侧
ax_main.legend(
    handles=legend_handles,
    title='interaction',
    loc='upper left',
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0.
)'''

plt.tight_layout()
plt.savefig(r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\figure_4\figure_4_B_heatmap\heatmap\3\heatmap_cluster3.png', dpi=600,
            bbox_inches='tight',  # 自动包含所有文字
            pad_inches=0.3  # 给文字留空白
            )
# plt.show()
