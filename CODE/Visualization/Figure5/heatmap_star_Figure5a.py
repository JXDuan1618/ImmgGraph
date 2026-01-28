# -*- coding: utf-8 -*-
# heatmap_no_interaction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'Arial'

## === 1) Read CSV (use the file path you provided)===
csv_path = r'/data/data_for_running/Figure5/Genes_summary_heatmap.csv'
df = pd.read_csv(csv_path)

# === 2) Generate row labels (as readable as possible)===
edge_labels = None
if {'src_gene', 'dst_gene'}.issubset(df.columns):
    edge_labels = df['src_gene'].astype(str) + ' -> ' + df['dst_gene'].astype(str)
elif 'Gene' in df.columns:
    edge_labels = df['Gene'].astype(str)
else:
    #  If there is no gene column, use row indices
    edge_labels = pd.Series([f'row_{i}' for i in range(len(df))])

# === 3) Select heatmap data columns: keep all numeric columns except Gene ===
meta_cols = {'Gene', 'src_gene', 'dst_gene', 'interaction', 'src', 'dst'}
candidate_cols = [c for c in df.columns if c not in meta_cols]
data_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
data = df[data_cols].copy()

# Color range
vmin, vmax = 0.0, 1.0

# === 4) Swap axes: X = genes, Y = clusters ===
# clusters： data
clusters = list(data.columns)
# genes：（ Gene gene），X do not show labels
genes = edge_labels.astype(str).tolist()

data_T = data.T

# # === 5) Row color bar (color by cluster0–3)===
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

#6) Layout: left row color bar + main heatmap + right colorbar ===
fig = plt.figure(figsize=(100, 20))
gs = gridspec.GridSpec(
    nrows=1, ncols=3,
    width_ratios=[2, 50, 3],
    #width_ratios=[2, 50],
    wspace=0.06
)

ax_rowbar = fig.add_subplot(gs[0, 0])  # Left row color bar
ax_main   = fig.add_subplot(gs[0, 1])  # Main heatmap
#ax_cbar   = fig.add_subplot(gs[0, 2])  # Numeric colorbar

#  —— Left row color bar ——（One color block per cluster）
if data_T.shape[0] > 0:
    row_idx = np.arange(data_T.shape[0]).reshape(-1, 1)   # n_rows x 1
    row_img = np.tile(row_idx, (1, 30))                   # Make it a vertical bar (widen for display)
    ax_rowbar.imshow(row_img, aspect='auto', cmap=row_cmap, interpolation='nearest')
else:
    ax_rowbar.text(0.5, 0.5, 'No rows', ha='center', va='center', fontsize=12)
ax_rowbar.set_xticks([])
ax_rowbar.set_yticks([])

# —— Main heatmap ——（=cluster，=gene）
im = ax_main.imshow(
    data_T.values, aspect='auto', cmap='Reds',
    vmin=vmin, vmax=vmax, interpolation='nearest'
)

# X （gene）——do not show labels
#ax_main.set_xticks([])
ax_main.set_xticks(np.arange(data_T.shape[1])+0.5)  # set x-axis ticks
ax_main.set_xticklabels(genes, rotation=45, fontsize=100)  # X labels（gene）
ax_main.tick_params(axis='x', which='both', bottom=False, top=True, labeltop=True, labelbottom=False)  #  move labels to the top



# Y （cluster）——as requested: do not show y-axis labels
ax_main.set_yticks([])
ax_main.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

# # —— Optional grid: remove these four lines if not needed ——
ax_main.set_xticks(np.arange(-0.5, data_T.shape[1], 1), minor=True)
ax_main.set_yticks(np.arange(-0.5, data_T.shape[0], 1), minor=True)

ax_main.tick_params(which='minor', bottom=False, left=False)


# Draw grid lines manually
for x in np.arange(0, data_T.shape[1]+1):
    ax_main.vlines(x-0.5, -0.5, data_T.shape[0]-0.5, color='black', linewidth=5)
for y in np.arange(0, data_T.shape[0]+1):
    ax_main.hlines(y-0.5, -0.5, data_T.shape[1]-0.5, color='black', linewidth=5)

#—— Right-side colorbar ——
show_colorbar = False  # set to True to show it again
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
    loc='upper right',                # legend
    bbox_to_anchor=(1.1, 1.0),      # (x, y)：shift a bit to the right
    fontsize=100,
    title_fontsize=100,
    frameon=False,
    handlelength=2,
    handleheight=2
)


plt.tight_layout()

# === 7) Save ===
out_path = r'/results/heatmap_no_interaction.png'
plt.savefig(out_path, dpi=600, bbox_inches='tight', pad_inches=0.3)
print(f'Saved to: {out_path}')
