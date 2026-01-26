import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Arial'

# 读取新的CSV文件
file_path = r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\patient_boxplot\merged_with_clusters.csv'
df = pd.read_csv(file_path)

# 选择需要的列，假设免疫细胞计数列名是 'Immune Cells Count' 和 'new_cluster'
immune_cells_column = 'Immune Cells Count'
cluster_column = 'new_cluster'

# 按照 new_cluster 分组，并获取每个组的免疫细胞计数
groups = df.groupby(cluster_column)[immune_cells_column].apply(list)

# 设置图形大小
plt.figure(figsize=(8, 8))

# 设置箱体宽度和间隔
width = 1.0  # 箱体宽度
gap = 0.5    # 箱子间隔

# 设置每个箱子的X轴位置
positions = [1 + i * (width + gap) for i in range(len(groups))]

# 绘制箱形图
box = plt.boxplot(
    [group for group in groups],
    patch_artist=True,
    positions=positions,      # 设置箱子位置
    widths=width,             # 设置箱体宽度
    flierprops=dict(marker='*', color='red', markersize=1),
    showmeans=False,          # 不显示均值
    meanline=False,
    medianprops=dict(color='blue', linewidth=3),
    whiskerprops=dict(linewidth=5),
    capprops=dict(linewidth=5),
    whis=100
)
# 直接打印查看 box 的类型和内容
print(type(box))  # 检查 box 的类型
print(box)        # 打印 box 内容，查看里面的键和元素
# 配置箱体样式
colors = ['#8583A9', '#CA8BA8', '#A0BDD5', '#EFC57F']  # 可根据需要修改配色
for i, (patch, color) in enumerate(zip(box['boxes'], colors)):
    patch.set_edgecolor('black')
    patch.set_facecolor(color)
    patch.set_linewidth(5)
    box['medians'][i].set_color('black')
    box['medians'][i].set_linewidth(5)

# 绘制分散的散点
for i, group in enumerate(groups):
    y = np.array(group)
    x = np.full_like(y, fill_value=positions[i], dtype=float)  # 设置每个箱子的X轴位置
    # 为了让散点不沿直线排列，使用一个小的随机偏移
    x = x + np.random.uniform(-0.2, 0.2, size=len(x))  # 在箱子位置左右随机偏移
    plt.plot(x, y, '.', color='black', markersize=10)

# 轴与网格设置
plt.ylabel("Immune Cells", fontsize=55, family='Arial', labelpad=30)
plt.grid(False)  # 不显示网格
plt.yticks(fontsize=55, family='Arial')
plt.tick_params(axis='y', which='both', direction='out', length=10, width=3)

# 设置边框线宽
for spine in plt.gca().spines.values():
    spine.set_linewidth(4)

# 设置X轴边界，并保留间隔
plt.xlim(min(positions) - 1, max(positions) + 1)  # 根据箱体位置和宽度调整X轴边界

plt.xticks([])  # 不显示X轴刻度

# 保存图像
out_path = r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\patient_boxplot\immune_cells_boxplot.png'
plt.savefig(out_path, dpi=600, bbox_inches="tight")
#plt.show()

print(f"箱形图已保存至: {out_path}")
