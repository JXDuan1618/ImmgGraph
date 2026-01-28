library(clusterProfiler)
library(org.Hs.eg.db)
library(ggplot2)
library(enrichplot)
library(dplyr)  # 用于数据处理，如果未安装：install.packages("dplyr")
library(ggnewscale)  # 用于多个颜色渐变，如果未安装：install.packages("ggnewscale")
library(tidyr)  # 用于complete，如果未安装：install.packages("tidyr")

theme_set(
  theme_minimal(base_family = "Arial") +
    theme(text = element_text(family = "Arial"))
)

# 读取两个数据集（根据实际情况调整路径和文件名）
# 假设kegg文件：keggano.csv（如果文件名不同，请替换）
# 两个dif文件：使用file.choose()选择，或指定固定路径
kegg1 <- read.table(file = "E:/Multi-omic Immunity/GCN_immune/scripts/LGG/picture/figure_4/figure_4_C_fuji/keggano.csv", sep = ",", header = T, row.names = 1)
dif1 <- read.table(file.choose(), sep = ",", header = T, row.names = 1)  # 或指定路径如 "E:/path/to/dif1.csv"

kegg2 <- read.table(file = "E:/Multi-omic Immunity/GCN_immune/scripts/LGG/picture/figure_4/figure_4_C_fuji/keggano.csv", sep = ",", header = T, row.names = 1)
dif2 <- read.table(file.choose(), sep = ",", header = T, row.names = 1)  # 或指定路径

# 调试：检查输入数据
print("检查 dif1 行数：")
print(nrow(dif1))
print("检查 genes1 长度：")
genes1 <- rownames(dif1)
print(length(genes1))
print(head(genes1))  # 查看前几个基因

print("检查 kegg1 结构：")
print(head(kegg1))
print("检查 keggGene1 中的独特基因数：")
keggName1 = kegg1[, c(2, 3)]  # 注意：如果只有2列，这里会出错，调整为 c(1,2) 如果必要
keggGene1 = kegg1[, c(2, 1)]
print(length(unique(keggGene1[,2])))  # 假设第2列是gene

# 检查匹配基因数
matched_genes1 <- intersect(genes1, unique(keggGene1[,2]))
print(paste("Group A 匹配基因数：", length(matched_genes1)))

# 同理检查 Group B
genes2 <- rownames(dif2)
matched_genes2 <- intersect(genes2, unique(keggGene1[,2]))  # 假设 kegg2 同 kegg1
print(paste("Group B 匹配基因数：", length(matched_genes2)))

# 为每个数据集准备TERM2GENE和TERM2NAME
# 数据集1（如果列索引不对，调整 c(2,1) 为正确 term/gene 顺序）
keggName1 = kegg1[, c(2, 3)]  # term to name，假设 col2=term, col3=name
keggGene1 = kegg1[, c(2, 1)]  # term to gene，假设 col2=term, col1=gene

# 数据集2
keggName2 = kegg2[, c(2, 3)]
keggGene2 = kegg2[, c(2, 1)]

# 进行富集分析（为每个数据集）
r1 = enricher(genes1, TERM2GENE = keggGene1, TERM2NAME = keggName1, pAdjustMethod = "fdr", pvalueCutoff = 1, qvalueCutoff = 1)
r2 = enricher(genes2, TERM2GENE = keggGene2, TERM2NAME = keggName2, pAdjustMethod = "fdr", pvalueCutoff = 1, qvalueCutoff = 1)

write.table(as.data.frame(r1), 
            file = "E:/Multi-omic Immunity/GCN_immune/scripts/LGG/picture/figure_4/figure_4_C_fuji/0/group_RNA_enrichment_results_KEGG.csv", 
            sep = ",", 
            col.names = TRUE, 
            row.names = TRUE)

# 保存 Group B 的富集结果到 CSV 文件
write.table(as.data.frame(r2), 
            file = "E:/Multi-omic Immunity/GCN_immune/scripts/LGG/picture/figure_4/figure_4_C_fuji/0/group_protein_enrichment_results_KEGG.csv", 
            sep = ",", 
            col.names = TRUE, 
            row.names = TRUE)
# 调试：检查富集结果
print("r1 结果行数：")
print(nrow(as.data.frame(r1)))
print(head(r1))

print("r2 结果行数：")
print(nrow(as.data.frame(r2)))

# 如果结果为空，停止或警告
if (nrow(as.data.frame(r1)) == 0 && nrow(as.data.frame(r2)) == 0) {
  stop("两个组的富集结果均为空。请检查输入基因是否匹配 KEGG 注释，或文件结构是否正确。")
}

# 继续原代码...
# 将两个富集结果转换为数据框，并添加分组列（Group）
df1 <- as.data.frame(r1) %>% mutate(Group = "A")
df2 <- as.data.frame(r2) %>% mutate(Group = "B")

# 合并两个数据框
df_combined <- bind_rows(df1, df2)

# 转换GeneRatio为数值型（原本是字符如"5/100"，需计算比例）
df_combined$GeneRatio <- sapply(df_combined$GeneRatio, function(x) {
  parts <- strsplit(x, "/")[[1]]
  as.numeric(parts[1]) / as.numeric(parts[2])
})

# 筛选top 7 per group（类似于showCategory=7）
# 添加检查：如果有 pvalue 列
if (!"pvalue" %in% colnames(df_combined)) {
  stop("df_combined 中缺少 pvalue 列，可能是因为富集结果为空。")
}
df_top <- df_combined %>%
  group_by(Group) %>%
  top_n(-3, pvalue) %>%
  ungroup()

# 剩余代码不变...
# 获取所有top 7的独特通路（并集）
all_desc <- unique(df_top$Description)

# 补全数据框，使每个Group都有所有通路（缺失设为NA，不绘制点）
df_combined <- df_top %>%
  complete(Group, Description = all_desc) %>%  # 补全组合，缺失列设NA
  group_by(Description) %>%
  fill(ID, BgRatio, .direction = "downup") %>%  # 填充共享列（如ID, BgRatio）
  ungroup() %>%
  # 对于缺失，显式设置绘图相关为NA（已默认NA）
  mutate(
    GeneRatio = if_else(is.na(pvalue), NA_real_, GeneRatio),
    Count = if_else(is.na(pvalue), NA_integer_, Count)
  ) %>%
  # 按平均GeneRatio排序y轴（忽略NA）
  group_by(Description) %>%
  mutate(mean_GeneRatio = mean(GeneRatio, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(Description = reorder(Description, mean_GeneRatio))

# 绘制单一dotplot，每个面板不同颜色渐变
# 绘制单一dotplot，每个面板不同颜色渐变
p_combined <- ggplot(df_combined) +
  scale_size_continuous( 
    range = c(10, 18), 
    guide = "none" 
    ) + 
  # Group A: 数据 + 点 + 第一种渐变（绿到红） 
  geom_point(data = df_combined %>% filter(Group == "A"), 
             aes(x = GeneRatio, y = Description, size = 
                   Count, color = pvalue)) + 
  scale_color_gradient(low = "#df6029", high = "#df6029", 
                       trans = "log10", guide = "none") + 
  # 新scale for Group B 
  new_scale_color() + 
  geom_point(data = df_combined %>% filter(Group == "B"), 
             aes(x = GeneRatio, y = Description, 
                 size = Count, color = pvalue)) + 
  scale_color_gradient(low = "#4cad47", high = "#4cad47", 
                       trans = "log10", guide = "none") + 
  # 统一size scale 
  
  # 使用共享的y轴 
  theme_minimal() + 
  theme( 
    panel.background = element_rect(fill = "white", colour = NA), 
    plot.background = element_rect(fill = "white", colour = NA), 
    axis.title = element_blank(), 
    axis.text.x = element_text(size = 30), 
    axis.text.y = element_text(size = 30), 
    legend.position = "none" # 移除所有图例 
    ) 
# 保存图像 
ggsave( 
  filename = "E:/Multi-omic Immunity/GCN_immune/scripts/LGG/picture/figure_4/figure_4_C_fuji/0/DNA_DEG_single_combined_diff_gradients_KEGG.png", 
  plot = p_combined, 
  width = 33, 
  height = 15, 
  units = "cm", 
  dpi = 600, 
  bg = "white" 
  ) 

print(p_combined)
dev.off()

