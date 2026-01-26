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

# 读取三个数据集（根据实际情况调整路径和文件名）
# 假设三个kegg文件：keggano1.csv, keggano2.csv, keggano3.csv（如果文件名不同，请替换）
# 三个dif文件：使用file.choose()选择，或指定固定路径
kegg1 <- read.table(file = "E:/Multi-omic Immunity/GCN_immune/scripts/LGG/picture/figure_4/figure_4_C_fuji/keggano.csv", sep = ",", header = T, row.names = 1)
dif1 <- read.table(file.choose(), sep = ",", header = T, row.names = 1)  # 或指定路径如 "E:/path/to/dif1.csv"

kegg2 <- read.table(file = "E:/Multi-omic Immunity/GCN_immune/scripts/LGG/picture/figure_4/figure_4_C_fuji/keggano.csv", sep = ",", header = T, row.names = 1)
dif2 <- read.table(file.choose(), sep = ",", header = T, row.names = 1)  # 或指定路径

kegg3 <- read.table(file = "E:/Multi-omic Immunity/GCN_immune/scripts/LGG/picture/figure_4/figure_4_C_fuji/keggano.csv", sep = ",", header = T, row.names = 1)
dif3 <- read.table(file.choose(), sep = ",", header = T, row.names = 1)  # 或指定路径

# 为每个数据集准备TERM2GENE和TERM2NAME
# 数据集1
keggName1 = kegg1[, c(2, 3)]
keggGene1 = kegg1[, c(2, 1)]
genes1 <- rownames(dif1)

# 数据集2
keggName2 = kegg2[, c(2, 3)]
keggGene2 = kegg2[, c(2, 1)]
genes2 <- rownames(dif2)

# 数据集3
keggName3 = kegg3[, c(2, 3)]
keggGene3 = kegg3[, c(2, 1)]
genes3 <- rownames(dif3)

# 进行富集分析（为每个数据集）
r1 = enricher(genes1, TERM2GENE = keggGene1, TERM2NAME = keggName1, pAdjustMethod = "fdr", pvalueCutoff = 1, qvalueCutoff = 1)
r2 = enricher(genes2, TERM2GENE = keggGene2, TERM2NAME = keggName2, pAdjustMethod = "fdr", pvalueCutoff = 1, qvalueCutoff = 1)
r3 = enricher(genes3, TERM2GENE = keggGene3, TERM2NAME = keggName3, pAdjustMethod = "fdr", pvalueCutoff = 1, qvalueCutoff = 1)


write.table(as.data.frame(r1), 
            file = "E:/Multi-omic Immunity/GCN_immune/scripts/LGG/picture/figure_4/figure_4_C_fuji/2/group_DNA_enrichment_results_KEGG.csv", 
            sep = ",", 
            col.names = TRUE, 
            row.names = TRUE)

# 保存 Group B 的富集结果到 CSV 文件
write.table(as.data.frame(r2), 
            file = "E:/Multi-omic Immunity/GCN_immune/scripts/LGG/picture/figure_4/figure_4_C_fuji/2/group_RNA_enrichment_results_KEGG.csv", 
            sep = ",", 
            col.names = TRUE, 
            row.names = TRUE)

write.table(as.data.frame(r3), 
            file = "E:/Multi-omic Immunity/GCN_immune/scripts/LGG/picture/figure_4/figure_4_C_fuji/2/group_Protein_enrichment_results_KEGG.csv", 
            sep = ",", 
            col.names = TRUE, 
            row.names = TRUE)

# 调试：查看pvalue范围（可选，为每个r）
print("pvalue summary for r1:")
print(summary(r1$pvalue))
print("pvalue summary for r2:")
print(summary(r2$pvalue))
print("pvalue summary for r3:")
print(summary(r3$pvalue))

# 将三个富集结果转换为数据框，并添加分组列（Group）
df1 <- as.data.frame(r1) %>% mutate(Group = "A")
df2 <- as.data.frame(r2) %>% mutate(Group = "B")
df3 <- as.data.frame(r3) %>% mutate(Group = "C")

# 合并三个数据框
df_combined <- bind_rows(df1, df2, df3)

# 转换GeneRatio为数值型（原本是字符如"5/100"，需计算比例）
df_combined$GeneRatio <- sapply(df_combined$GeneRatio, function(x) {
  parts <- strsplit(x, "/")[[1]]
  as.numeric(parts[1]) / as.numeric(parts[2])
})

# 筛选top 7 per group（类似于showCategory=7）
df_top <- df_combined %>%
  group_by(Group) %>%
  top_n(-3, pvalue) %>%
  ungroup()

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

# 绘制单一dotplot，使用facet_wrap分成三个面板（垂直），每个面板不同颜色渐变
p_combined <- ggplot(df_combined) +  
  
  scale_size_continuous(
    range = c(10, 18),
    name  = "Count",
    guide = guide_legend(order = 1)   # 👈 顺序=1
  ) +
  
  
  # Group A: 数据 + 点 + 第一种渐变（绿到红）
  geom_point(data = df_combined %>% filter(Group == "A"), 
             aes(x = GeneRatio, y = Description, size = Count, color = pvalue)) +
  scale_color_gradient(low = "#6697cc", high = "#6697cc", trans = "log10", name = "DNA", guide = guide_colorbar(order = 2, label = FALSE)) +  
  
  # 新scale for Group B
  new_scale_color() +
  geom_point(data = df_combined %>% filter(Group == "B"), 
             aes(x = GeneRatio, y = Description, size = Count, color = pvalue)) +
  scale_color_gradient(low = "#df6029", high = "#df6029", trans = "log10", name = "RNA", guide = guide_colorbar(order = 3, label = FALSE)) +  
  
  # 新scale for Group C
  new_scale_color() +
  geom_point(data = df_combined %>% filter(Group == "C"), 
             aes(x = GeneRatio, y = Description, size = Count, color = pvalue)) +
  scale_color_gradient(low = "#4cad47", high = "#4cad47", trans = "log10", name = "Protein", guide = guide_colorbar(order = 4, label = FALSE)) +  
  
  # 统一size scale

  
  # 使用共享的y轴
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "white", colour = NA),
    plot.background  = element_rect(fill = "white", colour = NA),
    
    axis.title = element_blank(),
    axis.text.x = element_text(size = 30),
    axis.text.y = element_text(size = 30),
    legend.text  = element_text(size = 30),
    legend.title = element_text(size = 30),
    legend.position = "none",  
    legend.box = "vertical")

# 保存图像
ggsave(
  filename = "E:/Multi-omic Immunity/GCN_immune/scripts/LGG/picture/figure_4/figure_4_C_fuji/2/DNA_DEG_single_combined_diff_gradients_nolabel_KEGG.png",
  plot = p_combined,
  width = 36,      
  height = 15,
  units = "cm",
  dpi = 600,
  bg = "white"
)

print(p_combined)
dev.off()

