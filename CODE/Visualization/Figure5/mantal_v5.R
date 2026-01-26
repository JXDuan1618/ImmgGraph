## =========================
## Mantel + 环境相关图 - 使用 Pearson 相关
## =========================

# ---- 依赖 ----
suppressPackageStartupMessages({
  library(ggplot2)
  library(vegan)
  library(dplyr)
  library(linkET)  
  library(reshape2)
})

theme_update(text = element_text(family = "Arial"))

# ---- 输入文件路径----
otu_file <- "E:/Multi-omic Immunity/GCN_immune/scripts/LGG/picture/figure_5/figure5_B/glul/glul.csv"
env_file <- "E:/Multi-omic Immunity/GCN_immune/scripts/LGG/picture/figure_5/figure5_B/glul/summary_original.csv"

# ---- 读取数据 ----
otu_raw <- read.csv(otu_file, header = TRUE, check.names = FALSE, row.names = 1)
env_raw <- read.csv(env_file, header = TRUE, check.names = FALSE, row.names = 1)

df <- data.frame(t(otu_raw), check.names = FALSE)

common_samples <- intersect(rownames(df), rownames(env_raw))
if (length(common_samples) == 0) {
  stop("df 与 env 没有共同样本")
}
df  <- df[common_samples, , drop = FALSE]
env <- env_raw[common_samples, , drop = FALSE]

# ---- 环境表数值化/清洗 ----
env_num <- env %>%
  mutate(across(everything(), ~ suppressWarnings(as.numeric(as.character(.)))))

na_all_cols <- sapply(env_num, function(x) all(is.na(x)))
if (any(na_all_cols)) {
  env_num <- env_num[, !na_all_cols, drop = FALSE]
}

env_scaled <- scale(env_num)

# ---- Mantel 检验 (每个 OTU 与所有环境变量)----
otu_names <- colnames(df)
env_names <- colnames(env_scaled)

mantel_results <- data.frame()

for (otu in otu_names) {
  for (env_var in env_names) {
    otu_dist <- vegdist(df[, otu, drop = FALSE], method = "euclidean")
    env_dist <- vegdist(env_scaled[, env_var, drop = FALSE], method = "euclidean")
    
    # Mantel test (这里仍然用spearman,因为是距离矩阵的比较)
    mantel_result <- mantel(otu_dist, env_dist, method = "spearman", permutations = 999)
    
    mantel_results <- rbind(mantel_results, data.frame(
      otu = otu,
      env = env_var,
      r = mantel_result$statistic,
      p.value = mantel_result$signif
    ))
  }
}

# 整体 Mantel
spec_dist <- vegdist(df, method = "bray")
env_dist_all <- vegdist(env_scaled, method = "euclidean")
mantel_all <- mantel(spec_dist, env_dist_all, method = "spearman", permutations = 999)

# ---- r / p 分档 ----
mantel_results <- mantel_results %>%
  mutate(
    df_r = cut(r, breaks = c(-Inf, 0.1, 0.2, 0.4, Inf),
               labels = c("< 0.1", "0.1 - 0.2", "0.2 - 0.4", ">= 0.4")),
    df_p = cut(p.value, breaks = c(-Inf, 0.01, 0.05, Inf),
               labels = c("< 0.01", "0.01 - 0.05", ">= 0.05"))
  )

print(colnames(env_num))
print(ncol(env_num))
# ---- 【修改】环境变量之间的相关性矩阵 - 改用 Pearson ----
n_vars <- ncol(env_num)
var_names <- colnames(env_num)

cor_matrix <- matrix(NA, n_vars, n_vars)
p_matrix <- matrix(NA, n_vars, n_vars)
rownames(cor_matrix) <- colnames(cor_matrix) <- var_names
rownames(p_matrix) <- colnames(p_matrix) <- var_names

for (i in 1:n_vars) {
  for (j in 1:n_vars) {
    if (i != j) {
      # 【关键修改】改为 pearson
      test <- cor.test(env_num[,i], env_num[,j], method = "pearson")
      cor_matrix[i, j] <- test$estimate
      p_matrix[i, j] <- test$p.value
    } else {
      cor_matrix[i, j] <- 1
      p_matrix[i, j] <- 0
    }
  }
}

print(colnames(cor_matrix))

cor_df <- melt(cor_matrix, varnames = c("y", "x"), value.name = "r")
p_df <- melt(p_matrix, varnames = c("y", "x"), value.name = "p.value")
env_cor_df <- merge(cor_df, p_df, by = c("x", "y"))

#print(env_cor_df)
#print(tail(env_cor_df))

# 只保留左上三角
env_cor_df <- env_cor_df %>%
  filter(as.numeric(factor(y, levels = var_names)) <=
           as.numeric(factor(x, levels = var_names)))

print(env_cor_df)  # 查看前几行
print(tail(env_cor_df))  # 查看后几行

# ---- 提取对角线数据 ----
diag_data <- data.frame(
  env = var_names,
  env_x = seq_along(var_names),
  r = 1,  
  p.value = 0  
)

# ---- 为绘图准备坐标 ----
cx <- 25
cy <- 0
r  <- 25

arc_start <- 115    # 弧的起点角度（度）
arc_end   <- 170  # 弧的终点角度（度）
n_pairs   <- 3     # 对的数量
delta     <- 5     # 每对内部两个点的角度间距（度），越小越“紧密”

# 计算基准角：分布在 [arc_start + delta/2, arc_end - delta/2] 上，避免越界
base_angles <- seq(arc_start + delta/2, arc_end - delta/2, length.out = n_pairs)

# 每个基准角左右各偏移 delta/2，形成一对
angles <- as.numeric(t(cbind(base_angles - delta/2, base_angles + delta/2)))

pair_r_add <- c(0.0, 5.0, 2.6)      # 三对分别为 r+0, r+1.5, r+3.0（可按需调整）
stopifnot(length(pair_r_add) == n_pairs)
r_vec <- rep(r + pair_r_add, each = 2)  # 对1重复2次、对2重复2次、对3重复2次

# 生成坐标（与 otu_names 对齐）
stopifnot(length(otu_names) == length(angles))  # 确保正好 6 个 OTU
otu_positions <- data.frame(
  otu   = otu_names,
  otu_x = cx + r_vec * cos(angles * pi / 180),
  otu_y = cy + r_vec * sin(angles * pi / 180),
  pair  = rep(paste0("pair", 1:n_pairs), each = 2),   # 可选：标注每一对
  spot  = rep(c("A","B"), times = n_pairs)            # 可选：对内第 1 / 第 2 个
)

env_positions <- data.frame(
  env = env_names,
  env_x = seq_along(env_names)
)

# 合并 Mantel 结果与坐标
mantel_plot <- mantel_results %>%
  left_join(otu_positions, by = "otu") %>%
  left_join(env_positions, by = "env")

# 显著性标签
env_cor_df <- env_cor_df %>%
  mutate(
    sig_label = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01  ~ "**",
      p.value < 0.05  ~ "*",
      TRUE ~ ""
    )
  )

# ---- 创建图形 ----
p_final <- ggplot() +
  geom_tile(data = env_cor_df, 
            aes(x = as.numeric(factor(x, levels = var_names)),
                y = as.numeric(factor(y, levels = var_names))),
            width = 1,         # 固定宽度 = 1
            height = 1,        # 固定高度 = 1
            color = "#9b9b9b",   # 白色边框
            linewidth = 0.5,     # 边框粗细
            fill = NA) +       # 无填充（透明）
  
  # 第二层：根据r值变化的彩色方块
  geom_tile(data = env_cor_df, 
            aes(x = as.numeric(factor(x, levels = var_names)),
                y = as.numeric(factor(y, levels = var_names)),
                fill = r,
                width = pmin(abs(r) * 1.5, 1.0),      # 设置宽度的最大值为1.0
                height = pmin(abs(r) * 1.5, 1.0)),    # 高度根据|r|变化
            color = NA) +      # 无边框
  
  coord_fixed(ratio = 1) +
  
  # 显著性标记
  geom_text(data = subset(env_cor_df, p.value < 0.05),
            aes(x = as.numeric(factor(x, levels = var_names)),
                y = as.numeric(factor(y, levels = var_names)),
                label = sig_label),
            color = "black", size = 5, fontface = "bold", family = "Arial") +
  
  # 对角线红色点
  geom_point(data = diag_data, aes(x = env_x-1, y = env_x), color = "red", size = 3, shape = 16) +
  
  # Mantel箭头 - 修改这里
  geom_segment(data = dplyr::filter(mantel_plot, df_p %in% c("< 0.01", "0.01 - 0.05")),
               aes(x = otu_x, y = otu_y,
                   xend = env_x-1, yend = env_x,
                   linewidth = df_r,
                   color = df_p),      # 添加颜色映射
               arrow = NULL) +
  
  # OTU标签点
  geom_point(data = otu_positions,
             aes(x = otu_x, y = otu_y),
             color = "black", size = 3, shape = 16) +
  
  geom_text(data = otu_positions,
            aes(x = otu_x, y = otu_y, label = otu),
            vjust = 0.5, hjust = 1.1, size = 6, family = "Arial") +
  
  # 配色方案
  scale_fill_gradient2(high = "#b74a95", mid = "white", low = "#138f91", 
                       name = "Pearson r", limits = c(-1, 1)) +
  
  scale_color_manual(
    values = c("< 0.01" = "#d35816", "0.01 - 0.05" = "#1b9266"),
    breaks = c("< 0.01", "0.01 - 0.05"),
    name = "Mantel's p"
  ) +
  
  scale_linewidth_manual(
    values = c("< 0.1" = 0.5, "0.1 - 0.2" = 1.0, "0.2 - 0.4" = 1.5, ">= 0.4" = 2.0),
    name = "Mantel's r"
  ) +
  
  scale_x_continuous(breaks = seq_along(env_names), labels = env_names,
                     expand = expansion(add = c(5, 0.3)),
                     position = "bottom") +
  scale_y_continuous(breaks = seq_along(env_names), labels = env_names,
                     expand = expansion(add = c(0.15, 0.15)),
                     position = "right") +
  
  theme_minimal() +
  theme(
    text = element_text(family = "Arial"),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 15, family = "Arial"),
    axis.text.y = element_text(size = 15, family = "Arial", 
                               hjust = 0,  # 左对齐，靠近主体
                               margin = margin(r = -5, unit = "pt")),  # 减少右边距，向左移
    
    # 图例字号控制
    legend.title = element_text(size = 15, family = "Arial"),   # 图例标题
    legend.text = element_text(size = 15, family = "Arial"),    # 图例内容
    legend.key.size = unit(0.8, "cm"),                          # 图例符号大小
    
    # 标题字号
    plot.title = element_text(size = 16, family = "Arial"),
    
    panel.grid = element_blank()
  ) +
  labs(x = "", y = "") +
  ggtitle(sprintf("",
                  mantel_all$statistic, mantel_all$signif))

print(p_final)


# ---- 保存 PNG ----
out_png <- "E:/Multi-omic Immunity/GCN_immune/scripts/LGG/picture/figure_5/figure5_B/glul/mantel_env_correlation_pearson.png"
ggsave(filename = out_png, plot = p_final, width = 15, height = 10, dpi = 600)
message("已保存:", out_png)

