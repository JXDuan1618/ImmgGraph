library(ggplot2)

## === 路径：按你的实际路径改 ===
input_csv <- "E:/Multi-omic Immunity/GCN_immune/scripts/LGG/picture/RNA_Pearson.csv"  # 读表路径
output_png <- "E:/Multi-omic Immunity/GCN_immune/scripts/LGG/picture/Pearson_v3/RNA_Pearson.png" # 保存图片路径

## 读表（用 base R，避免额外装包）
dat <- read.csv(input_csv, stringsAsFactors = FALSE, check.names = FALSE)

## 检查必须列是否存在
required_cols <- c("valid_imrna_1","valid_output_rna_1","valid_imrna_2","valid_output_rna_2")
missing <- setdiff(required_cols, names(dat))
if (length(missing) > 0) {
  stop("缺少列：", paste(missing, collapse = ", "), 
       "\n请确认 CSV 列名与代码一致。当前列名有：\n", paste(names(dat), collapse = ", "))
}

## 拼成长表（两组数据）
df <- rbind(
  data.frame(RNA_label = dat$valid_imrna_1, RNA_predict = dat$valid_output_rna_1, set = "set1"),
  data.frame(RNA_label = dat$valid_imrna_2, RNA_predict = dat$valid_output_rna_2, set = "set2")
)

## 转为数值并清理缺失/非法
suppressWarnings({
  df$RNA_label   <- as.numeric(df$RNA_label)
  df$RNA_predict <- as.numeric(df$RNA_predict)
})
df <- df[is.finite(df$RNA_label) & is.finite(df$RNA_predict), ]

p <- ggplot(df, aes(x = RNA_label, y = RNA_predict, color = set)) +
  geom_point(alpha = 0.8, size = 8) +
  geom_smooth(aes(group = set), method = "loess", se = TRUE, size = 5) +
  geom_abline(slope = 0.9186396, intercept = 0, linetype = "dashed", linewidth = 3) +
  labs(x = "True label", y = "Prediction", color = "") +
  scale_color_manual(
    values = c("set1" = "#1b9e77",  # 蓝色
               "set2" = "#d95f02"),  # 红色
    labels = c("set1" = "PC1", "set2" = "PC2")  # 设置图例标签
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "none",
    #legend.position = "bottom",  # 图例位置
    panel.background = element_rect(fill = "white", colour = NA),  # 白底
    panel.border     = element_rect(colour = "black", fill = NA, linewidth = 5), # 面板边框
    axis.text.x      = element_blank(),    # 去掉横坐标数字
    plot.title       = element_blank(),    # 去掉标题
    axis.ticks.y     = element_line(size = 1),  # 延伸纵坐标的 TICK
    axis.line        = element_line(colour = "black"),  # 纵坐标线
    axis.text.y      = element_text(size = 70),  # 纵坐标字体更大
    axis.title.x     = element_text(size = 70, margin = margin(t = 30)),  # 横坐标标题字体加大
    axis.title.y     = element_text(size = 70, margin = margin(r = 30)),   # 纵坐标标题字体加大
    legend.text      = element_text(size = 20),  # 图例文字大小
    legend.title     = element_text(size = 20),  # 图例标题大小
    legend.key.size  = unit(1.5, "cm"),  # 图例符号大小
    panel.grid.major = element_blank(),  # 去除主网格线
    panel.grid.minor = element_blank()   # 去除次网格线
  ) +
  ylim(min(df$RNA_predict) - 0.1, max(df$RNA_predict) + 0.1) +  # 延伸纵坐标范围
  annotate("text", x = max(df$RNA_label), y = min(df$RNA_predict), label = "Transcriptomics", 
           hjust = 0.97, vjust = 0.9, size = 23, color = "black")  # 添加文字

p <- p + scale_y_continuous(
  breaks = seq(floor(min(df$RNA_predict)) + 1, ceiling(max(df$RNA_predict)) - 1, by = 1) # 去掉最上面的刻度
)

# 保存为 PNG（600 dpi）
dir.create(dirname(output_png), recursive = TRUE, showWarnings = FALSE)
ggsave(filename = output_png, plot = p, width = 9.5, height = 8, units = "in", dpi = 600, bg = "white")

# 同时打印到绘图窗口并提示保存位置
print(p)
cat("已保存到：", output_png, "\n")

