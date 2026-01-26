library(ggplot2)

## === 路径 ===
input_csv <- "E:/Multi-omic Immunity/GCN_immune/scripts/LGG/picture/Clustering/okepoch160f4s0_valid_combined.csv"
output_png <- "E:/Multi-omic Immunity/GCN_immune/scripts/LGG/picture/Pearson_v4/All6_Scatter.png"

## 读表
dat <- read.csv(input_csv, stringsAsFactors = FALSE, check.names = FALSE)

## 6 组配对：Label=真实值, Predict=预测值
pairs <- list(
  list(label="valid_imrna_1",  pred="valid_output_rna_1",  series="RNA-PC1"),
  list(label="valid_imrna_2",  pred="valid_output_rna_2",  series="RNA-PC2"),
  list(label="valid_imdna_1",  pred="valid_output_dna_1",  series="DNA-PC1"),
  list(label="valid_imdna_2",  pred="valid_output_dna_2",  series="DNA-PC2"),
  list(label="valid_impro_1",  pred="valid_output_pro_1",  series="PRO-PC1"),
  list(label="valid_impro_2",  pred="valid_output_pro_2",  series="PRO-PC2")
)

## 检查列是否存在
need_cols <- unlist(lapply(pairs, function(x) c(x$label, x$pred)))
missing <- setdiff(need_cols, names(dat))
if (length(missing) > 0) {
  stop("缺少列：", paste(missing, collapse = ", "),
       "\n当前 CSV 列名：\n", paste(names(dat), collapse = ", "))
}

## 拼成长表（6 组）
make_one <- function(p) {
  data.frame(
    Label  = as.numeric(dat[[p$label]]),
    Predict= as.numeric(dat[[p$pred]]),
    Series = p$series
  )
}
df <- do.call(rbind, lapply(pairs, make_one))
df <- df[is.finite(df$Label) & is.finite(df$Predict), ]

## 手动颜色（6 种）+ 形状（6 种）
series_levels <- c("DNA-PC1","DNA-PC2","RNA-PC1","RNA-PC2","PRO-PC1","PRO-PC2")
series_cols   <- c("#060269","#ec0a07","#1b9e77","#d95f02","#7570b3","#e7298a")
series_shapes <- c(16,16,16,16,16,16)  # 圆、三角、方块、十字、星形、倒三角等

df$Series <- factor(df$Series, levels = series_levels)

## 画图（6 种点 + 各自平滑线）
p <- ggplot(df, aes(x = Label, y = Predict, color = Series, shape = Series)) +
  geom_point(alpha = 0.85, size = 8) +
  geom_smooth(aes(group = Series), method = "loess", se = TRUE, linewidth = 5) +
  geom_abline(slope = 0.903648477, intercept = 0, linetype = "dashed", linewidth = 3) +
  scale_color_manual(values = series_cols) +
  scale_shape_manual(values = series_shapes) +
  labs(x = "True label", y = "Prediction", color = NULL, shape = NULL) +
  ## 去掉最上面的 y 轴数字
  scale_y_continuous(
    breaks = seq(floor(min(df$Predict, na.rm=TRUE)) + 1,
                 ceiling(max(df$Predict, na.rm=TRUE)) - 1, by = 1)
  ) +
  theme_minimal(base_size = 12) +
  theme(
    # 把图例放到主图外，但尽量靠近边框
    legend.position       = c(1.2, 0.5),
    legend.justification  = c(0, 0.5),
    legend.text           = element_text(size = 70, margin = margin(l = 60, unit = "pt")),
    legend.key.size       = unit(2, "cm"),
    legend.spacing.y      = unit(55.0, "pt"),
    
    panel.background = element_rect(fill = "white", colour = NA),
    panel.border     = element_rect(colour = "black", fill = NA, linewidth = 5),
    
    axis.text.x      = element_blank(),
    axis.ticks.y     = element_line(linewidth = 1),
    axis.line        = element_line(colour = "black"),
    axis.text.y      = element_text(size = 70),
    axis.title.x     = element_text(size = 70, margin = margin(t = 30)),
    axis.title.y     = element_text(size = 70, margin = margin(r = 30)),
    
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    
    # 右侧给足空间放图例（重点）
    plot.margin = margin(t = 10, r = 530, b = 10, l = 10, unit = "pt")
  ) +
  # 自定义图例，增大键高度以辅助垂直间距
  guides(
    color = guide_legend(keyheight = unit(3, "cm"), override.aes = list(size = 8)),
    shape = guide_legend(keyheight = unit(3, "cm"), override.aes = list(size = 8))
  ) +
  # 保证黑框为正方形；clip = "off" 让框外图例不被裁切
  coord_fixed(ratio = 1, clip = "off") +
  coord_cartesian(
    ylim = c(min(df$Predict, na.rm=TRUE) - 0.1, max(df$Predict, na.rm=TRUE) + 0.1),
    xlim = c(min(df$Label,   na.rm=TRUE),        max(df$Label,   na.rm=TRUE))
  )+
  annotate("text", x = max(df$Label, na.rm = TRUE), y = min(df$Predict, na.rm = TRUE), 
           label = "Overall", hjust = 0.97, vjust = 0.2, size = 23, color = "black")  # 添加文本


# 画布加宽一点，给图例留空间（重点）
ggsave(filename = output_png, plot = p,
       width = 16.2, height = 8, units = "in", dpi = 600, bg = "white")

print(p)
cat("已保存到：", output_png, "\n")

