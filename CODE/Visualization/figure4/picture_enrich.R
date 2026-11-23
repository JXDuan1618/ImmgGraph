library(ggplot2)
library(dplyr)
library(ggnewscale)
library(tidyr)

theme_set(
  theme_minimal(base_family = "Arial") +
    theme(text = element_text(family = "Arial"))
)

#==========================================================
# 1. 安全读取富集结果
#    空文件或全是空白行时，返回0行数据框
#==========================================================

result_dir <- paste0(
  "E:/Multi-omic Immunity/GCN_immune/scripts/LGG/picture/figure_4/figure_4_C_fuji/new_20260903/KEGG&Rectome/1/shortname"
)

safe_read_enrichment <- function(filename, group_name) {
  
  filepath <- file.path(result_dir, filename)
  
  # 空结果模板
  empty_result <- data.frame(
    ID          = character(0),
    Description = character(0),
    GeneRatio   = character(0),
    BgRatio     = character(0),
    pvalue      = numeric(0),
    p.adjust    = numeric(0),
    qvalue      = numeric(0),
    geneID      = character(0),
    Count       = integer(0),
    Group       = character(0),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  
  #--------------------------------------------------------
  # 1. 文件不存在
  #--------------------------------------------------------
  
  if (!file.exists(filepath)) {
    warning(
      filename,
      " 不存在，按照0条富集结果处理。"
    )
    
    return(empty_result)
  }
  
  #--------------------------------------------------------
  # 2. 文件大小为0
  #--------------------------------------------------------
  
  file_size <- file.info(filepath)$size
  
  if (is.na(file_size) || file_size == 0) {
    warning(
      filename,
      " 是空文件，按照0条富集结果处理。"
    )
    
    return(empty_result)
  }
  
  #--------------------------------------------------------
  # 3. 以二进制方式读取，避免直接触发UTF-8错误
  #--------------------------------------------------------
  
  raw_content <- readBin(
    filepath,
    what = "raw",
    n = file_size
  )
  
  text_original <- rawToChar(raw_content)
  
  # 依次尝试常见编码
  encoding_candidates <- c(
    "UTF-8",
    "GB18030",
    "GBK"
  )
  
  last_error <- NULL
  
  for (current_encoding in encoding_candidates) {
    
    # 转换为UTF-8
    text_utf8 <- suppressWarnings(
      iconv(
        text_original,
        from = current_encoding,
        to = "UTF-8",
        sub = NA
      )
    )
    
    # 当前编码无法转换，尝试下一种
    if (
      length(text_utf8) == 0 ||
      is.na(text_utf8)
    ) {
      next
    }
    
    # 删除UTF-8 BOM
    text_utf8 <- sub(
      "^\ufeff",
      "",
      text_utf8
    )
    
    # 拆分为行，同时兼容Windows和Linux换行符
    file_lines <- strsplit(
      text_utf8,
      split = "\r\n|\n|\r",
      perl = TRUE
    )[[1]]
    
    # 删除真正的空白行，以及只包含逗号或分号的空记录
    valid_lines <- !grepl(
      "^[[:space:],;]*$",
      file_lines,
      perl = TRUE
    )
    
    file_lines <- file_lines[valid_lines]
    
    # 删除空行后没有内容
    if (length(file_lines) == 0) {
      warning(
        filename,
        " 只包含空白行，按照0条富集结果处理。"
      )
      
      return(empty_result)
    }
    
    # 将清理后的行重新组合
    clean_text <- paste(
      file_lines,
      collapse = "\n"
    )
    
    # 尝试解析CSV
    result <- tryCatch(
      {
        read.csv(
          text = clean_text,
          header = TRUE,
          row.names = 1,
          stringsAsFactors = FALSE,
          check.names = FALSE,
          quote = "\"",
          comment.char = "",
          fill = TRUE
        )
      },
      error = function(e) {
        last_error <<- conditionMessage(e)
        NULL
      }
    )
    
    # 当前编码成功读取
    if (!is.null(result)) {
      
      # 没有数据行但有表头时，也返回结构完整的空结果
      if (nrow(result) == 0) {
        warning(
          filename,
          " 只有表头、没有数据行，按照0条富集结果处理。"
        )
        
        return(empty_result)
      }
      
      # 添加分组
      result$Group <- rep(
        group_name,
        nrow(result)
      )
      
      message(
        filename,
        " 读取成功；使用编码：",
        current_encoding,
        "；数据行数：",
        nrow(result)
      )
      
      return(result)
    }
  }
  
  # 所有编码均读取失败
  stop(
    filename,
    " 无法读取。已尝试 UTF-8、GB18030 和 GBK 编码。",
    if (!is.null(last_error)) {
      paste0(" 最后一次错误：", last_error)
    } else {
      ""
    }
  )
}

# DNA
df1 <- safe_read_enrichment(
  filename = "group_DNA_KEGG&Reactome.csv",
  group_name = "A"
)

# RNA
df2 <- safe_read_enrichment(
  filename = "group_RNA_KEGG&Reactome.csv",
  group_name = "B"
)

# Protein
df3 <- safe_read_enrichment(
  filename = "group_Protein_KEGG&Reactome.csv",
  group_name = "C"
)

#==========================================================
# 2. 检查读取结果
#==========================================================

print("DNA enrichment result dimensions:")
print(dim(df1))

print("RNA enrichment result dimensions:")
print(dim(df2))

print("Protein enrichment result dimensions:")
print(dim(df3))

print("DNA pvalue summary:")
print(summary(df1$pvalue))

print("RNA pvalue summary:")
print(summary(df2$pvalue))

print("Protein pvalue summary:")
print(summary(df3$pvalue))

#==========================================================
# 3. 合并三个数据框
#==========================================================

df_combined <- bind_rows(df1, df2, df3)

if (nrow(df_combined) == 0) {
  stop("DNA、RNA 和 Protein 三个富集结果文件均为空，没有数据可以绘图。")
}

# 防止 GeneRatio 被 Excel 转换成类似“2月6日”
df_combined$GeneRatio <- as.character(df_combined$GeneRatio)
df_combined$GeneRatio <- gsub(
  "月",
  "/",
  df_combined$GeneRatio,
  fixed = TRUE
)
df_combined$GeneRatio <- gsub(
  "日",
  "",
  df_combined$GeneRatio,
  fixed = TRUE
)

# 转换 pvalue 和 Count 为数值型
df_combined$pvalue <- as.numeric(df_combined$pvalue)
df_combined$Count <- as.numeric(df_combined$Count)

# 转换GeneRatio为数值型（原本是字符如"5/100"，需计算比例）
df_combined$GeneRatio <- sapply(df_combined$GeneRatio, function(x) {
  parts <- strsplit(x, "/")[[1]]
  as.numeric(parts[1]) / as.numeric(parts[2])
})

# 筛选top 7 per group（类似于showCategory=7）
df_top <- df_combined %>%
  group_by(Group) %>%
  top_n(-9, pvalue) %>%
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
  
x_range <- range(df_combined$GeneRatio, na.rm = TRUE)

x_breaks <- seq(
  from = x_range[1],
  to = x_range[2],
  length.out = 3
)
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
  scale_x_continuous(
    breaks = x_breaks,
    labels = scales::label_number(accuracy = 0.001),
    expand = expansion(mult = c(0.15, 0.15))
  ) +
  coord_cartesian(clip = "off") +
  coord_cartesian(clip = "off") +
  # 使用共享的y轴
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "white", colour = NA),
    plot.background  = element_rect(fill = "white", colour = NA),
    
    axis.title = element_blank(),
    axis.text.x = element_text(size = 30),
    axis.text.y.left = element_text(
      size = 30,
      hjust = 1,
      margin = margin(r = 20, unit = "pt")
    ),
    legend.text  = element_text(size = 30),
    legend.title = element_text(size = 30),
    legend.position = "none",  
    legend.box = "vertical",
    
    plot.margin = margin(
      t = 5.5,
      r = 15,   # 增大右边距，使气泡绘图区变窄
      b = 5.5,
      l = 5.5,
      unit = "pt"
    ))

# 保存图像
ggsave(
  filename = "E:/Multi-omic Immunity/GCN_immune/scripts/LGG/picture/figure_4/figure_4_C_fuji/new_20260903/KEGG&Rectome/1/cluster2_DEG_single_combined_diff_gradients_nolabel_cluster0.png",
  plot = p_combined,
  width = 32,      
  height = 15,
  units = "cm",
  dpi = 600,
  bg = "white"
)

print(p_combined)

