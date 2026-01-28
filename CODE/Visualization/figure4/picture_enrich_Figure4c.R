library(clusterProfiler)
library(org.Hs.eg.db)
library(ggplot2)
library(enrichplot)
library(dplyr)  #  for data processing; if not installed: install.packages("dplyr")
library(ggnewscale)  # for multiple color gradients; if not installed: install.packages("ggnewscale")
library(tidyr)  # for complete(); if not installed: install.packages("tidyr")

theme_set(
  theme_minimal(base_family = "Arial") +
    theme(text = element_text(family = "Arial"))
)

# Read three datasets (adjust paths and filenames as needed)
# Assume three KEGG files: keggano1.csv, keggano2.csv, keggano3.csv (replace if filenames differ)
# Three dif files: select via file.choose() or specify fixed paths
kegg1 <- read.table(file = "/data/data_for_running/Figure4/keggano.csv", sep = ",", header = T, row.names = 1)
dif1 <- read.table(file.choose(), sep = ",", header = T, row.names = 1)  # or specify a path

kegg2 <- read.table(file = "/data/data_for_running/Figure4/keggano.csv", sep = ",", header = T, row.names = 1)
dif2 <- read.table(file.choose(), sep = ",", header = T, row.names = 1)  # or specify a path

kegg3 <- read.table(file = "/data/data_for_running/Figure4/keggano.csv", sep = ",", header = T, row.names = 1)
dif3 <- read.table(file.choose(), sep = ",", header = T, row.names = 1)  # or specify a path

# Prepare TERM2GENE and TERM2NAME for each dataset
# Dataset 1
keggName1 = kegg1[, c(2, 3)]
keggGene1 = kegg1[, c(2, 1)]
genes1 <- rownames(dif1)

##Dataset 2
keggName2 = kegg2[, c(2, 3)]
keggGene2 = kegg2[, c(2, 1)]
genes2 <- rownames(dif2)

#  Dataset 3
keggName3 = kegg3[, c(2, 3)]
keggGene3 = kegg3[, c(2, 1)]
genes3 <- rownames(dif3)

# Perform enrichment analysis (for each dataset)
r1 = enricher(genes1, TERM2GENE = keggGene1, TERM2NAME = keggName1, pAdjustMethod = "fdr", pvalueCutoff = 1, qvalueCutoff = 1)
r2 = enricher(genes2, TERM2GENE = keggGene2, TERM2NAME = keggName2, pAdjustMethod = "fdr", pvalueCutoff = 1, qvalueCutoff = 1)
r3 = enricher(genes3, TERM2GENE = keggGene3, TERM2NAME = keggName3, pAdjustMethod = "fdr", pvalueCutoff = 1, qvalueCutoff = 1)


write.table(as.data.frame(r1), 
            file = "/results/group_DNA_enrichment_results_KEGG.csv", 
            sep = ",", 
            col.names = TRUE, 
            row.names = TRUE)

#  Save Group B enrichment results to a CSV file
write.table(as.data.frame(r2), 
            file = "/results/group_RNA_enrichment_results_KEGG.csv", 
            sep = ",", 
            col.names = TRUE, 
            row.names = TRUE)

write.table(as.data.frame(r3), 
            file = "/results/group_Protein_enrichment_results_KEGG.csv", 
            sep = ",", 
            col.names = TRUE, 
            row.names = TRUE)

#  Debug: inspect p-value ranges (optional, for each r)
print("pvalue summary for r1:")
print(summary(r1$pvalue))
print("pvalue summary for r2:")
print(summary(r2$pvalue))
print("pvalue summary for r3:")
print(summary(r3$pvalue))

# Convert three enrichment results to data frames and add a group column (Group)
df1 <- as.data.frame(r1) %>% mutate(Group = "A")
df2 <- as.data.frame(r2) %>% mutate(Group = "B")
df3 <- as.data.frame(r3) %>% mutate(Group = "C")

# Merge the three data frames
df_combined <- bind_rows(df1, df2, df3)

# Convert GeneRatio to numeric (originally a string like "5/100"; compute the ratio)
df_combined$GeneRatio <- sapply(df_combined$GeneRatio, function(x) {
  parts <- strsplit(x, "/")[[1]]
  as.numeric(parts[1]) / as.numeric(parts[2])
})

# (similar to showCategory=7)
df_top <- df_combined %>%
  group_by(Group) %>%
  top_n(-3, pvalue) %>%
  ungroup()

#  Get the unique pathways across all top-7 entries (union)
all_desc <- unique(df_top$Description)

#  Complete the data frame so each Group has all pathways (missing as NA; do not plot points)
df_combined <- df_top %>%
  complete(Group, Description = all_desc) %>%  # Complete combinations; set missing entries to NA
  group_by(Description) %>%
  fill(ID, BgRatio, .direction = "downup") %>%  # Fill shared columns (e.g., ID, BgRatio)
  ungroup() %>%
  #  For missing entries, explicitly set plot-related fields to NA (already NA by default)
  mutate(
    GeneRatio = if_else(is.na(pvalue), NA_real_, GeneRatio),
    Count = if_else(is.na(pvalue), NA_integer_, Count)
  ) %>%
  #  Order the y-axis by mean GeneRatio (ignoring NA)
  group_by(Description) %>%
  mutate(mean_GeneRatio = mean(GeneRatio, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(Description = reorder(Description, mean_GeneRatio))

# Draw a single dotplot; use facet_wrap to create three vertical panels, each with a different color gradient
p_combined <- ggplot(df_combined) +  
  
  scale_size_continuous(
    range = c(10, 18),
    name  = "Count",
    guide = guide_legend(order = 1)   # 👈  order = 1
  ) +
  
  
  # Group A: data + points + first gradient
  geom_point(data = df_combined %>% filter(Group == "A"), 
             aes(x = GeneRatio, y = Description, size = Count, color = pvalue)) +
  scale_color_gradient(low = "#6697cc", high = "#6697cc", trans = "log10", name = "DNA", guide = guide_colorbar(order = 2, label = FALSE)) +  
  
  # New scale for Group B
  new_scale_color() +
  geom_point(data = df_combined %>% filter(Group == "B"), 
             aes(x = GeneRatio, y = Description, size = Count, color = pvalue)) +
  scale_color_gradient(low = "#df6029", high = "#df6029", trans = "log10", name = "RNA", guide = guide_colorbar(order = 3, label = FALSE)) +  
  
  # New scale for Group C
  new_scale_color() +
  geom_point(data = df_combined %>% filter(Group == "C"), 
             aes(x = GeneRatio, y = Description, size = Count, color = pvalue)) +
  scale_color_gradient(low = "#4cad47", high = "#4cad47", trans = "log10", name = "Protein", guide = guide_colorbar(order = 4, label = FALSE)) +  
  
  # Unified size scale

  
  # Use a shared y-axis
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

# Save the figure
ggsave(
  filename = "/results/DNA_DEG_single_combined_diff_gradients_nolabel_KEGG.png",
  plot = p_combined,
  width = 36,      
  height = 15,
  units = "cm",
  dpi = 600,
  bg = "white"
)

print(p_combined)
dev.off()

