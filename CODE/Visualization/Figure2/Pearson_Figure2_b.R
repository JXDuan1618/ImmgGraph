library(ggplot2)

##  === Paths: change to your actual paths ===
input_csv <- "/data/data_for_running/Figure2/RNA_Pearson.csv"  # Input CSV path
output_png <- "/rssults/RNA_Pearson.png" # Input CSV path

## Read CSV (use base R to avoid installing extra packages)
dat <- read.csv(input_csv, stringsAsFactors = FALSE, check.names = FALSE)

## Check whether required columns exist
required_cols <- c("valid_imrna_1","valid_output_rna_1","valid_imrna_2","valid_output_rna_2")
missing <- setdiff(required_cols, names(dat))
if (length(missing) > 0) {
  stop("Missing columns: ", paste(missing, collapse = ", "), 
       "\nPlease make sure the CSV column names match the code. Current column names are:\n", paste(names(dat), collapse = ", "))
}

## Bind into a long table (two sets of data)
df <- rbind(
  data.frame(RNA_label = dat$valid_imrna_1, RNA_predict = dat$valid_output_rna_1, set = "set1"),
  data.frame(RNA_label = dat$valid_imrna_2, RNA_predict = dat$valid_output_rna_2, set = "set2")
)

## Convert to numeric and remove missing/invalid values
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
    values = c("set1" = "#1b9e77",  # Blue
               "set2" = "#d95f02"),  # Red
    labels = c("set1" = "PC1", "set2" = "PC2")  # Set legend labels
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "none",
    #legend.position = "bottom",  # Legend position
    panel.background = element_rect(fill = "white", colour = NA),  # Legend position
    panel.border     = element_rect(colour = "black", fill = NA, linewidth = 5), # Panel border
    axis.text.x      = element_blank(),    # Remove x-axis tick labels
    plot.title       = element_blank(),    # Remove title
    axis.ticks.y     = element_line(size = 1),  # Extend y-axis ticks
    axis.line        = element_line(colour = "black"),  # Y-axis line
    axis.text.y      = element_text(size = 70),  # Larger y-axis tick text
    axis.title.x     = element_text(size = 70, margin = margin(t = 30)),  # Larger x-axis title text
    axis.title.y     = element_text(size = 70, margin = margin(r = 30)),   # Larger y-axis title text
    legend.text      = element_text(size = 20),  # Legend text size
    legend.title     = element_text(size = 20),  # Legend title size
    legend.key.size  = unit(1.5, "cm"),  # Legend key size
    panel.grid.major = element_blank(),  # Remove major grid lines
    panel.grid.minor = element_blank()   # Remove minor grid lines
  ) +
  ylim(min(df$RNA_predict) - 0.1, max(df$RNA_predict) + 0.1) +  # Extend y-axis range
  annotate("text", x = max(df$RNA_label), y = min(df$RNA_predict), label = "Transcriptomics", 
           hjust = 0.97, vjust = 0.9, size = 23, color = "black")  # Add text annotation

p <- p + scale_y_continuous(
  breaks = seq(floor(min(df$RNA_predict)) + 1, ceiling(max(df$RNA_predict)) - 1, by = 1) # Remove the topmost tick
)

# Save as PNG (600 dpi)
dir.create(dirname(output_png), recursive = TRUE, showWarnings = FALSE)
ggsave(filename = output_png, plot = p, width = 9.5, height = 8, units = "in", dpi = 600, bg = "white")

# Also print to the plotting window and show the save path
print(p)
cat("Saved to:", output_png, "\n")

