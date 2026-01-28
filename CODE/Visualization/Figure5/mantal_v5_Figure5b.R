## =========================
## Mantel + environmental correlation plot - using Pearson correlation
## =========================

# ---- Dependencies ----
suppressPackageStartupMessages({
  library(ggplot2)
  library(vegan)
  library(dplyr)
  library(linkET)  
  library(reshape2)
})

theme_update(text = element_text(family = "Arial"))

# ---- Input file paths ----
otu_file <- "/data/data_for_running/Figure5/glul.csv"
env_file <- "/data/data_for_running/Figure5/summary_original.csv"

#---- Read data ----
otu_raw <- read.csv(otu_file, header = TRUE, check.names = FALSE, row.names = 1)
env_raw <- read.csv(env_file, header = TRUE, check.names = FALSE, row.names = 1)

df <- data.frame(t(otu_raw), check.names = FALSE)

common_samples <- intersect(rownames(df), rownames(env_raw))
if (length(common_samples) == 0) {
  stop("df and env have no common samples")
}
df  <- df[common_samples, , drop = FALSE]
env <- env_raw[common_samples, , drop = FALSE]

#  ---- Numeric conversion / cleaning of environmental table ----
env_num <- env %>%
  mutate(across(everything(), ~ suppressWarnings(as.numeric(as.character(.)))))

na_all_cols <- sapply(env_num, function(x) all(is.na(x)))
if (any(na_all_cols)) {
  env_num <- env_num[, !na_all_cols, drop = FALSE]
}

env_scaled <- scale(env_num)

#  ---- Numeric conversion / cleaning of environmental table ----
otu_names <- colnames(df)
env_names <- colnames(env_scaled)

mantel_results <- data.frame()

for (otu in otu_names) {
  for (env_var in env_names) {
    otu_dist <- vegdist(df[, otu, drop = FALSE], method = "euclidean")
    env_dist <- vegdist(env_scaled[, env_var, drop = FALSE], method = "euclidean")
    
    # Mantel test (still use Spearman here because it compares distance matrices)
    mantel_result <- mantel(otu_dist, env_dist, method = "spearman", permutations = 999)
    
    mantel_results <- rbind(mantel_results, data.frame(
      otu = otu,
      env = env_var,
      r = mantel_result$statistic,
      p.value = mantel_result$signif
    ))
  }
}

#  Overall Mantel
spec_dist <- vegdist(df, method = "bray")
env_dist_all <- vegdist(env_scaled, method = "euclidean")
mantel_all <- mantel(spec_dist, env_dist_all, method = "spearman", permutations = 999)

# ---- Binning r / p ----
mantel_results <- mantel_results %>%
  mutate(
    df_r = cut(r, breaks = c(-Inf, 0.1, 0.2, 0.4, Inf),
               labels = c("< 0.1", "0.1 - 0.2", "0.2 - 0.4", ">= 0.4")),
    df_p = cut(p.value, breaks = c(-Inf, 0.01, 0.05, Inf),
               labels = c("< 0.01", "0.01 - 0.05", ">= 0.05"))
  )

print(colnames(env_num))
print(ncol(env_num))
#  ---- [Modification] Correlation matrix among environmental variables - switch to Pearson ----
n_vars <- ncol(env_num)
var_names <- colnames(env_num)

cor_matrix <- matrix(NA, n_vars, n_vars)
p_matrix <- matrix(NA, n_vars, n_vars)
rownames(cor_matrix) <- colnames(cor_matrix) <- var_names
rownames(p_matrix) <- colnames(p_matrix) <- var_names

for (i in 1:n_vars) {
  for (j in 1:n_vars) {
    if (i != j) {
      #[Key change] Switch to Pearson
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

#  Keep only the upper-left triangle
env_cor_df <- env_cor_df %>%
  filter(as.numeric(factor(y, levels = var_names)) <=
           as.numeric(factor(x, levels = var_names)))

print(env_cor_df)  # View the first few rows
print(tail(env_cor_df))  # View the last few rows

# ---- Extract diagonal data ----
diag_data <- data.frame(
  env = var_names,
  env_x = seq_along(var_names),
  r = 1,  
  p.value = 0  
)

# ---- Prepare coordinates for plotting ----
cx <- 25
cy <- 0
r  <- 25

arc_start <- 115    # Arc start angle (degrees)
arc_end   <- 170  # Arc end angle (degrees)
n_pairs   <- 3     # Number of pairs
delta     <- 5     # Angle spacing (degrees) within each pair; smaller = tighter


# Compute base angles: distribute within [arc_start + delta/2, arc_end - delta/2] to avoid out-of-range
base_angles <- seq(arc_start + delta/2, arc_end - delta/2, length.out = n_pairs)

# Offset each base angle by ± delta/2 to form a pair
angles <- as.numeric(t(cbind(base_angles - delta/2, base_angles + delta/2)))

pair_r_add <- c(0.0, 5.0, 2.6)      # Three pairs use r+0, r+1.5, r+3.0 (adjust as needed)
stopifnot(length(pair_r_add) == n_pairs)
r_vec <- rep(r + pair_r_add, each = 2)  # Pair 1 repeated twice, pair 2 repeated twice, pair 3 repeated twice

#  Generate coordinates (aligned with otu_names)
stopifnot(length(otu_names) == length(angles))  # Ensure exactly 6 OTUs
otu_positions <- data.frame(
  otu   = otu_names,
  otu_x = cx + r_vec * cos(angles * pi / 180),
  otu_y = cy + r_vec * sin(angles * pi / 180),
  pair  = rep(paste0("pair", 1:n_pairs), each = 2),   # Optional: label each pair
  spot  = rep(c("A","B"), times = n_pairs)            # Optional: within-pair position 1 / 2
)

env_positions <- data.frame(
  env = env_names,
  env_x = seq_along(env_names)
)

# Merge Mantel results with coordinates
mantel_plot <- mantel_results %>%
  left_join(otu_positions, by = "otu") %>%
  left_join(env_positions, by = "env")

# Significance labels
env_cor_df <- env_cor_df %>%
  mutate(
    sig_label = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01  ~ "**",
      p.value < 0.05  ~ "*",
      TRUE ~ ""
    )
  )

# ---- Create plot ----
p_final <- ggplot() +
  geom_tile(data = env_cor_df, 
            aes(x = as.numeric(factor(x, levels = var_names)),
                y = as.numeric(factor(y, levels = var_names))),
            width = 1,         # Fixed width = 1
            height = 1,        # Fixed height = 1
            color = "#9b9b9b",   # Light-gray border
            linewidth = 0.5,     # Border line width
            fill = NA) +       # No fill (transparent)
  
  # Second layer: colored tiles scaled by r
  geom_tile(data = env_cor_df, 
            aes(x = as.numeric(factor(x, levels = var_names)),
                y = as.numeric(factor(y, levels = var_names)),
                fill = r,
                width = pmin(abs(r) * 1.5, 1.0),      # Set max width to 1.0
                height = pmin(abs(r) * 1.5, 1.0)),    # Set max width to 1.0
            color = NA) +      #  No border
  
  coord_fixed(ratio = 1) +
  
  # Significance markers
  geom_text(data = subset(env_cor_df, p.value < 0.05),
            aes(x = as.numeric(factor(x, levels = var_names)),
                y = as.numeric(factor(y, levels = var_names)),
                label = sig_label),
            color = "black", size = 5, fontface = "bold", family = "Arial") +
  
  # Red points on the diagonal
  geom_point(data = diag_data, aes(x = env_x-1, y = env_x), color = "red", size = 3, shape = 16) +
  
  # Mantel arrows - modified here
  geom_segment(data = dplyr::filter(mantel_plot, df_p %in% c("< 0.01", "0.01 - 0.05")),
               aes(x = otu_x, y = otu_y,
                   xend = env_x-1, yend = env_x,
                   linewidth = df_r,
                   color = df_p),      # Add color mapping
               arrow = NULL) +
  
  # OTU label points
  geom_point(data = otu_positions,
             aes(x = otu_x, y = otu_y),
             color = "black", size = 3, shape = 16) +
  
  geom_text(data = otu_positions,
            aes(x = otu_x, y = otu_y, label = otu),
            vjust = 0.5, hjust = 1.1, size = 6, family = "Arial") +
  
  # Color scheme
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
                               hjust = 0,  # Left-aligned, closer to the main panel
                               margin = margin(r = -5, unit = "pt")),  # Reduce right margin; shift left
    
    # Legend font size control
    legend.title = element_text(size = 15, family = "Arial"),   # Legend title
    legend.text = element_text(size = 15, family = "Arial"),    # Legend text
    legend.key.size = unit(0.8, "cm"),                          # Legend key size
    
    # Title font size
    plot.title = element_text(size = 16, family = "Arial"),
    
    panel.grid = element_blank()
  ) +
  labs(x = "", y = "") +
  ggtitle(sprintf("",
                  mantel_all$statistic, mantel_all$signif))

print(p_final)


# ---- Save PNG ----
out_png <- "/results/mantel_env_correlation_pearson.png"
ggsave(filename = out_png, plot = p_final, width = 15, height = 10, dpi = 600)
message("Saved: ", out_png)

