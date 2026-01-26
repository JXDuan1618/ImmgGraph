# install.packages("plot3D") 
library(plot3D)
# 读数据 
data <- read.csv("E:/Multi-omic Immunity/GCN_immune/scripts/LGG/picture/Clustering/patient_new_clusters_4.csv") 
# 列 
imdna <- data$tSNE.1 
imrna <- data$tSNE.2 
impro <- data$tSNE.3 
cluster_label <- factor(data$new_cluster) # 例如 A,B,C,D,E 
# 颜色 
colors_main <- c("#8583A9", "#CA8BA8", "#A0BDD5", "#EFC57F", "#FF9933") 
colors_proj <- c("#8583A9", "#CA8BA8", "#A0BDD5", "#EFC57F", "#FF9961") 
point_size <- 1.5 

cluster_int <- as.integer(cluster_label) # 投影用的调色板：与主点一致，但更浅一些，避免喧宾夺主 
pal_proj <- grDevices::adjustcolor(colors_proj[seq_len(K)], alpha.f = 1) 
panelfirst <- function(pmat){ 
  ## 1) 地板：XY 平面（z 取最小值） 
  z0 <- min(impro, na.rm = TRUE) 
  XY <- trans3D(x = imrna, y = imdna, z = rep(z0, length(impro)), pmat = pmat) 
  scatter2D(XY$x, XY$y, 
            colvar = cluster_int, 
            col = pal_proj, 
            pch = 16, 
            cex = point_size * 0.5, 
            add = TRUE, 
            colkey = FALSE, 
            clim = clim, 
            breaks = breaks) 
  ## 2) 侧墙：YZ 平面（x 取最小值，作为“深度”墙） 
  x0 <- min(imrna, na.rm = TRUE) 
  YZ <- trans3D(x = rep(x0, length(impro)), y = imdna, z = impro, pmat = pmat) 
  scatter2D(YZ$x, YZ$y, 
            colvar = cluster_int, 
            col = pal_proj, 
            pch = 16, 
            cex = point_size * 0.5, 
            add = TRUE, 
            colkey = FALSE, 
            clim = clim, breaks = breaks) 
  } 
# 分段色条参数 
lev <- levels(cluster_label) 
K <- length(lev) 
pal <- colors_main[seq_len(K)] # 与 levels 对应 
lev_bar <- rev(lev) # 顶端 E, 底端 A（按需改） 
pal_bar <- rev(pal) 
clim <- c(0.5, K + 0.5) 
breaks <- seq(clim[1], clim[2], length.out = K + 1) 
# 输出 
png("E:/Multi-omic Immunity/GCN_immune/scripts/LGG/picture/Clustering/3D_scatter_plot_PC2_with_borders.png", 
    width = 8, height = 8, units = "in", res = 600) 
# 关键：不要用 xpd = NA（避免分隔线越界） 
par(mfrow = 1, mar = c(5, 5, 4, 3), xpd = FALSE) 
# 主散点：用 colvar + 调色板着色，色条离散分段 
scatter3D(x = imrna, y = imdna, z = impro, 
          colvar = as.integer(cluster_label), 
          col = pal, 
          pch = 16, 
          cex = point_size, 
          bty = "b2", 
          xlab = "tSNE-1", ylab = "tSNE-2", zlab = "tSNE-3", 
          main = "3D Scatter Plot with Projections and Borders", 
          ticktype = "detailed", 
          panel.first = panelfirst, 
          d = 6, 
          colkey = list( 
            clim = clim, 
            col = pal_bar, 
            breaks = breaks, # 分段 
            at = 1:K, 
            labels = lev_bar, # A~E 等标签 
            side = 4, 
            length = 0.6, 
            width = 0.6, 
            dist = 0.03, 
            addlines = TRUE, # 有分段线，但不再跨越 
            clab = "Cluster" ) 
          ) 
# 叠加黑色边框 
points3D(x = imrna, y = imdna, z = impro, add = TRUE, pch = 1, cex = point_size * 1.01, col = "black") 

dev.off()
