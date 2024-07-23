library(data.table)
library(parallel)
library(treemap)
library(SingleCellExperiment)
library(magick)

print("==============start script==============")
args <- commandArgs(trailingOnly = TRUE)
input_expression = args[1]
input_markers = args[2]
output_dir = args[3]
input_meta_data = args[4]
print("==============read csv==============")
expression = fread(input_expression, sep = ',', header = TRUE)
marker = fread(input_markers, header = TRUE, sep = ",")
metadata = fread(input_meta_data, header = TRUE, sep = ",")

palette.colors = hcl.colors(10000, palette = "YlOrBr", rev = TRUE)
merge_func <- function(colidx, marker_data, input_expression, output_filename) {
	
	merge_data = merge(marker_data, input_expression, by = 'id', all.x = T, no.dups = FALSE)
	merge_data[is.na(merge_data)] = 0
	merge_data$SIZE = 1

	sprintf("==============start create and save png, idx:%d==============",colidx)
	png(filename = output_filename, width = 10, height = 10, units = "in", bg = "white", res = 200)
	result = treemap(merge_data,
		index = c("organ","cell type","id"),
		vSize = "SIZE", #指定面积大小的列
		vColor = "expression", #指定颜色深浅的列
		type = "manual",
		palette = palette.colors,
		title.legend = NA,
		fontsize.labels = c(0,0), #设置标签字体大小
		align.labels = list(c("center", "center"), c("left", "top")), #设置标签对齐的方式
		border.col = c("black","black"), #设置边框的颜色 
		border.lwds = c(1,1), #设置边框的线条的宽度
		title = ""
		)
	# fwrite(result$tm, file = output_filename)
	dev.off()
}

parrallel_func <- function(colidx) {
	label = labels[colidx]
	cancer = dataset[colidx]
	input_expression = data.frame(row.names = 1:ncol(expression))
	input_expression$expression = t(expression[colidx,])
	input_expression$id = as.character(geneid)
	marker$id = as.character(marker$id)
	output_filename = paste0(output_dir, "/", label, "/", cancer, "_", label, "_" , colidx, ".png")
	merge_func(colidx, marker, input_expression, output_filename)
}

print("==============merge data==============")
dataset = metadata$orig
labels = metadata$label
for (dir_name in unique(labels)) { 
  create.dir <- paste0(output_dir,"/", dir_name)
  if (!dir.exists(create.dir)) {
    dir.create(paste0(output_dir,"/", dir_name))
  }
}
transfer.data <- fread('/home/liqiang/nas230/yuz/scdataset/expression/ensemble_ID_transfer_new.csv', sep = ',', header = TRUE)
symbol <- as.data.frame(colnames(expression))
colnames(symbol)[1] <- "symbol"
colnames(transfer.data) <- c('ensgid','geneid','symbol')
transfer.data <- transfer.data[!duplicated(transfer.data$symbol), ]
geneid <- dplyr::left_join(symbol, transfer.data, by = "symbol")$geneid
start <- Sys.time()
mclapply(1:nrow(expression), FUN = parrallel_func, mc.cores = detectCores())
lapply.time <- Sys.time() - start
print(lapply.time)



