if (!require('treemap', quietly=TRUE))
	install.packages('treemap')
if (!require('data.table', quietly=TRUE))
	install.packages('data.table')
if (!require('parallel', quietly=TRUE))
	install.packages('parallel')
if (!require('magick', quietly=TRUE))
	install.packages('magick')
if (!require('getopt', quietly=TRUE))
	install.packages('getopt')

if (!require("BiocManager", quietly = TRUE))
	install.packages("BiocManager")

if (!require("SingleCellExperiment", quietly = TRUE))
	BiocManager::install('SingleCellExperiment')

library(treemap)
library(data.table)
library(parallel)
library(magick)
library(getopt)
library(SingleCellExperiment)

print("==============start script==============")

spec = matrix(c(
	'expression', 'e', 1, 'character',
	'markers', 'm', 1, 'character',
	'output_dir', 'o', 1, 'character',
	'type', 't', 1, 'character',
	"numworkers", 'n', 1, 'integer',
	'meta', 'i', 2, 'character',
	'transferdb', 'f', 2, 'character',
	'draw', 'd', 2, 'integer'
), byrow = TRUE, ncol = 4)

opt = getopt(spec)

if (is.null(opt$expression)) stop('expression can not be empty')
if (is.null(opt$markers)) stop('markers can not be empty')
if (is.null(opt$output_dir)) stop('output_dir can not be empty')
if (is.null(opt$type)) stop('type can not be empty')

input_expression = opt$expression
input_markers = opt$markers
output_dir = opt$output_dir
is_draw = opt$draw > 0

if (opt$type == 'df') {
	if (is.null(opt$meta)) {
		stop("meta can not be empty if type equal df")
	}
	input_meta_data = opt$meta
}

print("==============read csv==============")
marker = fread(input_markers, header = TRUE, sep = ",")

palette.colors = hcl.colors(10000, palette = "YlOrBr", rev = TRUE)
merge_func <- function(colidx, marker_data, input_expression, output_filename) {
	
	merge_data = merge(marker_data, input_expression, by = 'id', all.x = T, no.dups = FALSE)
	merge_data[is.na(merge_data)] = 0
	merge_data$SIZE = 1

	sprintf("==============start create and save png, idx:%d==============",colidx)
	if (is_draw) {
		png(filename = output_filename, width = 10, height = 10, units = "in", bg = "white", res = 100)
	}
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
		title = "",
		draw = is_draw
		)
	if (is_draw) {
		dev.off()
	} else {
		output = result$tm[c('organ', 'cell type', 'id', 'vColor', 'level', 'x0', 'y0', 'w', 'h')]
		output = output[!is.na(output$id), ]
		fwrite(output, file = output_filename)
	}
}

parrallel_func <- function(colidx) {
	label = tolower(labels[colidx])
	input_expression = data.frame(row.names = 1:ncol(expression))
	input_expression$expression = as.vector(t(expression[colidx,]))
	input_expression$id = as.character(geneid)
	marker$id = as.character(marker$id)
	suffix = ".csv.gz"
	if (is_draw) {
		suffix = ".png"
	}
	output_filename = paste0(output_dir, "/", label, "/", label, "_" , colidx, suffix)
	merge_func(colidx, marker, input_expression, output_filename)
}

print("==============image generation==============")
start <- Sys.time()
cores = detectCores()
if (opt$numworkers < cores) {
	cores = opt$numworkers
}

if (opt$type == "seurat") {
	transfer.data <- fread(opt$transferdb, sep = ',', header = TRUE)
	colnames(transfer.data) <- c('ensgid','geneid','symbol')
	transfer.data <- transfer.data[!duplicated(transfer.data$symbol), ]

	dataset <- readRDS(input_expression)
	if (sum(names(assays(dataset)) %in% "normcounts") > 0) {
		expression <- assay(dataset, "normcounts")
	} else if (sum(names(assays(dataset)) %in% "logcounts") > 0) {
		expression <- assay(dataset, "logcounts")
	}
	labels <- dataset$label
	for (dir_name in unique(labels)) {
		create.dir <- paste0(output_dir,"/", dir_name)
		if (!dir.exists(create.dir)) {
			dir.create(paste0(output_dir,"/", dir_name))
		}
	}
	symbol <- as.data.frame(rowData(dataset)$symbol)
	colnames(symbol)[1] <- "symbol"
	geneid <- dplyr::left_join(symbol, transfer.data, by = "symbol")$geneid

	expression = t(expression)
} else {
	expression = fread(input_expression, sep = ',', header = TRUE)
	metadata = fread(input_meta_data, header = TRUE, sep = ",")

	labels = metadata$label
	for (dir_name in unique(labels)) {
		create.dir <- paste0(output_dir,"/", dir_name)
		if (!dir.exists(create.dir)) {
			dir.create(paste0(output_dir,"/", dir_name))
		}
	}

	geneid = colnames(expression)
}



mclapply(1:nrow(expression), FUN = parrallel_func, mc.cores = cores)
lapply.time <- Sys.time() - start
print(lapply.time)



