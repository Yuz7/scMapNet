#脚本参数，获取对应的变量
# e 细胞表达数据地址
# m 靶点数据地址
# o 图片输出地址
# type 是训练数据还是测试数据
# f 基因id文件，这里需要将名称转为id再出图，所以要有id文件
# d 是否需要出图，探究模型可解释性时，只需要图片数据即可，不需要出图
# n 进程数

while getopts e:m:o:t:i:s:n:f:nd: opt
do 
	case "${opt}" in
		e) expression=${OPTARG};;
		m) markers=${OPTARG};;
		o) originOutput=${OPTARG};;
        t) type=${OPTARG};;
        i) meta=${OPTARG};;
        s) style=${OPTARG};;
        n) workers=${OPTARG};;
        f) transfer=${OPTARG};;
        d) draw=${OPTARG};;
	esac
done

if [ "$style" = "train" ];then
output=${originOutput}"/train"
elif [ "$style" = "test" ];then
output=${originOutput}"/test"
elif [ "$style" = "all" ];then
output=${originOutput}"/train"
else
echo "style is illegal"
exit 2
fi

if [ ! -d "$output" ]; then
    mkdir $output
fi

Rscript treemap/merge_script_copy.r -e $expression \
-m $markers -n $workers \
-o $output -t $type -i $meta -f $transfer -d $draw

python treemap/crop_image.py \
--num_workers $workers \
--save_dir $originOutput \
--data_path $output \
--style $style

