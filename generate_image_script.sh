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

