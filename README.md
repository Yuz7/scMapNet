## scMapNet: marker-based cell type annotation of single-cell RNA-seq data using vision transfer learning with Tabular-to-Image transformations

<div align="center">  
<img src="static/model_overview.png" width="600">
</div>

Official repo for [scMapNet: marker-based cell type annotation of single-cell RNA-seq data using vision transfer learning with Tabular-to-Image ], which is based on [Treemap](https://cran.r-project.org/web/packages/treemap/treemap.pdf) and [MAE](https://github.com/facebookresearch/mae)

# 🔧Install

[![scipy-1.5.4](https://img.shields.io/badge/scipy-1.5.4-yellowgreen)](https://github.com/scipy/scipy) [![torch-1.8.1](https://img.shields.io/badge/torch-1.8.1-orange)](https://github.com/pytorch/pytorch) [![numpy-1.19.2](https://img.shields.io/badge/numpy-1.19.2-red)](https://github.com/numpy/numpy) [![pandas-1.1.5](https://img.shields.io/badge/pandas-1.1.5-lightgrey)](https://github.com/pandas-dev/pandas) [![scanpy-1.7.2](https://img.shields.io/badge/scanpy-1.7.2-blue)](https://github.com/theislab/scanpy) [![scikit__learn-0.24.2](https://img.shields.io/badge/scikit__learn-0.24.2-green)](https://github.com/scikit-learn/scikit-learn) [![transformers-4.6.1](https://img.shields.io/badge/transformers-4.6.1-yellow)](https://github.com/huggingface/transformers)


# 🌱Fine-tuning with scMapNet weights

To fine tune scMapNet on your own data, follow these steps:

1. Download the scMapNet pre-trained weights:


2. Generate treemap images with single-cell data (download baron dataset [here](https://drive.google.com/file/d/1YrG3xP_NeAlKKM7RzC38m2x9gtlffh6Y/view?usp=drive_link))

```

# -e expression file
# -m prior marker information
# -o output directory
# -f transfer file(symbol to id)
# -t expression file type: seurat(rds) df(csv)
# -s strategies: train test(only test) all(for assess)
# -n thread number

# an example of baron dataset

cd scMapNet
nohup ./generate_image_script.sh -e ../scdataset/sce_baron.rds -m treemap/marker_location.csv -o ../data/ -f treemap/ensemble_ID_transfer_new.csv -d 0 -t seurat -s all -n 48 > log/generate_image.log 2>&1 &

``` 

1. Start fine-tuning (download pretrain weights [here](https://drive.google.com/file/d/1ZlguObYTDVE-H9AqX48lfMSnvP0iTUg5/view?usp=drive_link);use pancreas as example). A fine-tuned checkpoint will be saved during training. Evaluation will be run after training.

```

torchrun \
--standalone --nnodes 1 --nproc_per_node 2 main_finetune.py \
--output_dir ../finetune_pancreas \
--batch_size 16 \
--epochs 25 \
--nb_classes 11 \
--finetune ../pretrain_weights/checkpoint.pth \
--blr 1e-3 --layer_decay 0.75 \
--eval \
--weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --smoothing 0.0 \
--dist_eval --data_path ../data/ > log/fintune_pancreas.log

```

1. For evaluation only 

```

torchrun \
--standalone --nnodes 1 --nproc_per_node 2 main_finetune.py \
--test \
--resume ../finetune_pancreas/checkpoint-22.pth \
--model vit_large_patch16 \
--batch_size 64 --nb_classes 15 \
--data_path ../data/ > log/mae_test.log 2>&1 &

```

