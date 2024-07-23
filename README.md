## scMapNet - marker-based cell annotation using vision transfer learning with treemap transforming

<div align="center">  
<img src="static/model_overview.png" width="600">
</div>

Official repo for [scMapNet - marker-based cell annotation using vision transfer learning with treemap transforming], which is based on [Treemap](https://cran.r-project.org/web/packages/treemap/treemap.pdf) and [MAE](https://github.com/facebookresearch/mae)

# 🔧Install

[![scipy-1.5.4](https://img.shields.io/badge/scipy-1.5.4-yellowgreen)](https://github.com/scipy/scipy) [![torch-1.8.1](https://img.shields.io/badge/torch-1.8.1-orange)](https://github.com/pytorch/pytorch) [![numpy-1.19.2](https://img.shields.io/badge/numpy-1.19.2-red)](https://github.com/numpy/numpy) [![pandas-1.1.5](https://img.shields.io/badge/pandas-1.1.5-lightgrey)](https://github.com/pandas-dev/pandas) [![scanpy-1.7.2](https://img.shields.io/badge/scanpy-1.7.2-blue)](https://github.com/theislab/scanpy) [![scikit__learn-0.24.2](https://img.shields.io/badge/scikit__learn-0.24.2-green)](https://github.com/scikit-learn/scikit-learn) [![transformers-4.6.1](https://img.shields.io/badge/transformers-4.6.1-yellow)](https://github.com/huggingface/transformers)


# 🌱Fine-tuning with scMapNet weights

To fine tune scMapNet on your own data, follow these steps:

1. Download the scMapNet pre-trained weights:


2. Generate treemap images with single-cell data

```
cd scMapNet
sh create_image --data xxx.h5ad --type a
``` 

3. Start fine-tuning (use pancreas as example). A fine-tuned checkpoint will be saved during training. Evaluation will be run after training.


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
--dist_eval --data_path ../pancreas_data/ > ../finetune_pancreas/fintune_pancreas.log

```


4. For evaluation only (download data and model checkpoints [here](BENCHMARK.md); change the path below)


```
torchrun \
--standalone --nnodes 1 --nproc_per_node 2 main_finetune.py \
--test \
--resume ../finetune_pancreas/checkpoint-22.pth \
--model vit_large_patch16 \
--batch_size 64 --nb_classes 15 \
--data_path ../pancreas_data/ > ../test_pancreas/mae_test.log 2>&1 &

```


### Load the model and weights (if you want to call the model in your code)

```python
import torch
import models_vit
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_

# call the model
model = models_vit.__dict__['vit_large_patch16'](
    num_classes=2,
    drop_path_rate=0.2,
    global_pool=True,
)

# load RETFound weights
checkpoint = torch.load('scMapNet_weights.pth', map_location='cpu')
checkpoint_model = checkpoint['model']
state_dict = model.state_dict()
for k in ['head.weight', 'head.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

# interpolate position embedding
interpolate_pos_embed(model, checkpoint_model)

# load pre-trained model
msg = model.load_state_dict(checkpoint_model, strict=False)

assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

# manually initialize fc layer
trunc_normal_(model.head.weight, std=2e-5)

print("Model = %s" % str(model))
```
