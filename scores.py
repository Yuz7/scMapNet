import pandas as pd
import numpy as np
import scanpy as sc
import cv2
from PIL import Image
# import pyreadr

geneid_to_symbol = pd.read_csv('/home/liqiang/nas230/yuz/scdataset/expression/ensemble_ID_transfer_new.csv')

def transfer(id):
    if np.isnan(id):
        return 'none'
    symbol = geneid_to_symbol[geneid_to_symbol['NCBI_Gene_ID'] == int(id)].SYMBOL.values
    if len(symbol) > 0:
        return str(symbol[0])
    else:
        return 'none'
    
def extract_genes(data, left1, left2, right1, right2, wide, hight):
    return data[data.y0 >= (left1 / hight)][data.y1 < (left2 / hight)][data.x0 >= (right1 / wide)][data.x1 < (right2 / wide)]

import os
import time
import warnings

def compute_scores(target_class, row_idx_start, imps):
    result = pd.DataFrame({'gene':[],'score':[]})
    result.set_index('gene', inplace = True)
    target_dir = os.path.join(root_path, target_class)
    if not os.path.isdir(target_dir):
        return
    symbols = None
    row_idx = row_idx_start
    timepoint0 = time.time()
    for file_name in sorted(os.listdir(target_dir)):
        data = pd.read_csv(os.path.join(target_dir, file_name))
        data['x1'] = data.x0 + data.w
        data['y1'] = data.y0 + data.h
        if symbols == None:
            symbols = list(map(transfer, data.id.values))
        data['symbol'] = symbols
        imp = imps[row_idx].X / imps[row_idx].X.max()
        sort_imp = sorted(imp[0], reverse=True)
        mask = imp.reshape(14,14)
        imp_symbols = pd.DataFrame({'gene':[], 'score':[]})
        for i in range(10):
            y = np.where(mask == sort_imp[i])[0]
            x = np.where(mask == sort_imp[i])[1]
            pot_genes = extract_genes(data, y[0], y[-1] + 1, x[0], x[-1] + 1, 14, 14)
            pot_genes = pot_genes[pot_genes.symbol != 'none'][pot_genes.vColor > 0]
            imp_symbols = pd.concat((imp_symbols,pd.DataFrame({'gene':pot_genes.symbol.values, 'score':sort_imp[i]})))
        imp_symbols.drop_duplicates(subset=['gene'],inplace=True)
        imp_symbols.set_index('gene', inplace= True)
        if result.empty:
            result = imp_symbols
        else:
            result = pd.concat((result, imp_symbols), axis = 1).fillna(0)
        row_idx += 1
    timepoint3 = time.time()
    print(target_class + ' finished')
    print(timepoint3 - timepoint0)
    result.to_csv('baron_' + target_class + '_10_scores.csv', index = True, header = True)

import multiprocessing

warnings.filterwarnings('ignore')
root_path = "/home/liqiang/nas230/yuz/scdataset/pancreas/baron_image_source/test"
classes = sorted(entry.name for entry in os.scandir(root_path) if entry.is_dir())
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
begin = time.time()

row_idx_start = []
Process_job = []
for target_class in sorted(class_to_idx.keys()):
    target_dir = os.path.join(root_path, target_class)
    if len(row_idx_start) == 0:
        row_idx_start.append(0)
        continue
    row_idx_start.append(len(sorted(os.listdir(target_dir))))

imp_h5ad_addr = '/home/liqiang/nas230/yuz/scMapNet_output/job_output_test/job_output_test_pancreas/sce_baron/baron_embed.h5ad'
imps = sc.read_h5ad(imp_h5ad_addr)

n_proc = len(sorted(class_to_idx.keys()))
for i in range(n_proc):
    p = multiprocessing.Process(target = compute_scores, args = (sorted(class_to_idx.keys())[i], row_idx_start[i], imps))
    Process_job.append(p)
    p.start()
for i in range(n_proc):
    Process_job[i].join()
end = time.time()
print("crop image finish with time: %d" % (end - begin))