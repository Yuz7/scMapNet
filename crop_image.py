import argparse
import os
from PIL import Image
import multiprocessing
from sklearn.model_selection import KFold
import time
# import pandas as pd


def get_args_parser():
    parser = argparse.ArgumentParser('crop image for image classification', add_help=False)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--data_path', default='', type=str,
                        help='image dataset path')
    parser.add_argument('--save_dir', default='', type=str,
                        help='save directory')
    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    n_proc = args.num_workers

    kf = KFold(n_splits=n_proc)
    save_dir = args.save_dir
    image_dir_path = args.data_path
    image_names = os.listdir(image_dir_path)

    # metadata = pd.read_csv('/home/liqiang/nas230/yuz/scdataset/myeloid/obs.csv')
    # labels = metadata.label
    # cancer = metadata.cancer

    proc_indexs = []
    for _, (_, test_index) in enumerate(kf.split(image_names)):
        proc_indexs.append(test_index)

    Process_job = []
    def crop_img(indexs, image_names):
        for idx in indexs:
            image_name = image_names[idx]
            if not image_name.endswith(".png"):
                continue
            image_split_name = image_name.split("_")[1:]
            image_split_name.pop()
            cls_name = "_".join(image_split_name)
            lock.acquire()
            if not os.path.exists(os.path.join(save_dir, cls_name)):
                os.makedirs(os.path.join(save_dir, cls_name))
            lock.release()
            image_path = os.path.join(image_dir_path, image_name)
            image = Image.open(image_path)
            if image.size == (2888, 2720):
                continue
            image = image.crop((56, 26, 2944, 2746))
            image.save(os.path.join(save_dir, cls_name, image_name.lower()))
            
    begin = time.time()
    lock = multiprocessing.Lock()
    for i in range(n_proc):
        p = multiprocessing.Process(target = crop_img, args = (proc_indexs[i], image_names))
        Process_job.append(p)
        p.start()
    for i in range(n_proc):
        Process_job[i].join()
    end = time.time()
    print("time: %d" % (end - begin))