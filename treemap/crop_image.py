import argparse
import os
from PIL import Image
import multiprocessing
from sklearn.model_selection import KFold, train_test_split
import time
import shutil

def get_args_parser():
    parser = argparse.ArgumentParser('crop image for image classification', add_help=False)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--data_path', default='', type=str,
                        help='image dataset path')
    parser.add_argument('--save_dir', default='', type=str,
                        help='save directory')
    parser.add_argument('--style', default='train', type=str,
                        help='split style')
    return parser

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    n_proc = args.num_workers
    
    save_dir = args.save_dir
    eval_path = args.save_dir + "/eval"
    test_path = args.save_dir + "/test"
    
    if not os.path.exists(os.path.join(eval_path)):
        os.makedirs(os.path.join(eval_path))
    if not os.path.exists(os.path.join(test_path)):
        os.makedirs(os.path.join(test_path))
        
    if args.style == "test":
        save_dir = save_dir + "/test"
    else:
        save_dir = save_dir + "/train"
    image_dir_path = args.data_path
    image_dirs = os.listdir(image_dir_path)
    n_proc = len(image_dirs) if len(image_dirs) < n_proc else n_proc
    
    kf = KFold(n_splits=n_proc)

    proc_indexs = []
    for _, (_, test_index) in enumerate(kf.split(image_dirs)):
        proc_indexs.append(test_index)

    Process_job = []
    def crop_img(indexs, image_dirs):
        for idx in indexs:
            image_dir = image_dirs[idx]
            for file in os.listdir(os.path.join(image_dir_path, image_dir)):
                if not file.endswith(".png"):
                    continue
                lock.acquire()
                if not os.path.exists(os.path.join(save_dir, image_dir)):
                    os.makedirs(os.path.join(save_dir, image_dir))
                lock.release()
                image_path = os.path.join(image_dir_path, image_dir, file)
                image = Image.open(image_path)
                if image.size == (3000, 3000):
                    image = image.crop((56, 26, 2944, 2746))
                elif image.size == (1000, 1000):
                    image = image.crop((18, 8, 982, 916))
                image.save(os.path.join(save_dir, image_dir, file))
            
    begin = time.time()
    print("==============start crop image==============")
    lock = multiprocessing.Lock()
    for i in range(n_proc):
        p = multiprocessing.Process(target = crop_img, args = (proc_indexs[i], image_dirs))
        Process_job.append(p)
        p.start()
    for i in range(n_proc):
        Process_job[i].join()
    end = time.time()
    print("crop image finish with time: %d" % (end - begin))
    
    print("==============start split dataset==============")
    if args.style == "test":
        pass
    elif args.style == "all" or args.style == "train":
        for sub_dir in os.listdir(os.path.join(save_dir)):
            sub_dir_path = os.path.join(save_dir, sub_dir)
            
            test_size = 0.15
            if args.style == "all":
                test_size = 0.2
                if not os.path.exists(os.path.join(test_path, sub_dir)):
                    os.makedirs(os.path.join(test_path, sub_dir))
                file_train, file_test = train_test_split(os.listdir(sub_dir_path),random_state = 44, test_size = test_size)
                for file in file_test:
                    shutil.move(os.path.join(sub_dir_path, file), os.path.join(test_path, sub_dir, file))
                
            if not os.path.exists(os.path.join(eval_path, sub_dir)):
                os.makedirs(os.path.join(eval_path, sub_dir))
            file_train, file_eval = train_test_split(os.listdir(sub_dir_path),random_state = 44, test_size = test_size)
            for file in file_eval:
                shutil.move(os.path.join(sub_dir_path, file), os.path.join(eval_path, sub_dir, file))
        pass