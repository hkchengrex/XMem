import sys
import os
from os import path

from PIL import Image
import numpy as np
from progressbar import progressbar
from multiprocessing import Pool
from functools import partial

new_min_size = 480

def resize_vid_jpeg(vid_name, old_img_dir, new_base_dir):
    vid_path = path.join(old_img_dir, vid_name)
    vid_out_path = path.join(new_base_dir, 'JPEGImages', vid_name)
    os.makedirs(vid_out_path, exist_ok=True)

    for im_name in os.listdir(vid_path):
        hr_im = Image.open(path.join(vid_path, im_name))
        w, h = hr_im.size

        ratio = new_min_size / min(w, h)

        lr_im = hr_im.resize((int(w*ratio), int(h*ratio)), Image.BICUBIC)
        lr_im.save(path.join(vid_out_path, im_name))

def resize_vid_anno(vid_name, old_anno_dir, new_base_dir):
    vid_path = path.join(old_anno_dir, vid_name)
    vid_out_path = path.join(new_base_dir, 'Annotations', vid_name)
    os.makedirs(vid_out_path, exist_ok=True)

    for im_name in os.listdir(vid_path):
        hr_im = Image.open(path.join(vid_path, im_name)).convert('P')
        w, h = hr_im.size

        ratio = new_min_size / min(w, h)

        lr_im = hr_im.resize((int(w*ratio), int(h*ratio)), Image.NEAREST)
        lr_im.save(path.join(vid_out_path, im_name))

def resize_all(in_path, out_path):
    for folder in os.listdir(in_path):

        if folder not in ['JPEGImages', 'Annotations']:
            continue
        folder_path = path.join(in_path, folder)
        videos = os.listdir(folder_path)

        if folder == 'JPEGImages':
            print('Processing images')
            os.makedirs(path.join(out_path, 'JPEGImages'), exist_ok=True)
            partial_func = partial(resize_vid_jpeg, old_img_dir=folder_path, new_base_dir=out_path)

            pool = Pool(processes=16)
            for _ in progressbar(pool.imap_unordered(partial_func, videos), max_value=len(videos)):
                pass
        else:
            print('Processing annotations')
            os.makedirs(path.join(out_path, 'Annotations'), exist_ok=True)
            partial_func = partial(resize_vid_anno, old_anno_dir=folder_path, new_base_dir=out_path)

            pool = Pool(processes=16)
            for _ in progressbar(pool.imap_unordered(partial_func, videos), max_value=len(videos)):
                pass

    print('Done.')

if __name__ == '__main__':
    in_path = sys.argv[1]
    out_path = sys.argv[2]

    resize_all(in_path, out_path)
