import time
import os
nproc = 16
os.environ['MKL_NUM_THREADS'] = str(nproc)
os.environ['OPENBLAS_NUM_THREADS'] = str(nproc)
os.environ["NUMEXPR_NUM_THREADS"] = str(nproc)
from pathlib import Path
import multiprocessing
import tqdm
import rawpy
from PIL import Image
import json
import argparse

def make_zoom_crop(img, ann, zoom_fraction):
    ((left, top), (right, bottom)) = ann['points']['exterior']
    width = right*4 - left*4
    height = bottom*4 - top*4


    margin = ((width + height) / 2) * zoom_fraction
    
    crop_area = [left*4 - margin, top*4 - margin, right*4 + margin, bottom*4 + margin]

    if crop_area[0] < 0:
        crop_area[0] = 0
    if crop_area[1] < 0:
        crop_area[1] = 0
    if crop_area[2] > img.size[0]:
        crop_area[2] = img.size[0]
    if crop_area[3] > img.size[1]:
        crop_area[3] = img.size[1]

    return img.crop(crop_area)

def iterate(args):
    data_dir, crop_dir, annotation_paths, image_id, zooms = args

    with open(f'{data_dir}/oads_arw/ARW/{image_id}.ARW', 'rb') as f:
        img = rawpy.imread(f).postprocess()

    img = Image.fromarray(img)

    with open(annotation_paths[image_id], 'r') as f:
        annotations = json.load(f)

    for index, ann in tqdm.tqdm(enumerate(annotations['objects']), disable=True):
        c = ann['classTitle']

        for zoom in zooms:
            crop = make_zoom_crop(img, ann, zoom)
            os.makedirs(os.path.join(crop_dir, f'ML_zoom-{int(zoom*100)}', c), exist_ok=True)
            crop.save(os.path.join(crop_dir, f'ML_zoom-{int(zoom*100)}', c, f'{image_id}_{index}.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc', type=int, default=16)
    parser.add_argument('--i', type=int)
    parser.add_argument('--target_dir', type=str)
    args = parser.parse_args()

    home_path = os.path.expanduser('~')
    data_dir = f'{home_path}/projects/data/oads'

    target_dir = args.target_dir

    
    image_ids = [x.split('.')[0] for x in os.listdir(os.path.join(data_dir, 'oads_arw', 'ARW'))]

    annotation_paths = {}
    for dataset in os.listdir(os.path.join(data_dir, 'oads_annotations')):
        if dataset.endswith('.json'):
            continue
        for file in os.listdir(os.path.join(data_dir, 'oads_annotations',  dataset, 'ann')):
            if file.split('.')[0] in image_ids:
              annotation_paths[file.split('.')[0]] = os.path.join(data_dir, 'oads_annotations',  dataset, 'ann', file)

    

    # crop_dir = os.path.join(data_dir, 'oads_arw', 'crops')
    # os.makedirs(crop_dir, exist_ok=True)

    old_image_ids = [image_id for image_id in image_ids if image_id in annotation_paths.keys()]

    image_ids = []
    # annotations = []
    for image_id in old_image_ids:
        with open(annotation_paths[image_id], 'r') as f:
            anns = json.load(f)
            # annotations.append(anns)

        all = True
        for i, ann in enumerate(anns['objects']):
            c = ann['classTitle']
            if os.path.exists(os.path.join(crop_dir, 'ML_zoom-100', c, f'{image_id}_{i}.png')):
                continue
            all = False
        if not all:
            image_ids.append(image_id)

    image_ids = [x for x in image_ids if x in annotation_paths.keys()]

    split_size = len(image_ids) // args.nproc
    image_ids = image_ids[split_size*args.i:split_size*(args.i+1)]

    zooms = [0.8, 1.0, 1.5]# 0.1, 0.3, 0.5]

    for image_id in tqdm.tqdm(image_ids):
        iterate((data_dir, target_dir, annotation_paths, image_id, zooms))

