import os
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
import argparse

def iter(args):
    target_dir, img_id, coco, train_data_dir = args
    ann_ids = coco.getAnnIds(imgIds=img_id)
    # Dictionary: target coco_annotation file for an image
    coco_annotation = coco.loadAnns(ann_ids)

    # path for input image
    path = coco.loadImgs(img_id)[0]['file_name']


    img = Image.open(os.path.join(train_data_dir, path))

    for i in range(len(coco_annotation)):
        class_name = [coco.dataset['categories'][j]['name'] for j in range(len(coco.dataset['categories'])) if coco.dataset['categories'][j]['id'] == coco_annotation[i]['category_id']][0]

        xmin = coco_annotation[i]['bbox'][0]
        ymin = coco_annotation[i]['bbox'][1]
        xmax = xmin + coco_annotation[i]['bbox'][2]
        ymax = ymin + coco_annotation[i]['bbox'][3]

        width = xmax - xmin
        height = ymax - ymin

        if xmin == xmax or ymin == ymax:
            continue

        for zoom in [0.0, 0.8, 1.5]:

            zoom_target_dir = os.path.join(target_dir, f'coco_objects_zoom-{int(zoom*100)}', class_name)
            os.makedirs(zoom_target_dir, exist_ok=True)

            margin = ((width + height) / 2) * zoom
    
            crop_area = [xmin - margin, ymin - margin, xmax + margin, ymax + margin]

            if crop_area[0] < 0:
                crop_area[0] = 0
            if crop_area[1] < 0:
                crop_area[1] = 0
            if crop_area[2] > img.size[0]:
                crop_area[2] = img.size[0]
            if crop_area[3] > img.size[1]:
                crop_area[3] = img.size[1]


            crop = img.crop(crop_area) # (xmin, ymin, xmax, ymax)

            crop.save(os.path.join(zoom_target_dir, f'{img_id}_{i}.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc', type=int, default=16)
    parser.add_argument('--i', type=int)
    parser.add_argument('--target_dir', type=str)
    args = parser.parse_args()

    target_dir = args.target_dir

    root_dir = '/projects/2/managed_datasets/COCO/processed'

    train_data_dir = os.path.join(root_dir, f'train2017')
    test_data_dir = os.path.join(root_dir, f'test2017')
    validation_data_dir = os.path.join(root_dir, f'validation2017')

    coco = COCO(os.path.join(root_dir, f'annotations/instances_train2017.json'))

    ids = list(sorted(coco.imgs.keys()))

    split_size = len(ids) // args.nproc
    ids = ids[split_size*args.i:split_size*(args.i+1)]

    for img_id in tqdm(ids, total=len(ids)):
        iter((target_dir, img_id, coco, train_data_dir))