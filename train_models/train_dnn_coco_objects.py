import os
from pycocotools.coco import COCO
import argparse
from datetime import datetime
import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from base.models import get_model_instance_segmentation
from base.datasets import COCO_Objects
from base.training import train_objects, collate_fn
from base.result_manager import ResultManager

def get_image_info(root, split):
    dir = os.path.join(root, split)

    coco = COCO('/projects/2/managed_datasets/COCO/processed/annotations/instances_train2017.json')
    ids = list(sorted(coco.imgs.keys()))

    all_images = {}

    # images = os.listdir(os.path.join(dir))

    for img_id in tqdm.tqdm(ids, total=len(ids)):
        # img_id, coco, train_data_dir = args
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        for index in range(len(coco_annotation)):
            label = [coco.dataset['categories'][j]['name'] for j in range(len(coco.dataset['categories'])) if coco.dataset['categories'][j]['id'] == coco_annotation[index]['category_id']][0]

            xmin = coco_annotation[index]['bbox'][0]
            ymin = coco_annotation[index]['bbox'][1]
            xmax = xmin + coco_annotation[index]['bbox'][2]
            ymax = ymin + coco_annotation[index]['bbox'][3]

            all_images[f'{img_id}_{index}'] = {'bbox': (xmin, ymin, xmax, ymax), 'label': label, 'path': os.path.join(dir, path), 'index': index}

    return all_images



if __name__ == '__main__':

    c_time = datetime.now().strftime("%d-%m-%y-%H%M%S")

    # Instantiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--output_dir', help='Path to output directory.', default=f'/home/nmuller/projects/prjs0391/coco_objects_results/{c_time}')
    parser.add_argument(
        '--model_name', help='Model name to save the model under', default=f'model_{c_time}')
    parser.add_argument(
        '--n_epochs', help='Number of epochs for training.', default=30)
    parser.add_argument(
        '--optimizer', help='Optimizer to use for training', default='adam')

    parser.add_argument(
        '--model_type', help='Model to use for training.', default='fasterrcnn_resnet50_fpn_coco')
    parser.add_argument(
        '--image_representation', help='Way images are represented. Can be `RGB`, `COC` (color opponent channels), or `RGBCOC` (stacked RGB and COC)', default='RGB')
    parser.add_argument(
        '--n_processes', help='Number of processes to use.', default=18)
    parser.add_argument(
        '--batch_size', help='Batch size for training.', default=256)
    parser.add_argument(
        '--image_size', help='Batch size for training.', default=400)
    parser.add_argument(
        '--zoom', help='Percentage of object crop zoom out', default=0, type=int)



    args = parser.parse_args()

    # Check if GPU is available
    if not torch.cuda.is_available():
        print("GPU not available. Exiting ....")
        device = torch.device('cpu')
        # exit(1)
    else:
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        print(f"Using GPU: {device}!")

    torch.multiprocessing.set_start_method('spawn')


    size = (int(args.image_size), int(args.image_size))
    n_input_channels = 3
    batch_size = int(args.batch_size) # 256

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    result_manager = ResultManager(root=args.output_dir)

    training_images = get_image_info('data/COCO/processed', 'train2017')


    # Get the custom dataset and dataloader
    print(f"Getting data loaders")
    transform_list = []

    transform_list.append(transforms.Resize(size)) # , interpolation=transforms.InterpolationMode.NEAREST if bool(args.pre_resize) else transforms.InterpolationMode.BILINEAR
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(transform_list)


    traindataset = COCO_Objects(training_images, transform=transform, zoom_fraction=args.zoom/100)
    print(f'Number of classes: {len(traindataset.classes)}')


    train_size = int(len(traindataset) * 0.8)
    test_size = len(traindataset) - train_size
    traindataset, testdataset  = random_split(traindataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    test_size = int(len(testdataset) * 0.5)
    val_size = len(testdataset) - test_size
    testdataset, valdataset = random_split(testdataset, [test_size, val_size], generator=torch.Generator().manual_seed(42))

    print(f"Train size: {len(traindataset)}")
    print(f"Val size: {len(valdataset)}")
    print(f"Test size: {len(testdataset)}")



    # Create loaders - shuffle training set, but not validation or test set
    trainloader = DataLoader(traindataset, collate_fn=collate_fn,
                                batch_size=batch_size, shuffle=True, num_workers=int(args.n_processes))
    valloader = DataLoader(valdataset, collate_fn=collate_fn,
                            batch_size=batch_size, shuffle=False, num_workers=int(args.n_processes))
    testloader = DataLoader(testdataset, collate_fn=collate_fn,
                            batch_size=batch_size, shuffle=False, num_workers=int(args.n_processes))

    print(f"Loaded data loaders")

    output_channels = 80

    # 'fasterrcnn_resnet50_fpn_coco' in args.model_type:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=0, world_size=1)
    
    num_classes = 100
    # num_epochs = 100
    model = get_model_instance_segmentation(num_classes)
    torch.cuda.set_device(device)
    model.cuda(device)
    model = DDP(model, device_ids=[0])
    faster_ccn_model_path = '/home/nmuller/projects/coco_results/27-02-24-173725/best_model_Tue_Feb_27_17:38:03_2024.pth'
    model.load_state_dict(torch.load(faster_ccn_model_path, map_location='cuda:0'))
    model = model.module

    dist.destroy_process_group()

    model = model.backbone

    # model.body = 

    # dict_to_tensor = lambda x: x['out']
    class DictToTensor(nn.Module):
        def __init__(self):
            super(DictToTensor, self).__init__()
        def forward(self, x):
            return x['3']

    

    if 'fc' in args.model_type:
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = False
    elif 'layer4' in args.model_type:
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = False

        for child in model.body.layer4.children():
            for param in child.parameters():
                param.requires_grad = True


    model.fpn = torch.nn.Sequential(
        DictToTensor(),
        torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        torch.nn.Flatten(1),
        torch.nn.Linear(in_features=2048, out_features=output_channels, bias=True)
    )
    
    for child in model.fpn.children():
        for param in child.parameters():
            param.requires_grad = True


    criterion = nn.CrossEntropyLoss()

    n_epochs = int(args.n_epochs)
    
    # Use DataParallel to make use of multiple GPUs if available
    # if type(model) is not torch.nn.DataParallel:
    #     model = model.to('cuda:0')
    #     model = torch.nn.DataParallel(model, device_ids=[0,1])
    # else:
    # model = model.to(device)  # , dtype=torch.float32
            
    # if 'fasterrcnn_resnet50_fpn_coco' in args.model_type:
    #     model = model.module

    model = torch.nn.parallel.DataParallel(model)
    model = model.to(device)  

    

    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    # Learning Rate Scheduler
    # plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 'min', patience=5)
    plateau_scheduler = None
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    info = {
        'train_size': len(traindataset),
        'test_size': len(testdataset),
        'val_size': len(valdataset),
        'optimizer': str(optimizer),
        'scheduler': str(lr_scheduler),
        'model': str(model),
        'args': str(args),
        'device': str(device),
        'criterion': str(criterion),
        'size': size,
        'transform': transform,
    }


    result_manager.save_result(
        result=info, filename=f'fitting_description_{c_time}.yaml')

    train_objects(model=model, trainloader=trainloader, valloader=valloader, device=device,
            loss_fn=criterion, optimizer=optimizer, n_epochs=n_epochs, result_manager=result_manager,
            testloader=testloader, plateau_lr_scheduler=plateau_scheduler, lr_scheduler=lr_scheduler, current_time=c_time, save_per_epoch=bool(args.save_per_epoch))
