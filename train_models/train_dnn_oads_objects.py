import os

nproc = 18

os.environ["OMP_NUM_THREADS"] = str(nproc)
os.environ["OPENBLAS_NUM_THREADS"] = str(nproc)
os.environ["MKL_NUM_THREADS"] = str(nproc)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(nproc)
os.environ["NUMEXPR_NUM_THREADS"] = str(nproc)

import argparse
import os
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import sys
sys.path.append('/home/nmuller/projects/oads_texture_shape/journal_code')

from base.models import ADE20K_ImageNetModel, get_model_instance_segmentation
from base.training import train_objects, collate_fn
from base.result_manager import ResultManager
from base.datasets import OADS_Objects
  
if __name__ == '__main__':

    c_time = datetime.now().strftime("%d-%m-%y-%H%M%S")

    # Instantiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_dir', help='Path to input directory.', default='data/oads')
    parser.add_argument(
        '--output_dir', help='Path to output directory.', default=f'oads_results/{c_time}')
    parser.add_argument(
        '--n_epochs', help='Number of epochs for training.', default=30)
    parser.add_argument(
        '--optimizer', help='Optimizer to use for training', default='adam')
    parser.add_argument(
        '--criterion', help='What kind of loss function to use. Can be `classes` for simple CrossEntropyLoss on the last layer or `sc_ce_classes` for additional constraint on the first layer to predict SC/CE.', default='classes')
    parser.add_argument(
        '--model_path', help='Path to model to continue training on.', default=None)
    parser.add_argument(
        '--model_type', help='Model to use for training.', default='resnet50')
    parser.add_argument(
        '--image_representation', help='Way images are represented. Can be `RGB`, `COC` (color opponent channels), or `RGBCOC` (stacked RGB and COC)', default='RGB')
    parser.add_argument(
        '--n_processes', help='Number of processes to use.', default=18)
    parser.add_argument(
        '--batch_size', help='Batch size for training.', default=256)
    parser.add_argument(
        '--image_size', help='Batch size for training.', default=400)
    parser.add_argument(
        '--random_state', help='Random state for train test split function. Control for crossvalidation. Can be integer or None. Default 42', default=42)
    parser.add_argument(
        '-imagenet_cropping', help='Whether to first resize and then crop (like imagenet training)', action='store_true')
    parser.add_argument(
        '-save_per_epoch', help='Whether to save model checkpoint after every epoch', action='store_true')

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

    # Setting weird stuff
    torch.multiprocessing.set_start_method('spawn')
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # initialize data access
    # home = '../../data/oads/mini_oads/'
    size = (int(args.image_size), int(args.image_size))

    n_input_channels = 3


    crop_zoom = None

    if args.image_representation == 'RGB':
        file_formats = ['.ARW', '.png']
    elif 'RGB_Zoom' in args.image_representation:
        file_formats = ['.ARW', '.png']
        crop_zoom = int(args.image_representation.split('Zoom-')[-1])
    else:
        print(f"Image representation is not know. Exiting.")
        exit(1)
    print(
        f"Image representation: {args.image_representation}. File format: {file_formats}")


    exclude_classes = ['MASK', "Xtra Class 1", 'Xtra Class 2']

    home = args.input_dir
    image_dir = os.path.join(home, 'oads_arw', 'crops', 'ML')
    
    result_manager = ResultManager(root=args.output_dir)


    output_channels = 21

    batch_size = int(args.batch_size) # 256

    # OADS RGB Crops (400,400) mean, std
    mean = [0.3410, 0.3123, 0.2787]
    std = [0.2362, 0.2252, 0.2162]

    # Get the custom dataset and dataloader
    print(f"Getting data loaders")
    transform_list = []

    if bool(args.imagenet_cropping):
        resize_size = 800 # 336
        transform_list.append(transforms.Resize((resize_size, resize_size)))

        transform_list.append(transforms.CenterCrop(size))

    else:
        transform_list.append(transforms.Resize(size))

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(transform_list)

    all_ids = [
        (c, os.listdir(os.path.join(image_dir, c))) for c in os.listdir(image_dir) if c not in exclude_classes if os.path.isdir(os.path.join(image_dir, c))
    ]

    all_ids = list(set([(x, c) for (c, l) in all_ids for x in l]))
    all_ids = sorted(all_ids)

    # get train, val, test split, using crops if specific size
    random_state = int(args.random_state) if args.random_state != 'None' else None
    # train_ids, val_ids, test_ids = oads.get_train_val_test_split_indices(use_crops=True, random_state=random_state, shuffle=True)
    val_size = 0.1
    test_size = 0.1
    train_ids, test_ids = train_test_split(all_ids, test_size=val_size+test_size, shuffle=True, random_state=random_state)
    test_ids, val_ids = train_test_split(test_ids, test_size=test_size / (val_size+test_size), shuffle=True, random_state=random_state)

    
    print(f"Loaded data with train_ids.shape: {len(train_ids)}")
    print(f"Loaded data with val_ids.shape: {len(val_ids)}")
    print(f"Loaded data with test_ids.shape: {len(test_ids)}")
    print(f"Total of {len(train_ids) + len(val_ids) + len(test_ids)} datapoints.")
    
    # Created custom OADS datasets
    traindataset = OADS_Objects(root=image_dir, item_ids=train_ids, return_index=True, transform=transform, device=device)
    valdataset = OADS_Objects(root=image_dir, item_ids=val_ids, return_index=True, transform=transform, device=device)
    testdataset = OADS_Objects(root=image_dir, item_ids=test_ids, return_index=True, transform=transform, device=device)

    # Create loaders - shuffle training set, but not validation or test set
    trainloader = DataLoader(traindataset, collate_fn=collate_fn,
                                batch_size=batch_size, shuffle=True, num_workers=args.n_processes)
    valloader = DataLoader(valdataset, collate_fn=collate_fn,
                            batch_size=batch_size, shuffle=False, num_workers=args.n_processes)
    testloader = DataLoader(testdataset, collate_fn=collate_fn,
                            batch_size=batch_size, shuffle=False, num_workers=args.n_processes)

    print(f"Loaded data loaders")

    
    if args.model_type == 'resnet50':
        model = resnet50()
        model.conv1 = torch.nn.Conv2d(in_channels=n_input_channels, out_channels=model.conv1.out_channels, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        model.fc = torch.nn.Linear(
            in_features=2048, out_features=output_channels, bias=True)
    
    elif 'resnet50_ade20k_scenes' in args.model_type:
        model = ADE20K_ImageNetModel(output_channels)

        if 'fc' in args.model_type:
            for child in model.children():
                for param in child.parameters():
                    param.requires_grad = False
        elif 'layer4' in args.model_type:
            for child in model.children():
                for param in child.parameters():
                    param.requires_grad = False

            for child in model.model.encoder.stages[3].children():
                for param in child.parameters():
                    param.requires_grad = True

        for child in model.head.children():
            for param in child.parameters():
                param.requires_grad = True

    elif 'resnet50_ade20k_objects' in args.model_type:
        model = resnet50()
        model.conv1 = torch.nn.Conv2d(in_channels=n_input_channels, out_channels=model.conv1.out_channels, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        model.fc = torch.nn.Linear(
            in_features=2048, out_features=3542, bias=True)

        model_path = args.model_path
        # if 'zoom-0' in args.model_type:
        #     model_path = '/home/nmuller/projects/prjs0391/ade20k_objects_results/resnet50/rgb_zoom-0/reps/2024-05-02-193520_400x400/best_model_02-05-24-193527.pth'
        # elif 'zoom-80' in args.model_type:
        #     model_path = '/home/nmuller/projects/prjs0391/ade20k_objects_results/resnet50/rgb_zoom-80/reps/2024-05-02-193213_400x400/best_model_02-05-24-193219.pth'
        # elif 'zoom-150' in args.model_type:
        #     model_path = '/home/nmuller/projects/prjs0391/ade20k_objects_results/resnet50/rgb_zoom-150/reps/2024-05-02-185437_400x400/best_model_02-05-24-185445.pth'
        # else:
        #     print(f"Model type not known. Exiting.")
        #     exit(1)

        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except RuntimeError:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.module

        if 'fc' in args.model_type:
            for child in model.children():
                for param in child.parameters():
                    param.requires_grad = False
        elif 'layer4' in args.model_type:
            for child in model.children():
                for param in child.parameters():
                    param.requires_grad = False

            for child in model.layer4.children():
                for param in child.parameters():
                    param.requires_grad = True

        model.fc = torch.nn.Linear(in_features=2048, out_features=output_channels, bias=True)

    elif 'resnet50_coco_objects' in args.model_type:
        model = resnet50()
        model.conv1 = torch.nn.Conv2d(in_channels=n_input_channels, out_channels=model.conv1.out_channels, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        model.fc = torch.nn.Linear(
            in_features=2048, out_features=80, bias=True)
        
        model_path = args.model_path
        # model_path = '/home/nmuller/projects/prjs0391/coco_objects_results/resnet50/rgb/reps/2024-04-23-093017_400x400/best_model_23-04-24-093029.pth'
        # # if 'zoom-0' in args.model_type:
        # #     model_path = '/home/nmuller/projects/prjs0391/ade20k_objects_results/resnet50/rgb_zoom-0/reps/2024-05-02-193520_400x400/best_model_02-05-24-193527.pth'
        # if 'zoom-80' in args.model_type:
        #     model_path = '/home/nmuller/projects/prjs0391/coco_objects_results/resnet50/rgb_zoom-80/reps/2024-05-05-005047_400x400/best_model_05-05-24-005056.pth'
        # elif 'zoom-150' in args.model_type:
        #     model_path = '/home/nmuller/projects/prjs0391/coco_objects_results/resnet50/rgb_zoom-150/reps/2024-05-05-010418_400x400/best_model_05-05-24-010426.pth'
        # else:
        #     print(f"Model type not known. Exiting.")
        #     exit(1)

        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except RuntimeError:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.module
        
        if 'fc' in args.model_type:
            for child in model.children():
                for param in child.parameters():
                    param.requires_grad = False
        elif 'layer4' in args.model_type:
            for child in model.children():
                for param in child.parameters():
                    param.requires_grad = False

            for child in model.layer4.children():
                for param in child.parameters():
                    param.requires_grad = True

        model.fc = torch.nn.Linear(in_features=2048, out_features=output_channels, bias=True)
        

    elif 'fasterrcnn_resnet50_fpn_coco' in args.model_type:
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

    else:
        raise ValueError(f"Model {args.model_type} not recognized")
        
    print(f"Created model {args.model_type}")

    

    criterion = nn.CrossEntropyLoss()

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
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not recognized")

    # Learning Rate Scheduler
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5)

    info = {
        'training_indices': train_ids,
        'testing_indices': test_ids,
        'validation_indices': val_ids,
        'optimizer': str(optimizer),
        'scheduler': str(plateau_scheduler),
        'model': str(model),
        'args': str(args),
        'device': str(device),
        'criterion': str(criterion),
        'size': size,
        'transform': transform,
        'file_formats': file_formats,
        'image_representation': args.image_representation,
        'input_dir': args.input_dir,
        'output_dir': args.output_dir,
    }


    result_manager.save_result(
        result=info, filename=f'fitting_description_{c_time}.yaml')

    train_objects(model=model, trainloader=trainloader, valloader=valloader, device=device, results={},
            loss_fn=criterion, optimizer=optimizer, n_epochs=args.n_epochs, result_manager=result_manager,
            testloader=testloader, plateau_lr_scheduler=plateau_scheduler, current_time=c_time, save_per_epoch=bool(args.save_per_epoch))
