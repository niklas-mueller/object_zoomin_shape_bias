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
import tqdm
import json

import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split


from base.datasets import ADE20K_Objects
from base.training import train_objects, collate_fn
from base.result_manager import ResultManager

def get_image_info(root, split):
    dir = os.path.join(root, split)

    supclasses = os.listdir(dir)

    all_images = {}

    for supclass in tqdm.tqdm(supclasses, total=len(supclasses)):
        subclasses = os.listdir(os.path.join(dir, supclass))

        for subclass in subclasses:
            images = os.listdir(os.path.join(dir, supclass, subclass))

            for image_filename in [image for image in images if image.endswith('.json')]:
                path = os.path.join(dir, supclass, subclass, image_filename)

                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                except UnicodeDecodeError:
                    continue

                # index = 20
                for index in range(len(data['annotation']['object'])):
                    x_min, y_min = min(data['annotation']['object'][index]['polygon']['x']), min(data['annotation']['object'][index]['polygon']['y'])
                    x_max, y_max = max(data['annotation']['object'][index]['polygon']['x']), max(data['annotation']['object'][index]['polygon']['y'])
                    label = data['annotation']['object'][index]['name']

                    all_images[f'{image_filename}_{index}'] = {'bbox': (x_min, y_min, x_max, y_max), 'label': label, 'path': path, 'index': index, 'supclass': supclass, 'subclass': subclass}

    return all_images

    
if __name__ == '__main__':

    c_time = datetime.now().strftime("%d-%m-%y-%H%M%S")

    # Instantiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--output_dir', help='Path to output directory.', default=f'trained_models/ade20k_objects_results/{c_time}')
    parser.add_argument(
        '--n_epochs', help='Number of epochs for training.', default=30)
    parser.add_argument(
        '--optimizer', help='Optimizer to use for training', default='adam')
    parser.add_argument(
        '--model_type', help='Model to use for training.', default='resnet50')
    parser.add_argument(
        '--n_processes', help='Number of processes to use.', default=18)
    parser.add_argument(
        '--batch_size', help='Batch size for training.', default=256)
    parser.add_argument(
        '--image_size', help='Batch size for training.', default=400)
    parser.add_argument(
        '--random_state', help='Random state for train test split function. Control for crossvalidation. Can be integer or None. Default 42', default=42)
    parser.add_argument(
        '--zoom', help='Percentage of object crop zoom out', default=0, type=int)
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

    
    torch.multiprocessing.set_start_method('spawn')
    
    size = (int(args.image_size), int(args.image_size))
    n_input_channels = 3
    batch_size = int(args.batch_size) # 256

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


    result_manager = ResultManager(root=args.output_dir)

    training_images = get_image_info('data/ade20k/ADE20K_2021_17_01/images/ADE', 'training')
    validation_images = get_image_info('data/ade20k/ADE20K_2021_17_01/images/ADE', 'validation')



    # Get the custom dataset and dataloader
    print(f"Getting data loaders")
    transform_list = []
    transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())

    if mean is not None and std is not None:
        transform_list.append(transforms.Normalize(mean, std))

    transform = transforms.Compose(transform_list)


    traindataset = ADE20K_Objects(training_images, other_images=validation_images, transform=transform, zoom_fraction=args.zoom/100)
    valdataset = ADE20K_Objects(validation_images, other_images=training_images, transform=transform, class_to_index=traindataset.class_to_index, zoom_fraction=args.zoom/100)

    print(f"Number of classes: {len(traindataset.class_to_index)}")


    train_size = int(len(traindataset) * 0.8)
    test_size = len(traindataset) - train_size
    traindataset, testdataset  = random_split(traindataset, [train_size, test_size])


    # Create loaders - shuffle training set, but not validation or test set
    trainloader = DataLoader(traindataset, collate_fn=collate_fn,
                                batch_size=batch_size, shuffle=True, num_workers=int(args.n_processes))
    valloader = DataLoader(valdataset, collate_fn=collate_fn,
                            batch_size=batch_size, shuffle=False, num_workers=int(args.n_processes))
    testloader = DataLoader(testdataset, collate_fn=collate_fn,
                            batch_size=batch_size, shuffle=False, num_workers=int(args.n_processes))

    print(f"Loaded data loaders")

    print(f"Training set size: {len(traindataset)}")
    print(f"Validation set size: {len(valdataset)}")
    print(f"Test set size: {len(testdataset)}")



    output_channels = 3542
    if args.model_type == 'resnet50':
        model = resnet50()
        model.conv1 = torch.nn.Conv2d(in_channels=n_input_channels, out_channels=model.conv1.out_channels, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        model.fc = torch.nn.Linear(
            in_features=2048, out_features=output_channels, bias=True)
    else:
        raise ValueError(f"Model {args.model_type} not recognized")

    print(f"Created model {args.model_type}")


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

    
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not recognized")

    # Learning Rate Scheduler
    # plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 'min', patience=5)
    plateau_scheduler = None
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    info = {
        # 'training_indices': train_ids,
        'train_size': len(traindataset),
        'test_size': len(testdataset),
        'val_size': len(valdataset),
        # 'testing_indices': test_ids,
        # 'validation_indices': val_ids,
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

    train_objects(model=model, trainloader=trainloader, valloader=valloader, device=device, results={},
            loss_fn=criterion, optimizer=optimizer, n_epochs=n_epochs, result_manager=result_manager,
            testloader=testloader, plateau_lr_scheduler=plateau_scheduler, lr_scheduler=lr_scheduler, current_time=c_time, save_per_epoch=bool(args.save_per_epoch))
