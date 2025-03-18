import os
import sys
home_path = os.path.expanduser('~')
import argparse
from datetime import datetime as datetime_function

import torchvision
from torchvision import transforms
from torchvision.models import resnet18, resnet50, alexnet, vgg16
from torchvision.datasets import ImageFolder
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torch
from torch import nn, optim
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split, Subset
from torch.nn.parallel import DistributedDataParallel as DDP


from pytorch_utils.fcn_resnet50_coco import own_fcn_resnet50

from base.training import train_objects, collate_fn
from base.datasets import ImageNetCueConflict
from base.result_manager import ResultManager
from base.models import get_model_instance_segmentation, ADE20K_ImageNetModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='Directory of dataset', default='/projects/2/managed_datasets/Imagenet')
    parser.add_argument(
        '--model_type', help='Model to use for training.', default='resnet50')
    parser.add_argument('--image_representation',
                        help='Way images are represented. Can be `rgb`, `coc` (color opponent channels) or `all` for training all 4 conditions', default='rgb')
    parser.add_argument(
        '--n_epochs', help='Number of epochs for training.', default=15)
    # parser.add_argument(
    #     '--user_name', help='systems username', default='niklas')
    parser.add_argument(
        '-use_jpeg', help='Whether to use JPEG Compression or not', action='store_true')
    parser.add_argument(
        '--finetune_full', help='Whether to finetune the whole network. Default is only finetuning the last layer (classifier).', default='layer4')
    parser.add_argument(
        '-no_save_per_epoch', help='Whether to only save best and last model or per epoch', action='store_false')
    parser.add_argument(
        '--batch_size', help='Batch size for training.', default=256)
    parser.add_argument(
        '--image_size', help='Batch size for training.', default=400)
    parser.add_argument(
        '--n_processes', help='Number of processes to use.', default=18)
    parser.add_argument(
        '--limit_classes', help='Whether to use a limited number of classes. If False, all 1000 classes will be used. If true, 18 deterministic classes will be used. If n, n random classes will be used.', default=False)

    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')


    # TODO fix this whole part about image_representation, use_jpeg and everything else that is outdated
    # TODO fix how to run this for the different finetuning stages

    model_type = args.model_type # 'resnet50'
    image_representation = args.image_representation # 'rgb'
    use_jpeg = args.use_jpeg # False
    finetune_full = (True if args.finetune_full == 'True' else False) if args.finetune_full != 'layer4' else 'layer4'
    image_size = None if args.image_size == 'None' else int(args.image_size)

    limit_classes = args.limit_classes
    texture_shape = False
    if limit_classes == 'True':
        use_classes = ['n01728572', 'n01882714', 'n02087046', 'n02100583', 'n02113186', 'n02259212', 'n02484975', 'n02791124', 'n02966193', 'n03188531', 'n03461385', 'n03706229', 'n03857828', 'n03998194', 'n04209239', 'n04389033', 'n04562935', 'n07753275']
    elif limit_classes == 'False':
        use_classes = None
    elif 'texture' in limit_classes or 'shape' in limit_classes or 'cue' in limit_classes:
        use_classes = 1000
        texture_shape = True
    else:
        use_classes = int(limit_classes)

    initial_output_channels = 21 # 21
    # output_channels = 1000 # imagenet-1k classes
    output_channels = (use_classes if type(use_classes) is int else len(use_classes)) if use_classes is not None else 1000
    gpu_name = 'cuda:0'
    device = torch.device(gpu_name if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')
    torch.cuda.empty_cache()
    batch_size = int(args.batch_size) # 512
    num_workers = int(args.n_processes) # 12

    
    ###################################### READ MODEL OVERVIEW ######################################
    yml_path = '/home/nmuller/projects/robustpy/model_overview_COPY_FROM_TUX20.yaml'
    import yaml

    with open(yml_path, 'r') as file:
        model_overview = yaml.load(file, Loader=yaml.FullLoader)

    pretraining_dataset = 'coco_objects' if 'coco_objects' in model_type else ('ade20k_objects' if 'ade20k_objects' in model_type else 'oads')
    zoom = image_representation.split('_')[-1] if 'zoom' in image_representation else 'rgb'
    # reps = 1
    finetuning = 'fc' if finetune_full == False else ('layer4' if finetune_full == 'layer4' else 'full')

    reps = 3
    while True:
        model_path = model_overview[pretraining_dataset][zoom]['reps'][reps]['path'].replace('training_results', 'best_model').replace('.yml', '.pth')
        if len(model_overview[pretraining_dataset][zoom]['reps'][reps]['finetuned']['imagenet'][finetuning]) > 0:
            reps += 1
        else:
            break

    # print(reps)
    # exit(1)

    if not os.path.exists(model_path):
        print(f'Model path does not exist: {model_path}')
        model_path = model_path.replace('best_model', 'final_model')
        if not os.path.exists(model_path):
            print(f'Model path also does not exist: {model_path}')
            exit(1)

    dt = 'reps/' + model_path.split('reps/')[-1].split('_')[0]
    print(f'Using reps {reps} with pretrained model: {model_path}')
    # exit(1)
    #################################################################################################


    if image_representation == 'all':
        image_representations = ['rgb_zoom-80'] # 'rgb_zoom-80', 'rgb_zoom-100', 'rgb_zoom-150'
        use_jpegs = [False] # False
        finetune_fulls = ['layer4'] #  # False, True, 'layer4'
        model_types = ['resnet50_coco_objects'] #'resnet18', 'flex_resnet50', 'fasterrcnn_resnet50_fpn_coco', 'vit_b_16'
        image_sizes = [400] # None, 400
    else:
        image_representations = [image_representation]
        use_jpegs = [use_jpeg]
        finetune_fulls = [finetune_full]
        model_types = [model_type]
        image_sizes = [image_size]

    print(f'Will compute {image_representations} and jpeg: {use_jpegs} for finetune_full: {finetune_fulls}')

    for finetune_full in finetune_fulls:
        for image_representation in image_representations:
            for use_jpeg in use_jpegs:
                for model_type in model_types:
                    for image_size in image_sizes:

                        print(f'Running: {image_representation} with jpeg: {use_jpeg} and finetune_full: {finetune_full}')

                        if type(finetune_full) is bool:
                            finetuned_dir_name = 'finetuned_full' if finetune_full else 'finetuned'
                        elif  type(finetune_full) is str:
                            finetuned_dir_name = finetune_full

                        if texture_shape:
                            finetuned_dir_name = f'{finetuned_dir_name}_subclasses'

                        if 'places365' not in model_type and ('coco' not in model_type or 'objects' in model_type) and ('ade20k' not in model_type or 'objects' in model_type):

                            state_dict = torch.load(model_path, map_location=gpu_name)
                            # if 'module.fc.weight' in state_dict:
                            #     initial_output_channels = state_dict['module.fc.weight'].shape[0]
                            # elif 'fc.weight' in state_dict:
                            #     initial_output_channels = state_dict['fc.weight'].shape[0]
                            # else:
                            if 'coco' in model_type:
                                initial_output_channels = 80
                            elif 'ade20k' in model_type:
                                initial_output_channels = 3542
                            else:
                                initial_output_channels = 21

                        else:
                            dt = datetime_function.now().strftime("%d-%m-%y-%H%M%S")
                            initial_output_channels = None

                        result_manager = ResultManager(root=os.path.join(f'{home_path}/projects/prjs0391/robustpy_finetuned_normalized', finetuned_dir_name, model_type, image_representation, "jpeg" if use_jpeg else "", dt))

                        if model_type == 'resnet50' or model_type == 'resnet50_coco_objects' or model_type == 'resnet50_ade20k_objects':
                            model = resnet50()
                            model.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=model.conv1.out_channels, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
                            model.fc = torch.nn.Linear(in_features=2048, out_features=initial_output_channels, bias=True)
                            try:
                                model.load_state_dict(torch.load(model_path, map_location=gpu_name))
                            except RuntimeError:
                                model = torch.nn.DataParallel(model)
                                model.load_state_dict(torch.load(model_path, map_location=gpu_name))
                                model = model.module
                            if not finetune_full:
                                for child in model.children():
                                    for param in child.parameters():
                                        param.requires_grad = False
                            elif type(finetune_full) is str and finetune_full == 'layer4':
                                for child in model.children():
                                    for param in child.parameters():
                                        param.requires_grad = False

                                for child in model.layer4.children():
                                    for param in child.parameters():
                                        param.requires_grad = True

                            model.fc = torch.nn.Linear(in_features=2048, out_features=output_channels, bias=True)

                        elif model_type == 'resnet50_places365':
                            model = places365_models.resnet50()
                            # model.fc = torch.nn.Linear(
                            #     in_features=2048, out_features=output_channels, bias=True)
                            
                            if not finetune_full:
                                for child in model.children():
                                    for param in child.parameters():
                                        param.requires_grad = False
                            elif type(finetune_full) is str and finetune_full == 'layer4':
                                for child in model.children():
                                    for param in child.parameters():
                                        param.requires_grad = False

                                for child in model.layer4.children():
                                    for param in child.parameters():
                                        param.requires_grad = True

                            model.fc = torch.nn.Linear(in_features=2048, out_features=output_channels, bias=True)

                        elif model_type == 'alexnet' or 'alexnet_ade20k_objects' in model_type or 'alexnet_coco_objects' in model_type:
                            model = alexnet()
                            model.features[0] = torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
                            model.classifier[6] = torch.nn.Linear(4096, initial_output_channels, bias=True)

                            try:
                                model.load_state_dict(torch.load(model_path, map_location=gpu_name))
                            except RuntimeError:
                                model = torch.nn.DataParallel(model)
                                model.load_state_dict(torch.load(model_path, map_location=gpu_name))
                                model = model.module

                            if not finetune_full:
                                for child in model.children():
                                    for param in child.parameters():
                                        param.requires_grad = False
                            elif type(finetune_full) is str and finetune_full == 'layer4':
                                for child in model.children():
                                    for param in child.parameters():
                                        param.requires_grad = False

                                # for child in model.layer4.children():
                                for child in [model.features[10], model.features[11], model.features[12]]:
                                    for param in child.parameters():
                                        param.requires_grad = True

                            model.classifier[6] = torch.nn.Linear(4096, output_channels, bias=True)
                            for child in model.classifier.children():
                                for param in child.parameters():
                                    param.requires_grad = True
                                    
                        elif 'vit_b_16_ade20k_scenes' in model_type:
                            # https://huggingface.co/Akide/SegViTv1/resolve/main/ade_51.3.pth?download=true
                            # https://arxiv.org/pdf/2210.05844
                            # https://github.com/zbwxp/SegVit?tab=readme-ov-file
                            pass
                        elif model_type == 'vit_b_16' or 'vit_b_16_ade20k_objects' in model_type or 'vit_b_16_coco_objects' in model_type:
                            model = torch.hub.load("facebookresearch/swag", model="vit_b16")
                            model.head = torch.nn.Linear(in_features=768, out_features=initial_output_channels, bias=True)

                            if 'ade20k_objects' in model_type or 'coco_objects' in model_type:
                                try:
                                    model.load_state_dict(torch.load(model_path, map_location=gpu_name))
                                except (RuntimeError, KeyError):
                                    model = torch.nn.DataParallel(model)
                                    model.load_state_dict(torch.load(model_path, map_location=gpu_name))
                                    model = model.module

                            if not finetune_full:
                                for child in model.children():
                                    for param in child.parameters():
                                        param.requires_grad = False
                            elif type(finetune_full) is str and finetune_full == 'layer4':
                                for layer_index, layer in enumerate(model.encoder.layers):
                                    if layer_index < 9:
                                        for param in layer.parameters():
                                            param.requires_grad = False
                                    else:
                                        for param in layer.parameters():
                                            param.requires_grad = True

                            # for children in model.head.children():
                            #     for param in children.parameters():
                            #         param.requires_grad = True

                            model.head = torch.nn.Linear(in_features=768, out_features=output_channels, bias=True)

                        elif 'fcn_resnet50_coco' in model_type:
                            # model = fcn_resnet50(pretrained=True)
                            model = own_fcn_resnet50(pretrained=True, num_classes=output_channels)

                            if not finetune_full:
                                for child in model.children():
                                    for param in child.parameters():
                                        param.requires_grad = False
                            elif type(finetune_full) is str and finetune_full == 'layer4':
                                for child in model.children():
                                    for param in child.parameters():
                                        param.requires_grad = False

                                for child in model.backbone.layer4.children():
                                    for param in child.parameters():
                                        param.requires_grad = True

                            for child in model.classifier.children():
                                for param in child.parameters():
                                    param.requires_grad = True

                        elif 'resnet50_ade20k_scenes' in model_type:
                            model = ADE20K_ImageNetModel(1000)

                            if not finetune_full:
                                for child in model.children():
                                    for param in child.parameters():
                                        param.requires_grad = False
                            elif type(finetune_full) is str and finetune_full == 'layer4':
                                for child in model.children():
                                    for param in child.parameters():
                                        param.requires_grad = False

                                for child in model.model.encoder.stages[3].children():
                                    for param in child.parameters():
                                        param.requires_grad = True

                            for child in model.head.children():
                                for param in child.parameters():
                                    param.requires_grad = True

                        elif 'fasterrcnn_resnet50_fpn_coco' in model_type:
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

                            
                        model = model.to(device)

                        #Imagenet-1k
                        mean_image = [0.485, 0.456, 0.406]
                        std_image = [0.229, 0.224, 0.225]

                        transform_list = []
                        transform_list.append(transforms.Resize(256))
                        transform_list.append(transforms.CenterCrop(224))
                        transform_list.append(transforms.ToTensor())
                        transform_list.append(transforms.Normalize(mean_image, std_image))
                        transform = transforms.Compose(transform_list)

                        basedir = args.input_dir

                        imagenet = ImageNetCueConflict(args.input_dir, split='train', transform=transform, root_extension="", return_index=True, val_label_filepath='/home/mullern/projects/data/imagenet-1k/')
                        train_size = int(len(imagenet) * 0.9)
                        test_size = len(imagenet) - train_size
                        train_dataset, test_dataset  = random_split(imagenet, [train_size, test_size])
                        
                        val_dataset = ImageNetCueConflict(args.input_dir, split='val', transform=transform, root_extension="", return_index=True, val_label_filepath='/home/mullern/projects/data/imagenet-1k/')


                        trainloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=int(args.n_processes), pin_memory=True, prefetch_factor=8)
                        testloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False, num_workers=int(args.n_processes), pin_memory=True, prefetch_factor=8)
                        valloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False, num_workers=int(args.n_processes), pin_memory=True, prefetch_factor=8)

                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.Adam(model.parameters(), lr=0.001)
                        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, 'min', patience=5)
                        

                        results = train_objects(model=model, result_manager=result_manager, trainloader=trainloader, testloader=testloader, valloader=valloader,
                                        device=device, n_epochs=int(args.n_epochs), optimizer=optimizer, loss_fn=criterion, plateau_lr_scheduler=plateau_scheduler, save_per_epoch=args.no_save_per_epoch)

                        print(f'Done finetuning {image_representation} with jpeg: {use_jpeg}')