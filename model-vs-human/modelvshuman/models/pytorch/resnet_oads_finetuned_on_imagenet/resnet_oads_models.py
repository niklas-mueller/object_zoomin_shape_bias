import torch
from torchvision.models import resnet18, resnet50, ResNet, alexnet
from torch.nn.parallel import DataParallel
from . import network_Gray_ResNet
from .fcn_resnet50_coco import own_fcn_resnet50
import os
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torchvision
from torch import nn
from transformers import MaskFormerForInstanceSegmentation

######### Scaling on Scales
from s2wrapper import forward as multiscale_forward
#########

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def fasterrcnn_resnet50_fpn_coco(num_classes):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=0, world_size=1)
    
    # num_classes = 100
    # num_epochs = 100
    model = get_model_instance_segmentation(100)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # torch.cuda.set_device(device)
    model.cuda(device)
    model = DDP(model, device_ids=[0])
    faster_ccn_model_path = '/home/nmuller/projects/fmg_storage/trained_models/coco_results/27-02-24-173725/best_model_Tue_Feb_27_17:38:03_2024.pth'
    model.load_state_dict(torch.load(faster_ccn_model_path, map_location='cuda:0'))
    model = model.module

    dist.destroy_process_group()

    model = model.backbone

    class DictToTensor(nn.Module):
        def __init__(self):
            super(DictToTensor, self).__init__()
        def forward(self, x):
            return x['3']

    model.fpn = torch.nn.Sequential(
        DictToTensor(),
        torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        torch.nn.Flatten(1),
        torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    )

    return model

home_path = os.path.expanduser('~')

__all__ = [
    'resnet18_oads_normalized_rgb_finetuned_layer4_on_imagenet',
    'resnet18_oads_normalized_rgb_jpeg_finetuned_layer4_imagenetsize_on_imagenet',
    'resnet18_oads_rgb_jpeg_finetuned_layer4_imagenetsize_on_imagenet',
    'resnet18_oads_rgb_finetuned_on_imagenet',
    'resnet18_oads_coc_finetuned_on_imagenet',
    'resnet18_oads_rgb_jpeg_finetuned_on_imagenet',
    'resnet18_oads_coc_jpeg_finetuned_on_imagenet',
    'resnet18_oads_rgb_finetuned_full_on_imagenet',
    'resnet18_oads_coc_finetuned_full_on_imagenet',
    'resnet18_oads_rgb_jpeg_finetuned_full_on_imagenet',
    'resnet18_oads_coc_jpeg_finetuned_full_on_imagenet',
    'resnet18_oads_rgb_finetuned_layer4_on_imagenet',
    'resnet18_oads_coc_finetuned_layer4_on_imagenet',
    'resnet18_oads_rgb_jpeg_finetuned_layer4_on_imagenet',
    'resnet18_oads_coc_jpeg_finetuned_layer4_on_imagenet',
    'resnet50_oads_rgb_finetuned_on_imagenet',
    'resnet50_oads_coc_finetuned_on_imagenet',
    'resnet50_oads_rgb_jpeg_finetuned_on_imagenet',
    'resnet50_oads_coc_jpeg_finetuned_on_imagenet',
    'resnet50_oads_rgb_finetuned_full_on_imagenet',
    'resnet50_oads_coc_finetuned_full_on_imagenet',
    'resnet50_oads_rgb_jpeg_finetuned_full_on_imagenet',
    'resnet50_oads_coc_jpeg_finetuned_full_on_imagenet',
    'resnet50_oads_rgb_finetuned_layer4_on_imagenet',
    'resnet50_oads_coc_finetuned_layer4_on_imagenet',
    'resnet50_oads_rgb_jpeg_finetuned_layer4_on_imagenet',
    'resnet50_oads_coc_jpeg_finetuned_layer4_on_imagenet',
    'resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_imagenetsize_on_imagenet',
    'resnet50_oads_normalized_rgb_jpeg_40_finetuned_layer4_imagenetsize_on_imagenet',
    'resnet50_oads_normalized_rgb_jpeg_60_finetuned_layer4_imagenetsize_on_imagenet',
    'resnet50_oads_normalized_rgb_jpeg_90_finetuned_layer4_imagenetsize_on_imagenet',
    'resnet50_oads_normalized_rgb_finetuned_layer4_on_imagenet',
    'resnet50_oads_normalized_rgb_jpeg_40_finetuned_full_imagenetsize_on_imagenet',
    'resnet50_oads_normalized_rgb_jpeg_60_finetuned_full_imagenetsize_on_imagenet',
    'resnet50_oads_normalized_rgb_jpeg_90_finetuned_full_imagenetsize_on_imagenet',
    'resnet50_oads_normalized_rgb_finetuned_full_on_imagenet',

    'resnet50_oads_normalized_rgb_finetuned_imagenetsize_on_imagenet',
    'resnet50_oads_normalized_rgb_jpeg_finetuned_imagenetsize_on_imagenet',

    'resnet50_oads_normalized_rgb_jpeg_finetuned_on_imagenet',
    'resnet50_oads_normalized_rgb_jpeg_finetuned_full_on_imagenet',
    'resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_on_imagenet',
    'resnet50_oads_normalized_rgb_finetuned_on_subimagenet',
    'resnet50_oads_normalized_rgb_jpeg_finetuned_on_subimagenet',
    'resnet50_oads_normalized_rgb_jpeg_finetuned_imagenetsize_on_subimagenet',
    'resnet50_oads_normalized_rgb_finetuned_imagenetsize_on_subimagenet',
    'resnet50_oads_normalized_rgb_finetuned_full_on_subimagenet',
    'resnet50_oads_normalized_rgb_jpeg_finetuned_full_on_subimagenet',
    'resnet50_oads_normalized_rgb_jpeg_finetuned_full_imagenetsize_on_subimagenet',
    'resnet50_oads_normalized_rgb_finetuned_full_imagenetsize_on_subimagenet',
    'resnet50_oads_normalized_rgb_finetuned_layer4_on_subimagenet',
    'resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_on_subimagenet',
    'resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_imagenetsize_on_subimagenet',
    'resnet50_oads_normalized_rgb_finetuned_layer4_imagenetsize_on_subimagenet',

    'resnet50_imagenet_subclasses',
    'resnet50_imagenet_grayscale',

    'resnet50_imagenet_112x112',
    'resnet50_imagenet_400x400',
    'resnet50_imagenet_500x500',
    'resnet50_imagenet_600x600',
    'resnet50_imagenet_10x10_to_224x224',
    'resnet50_imagenet_30x30_to_224x224',
    'resnet50_imagenet_80x80_to_224x224',
    'resnet50_imagenet_112x112_to_224x224',

    'resnet50_s2_imagenet',

    'vit_b_16_oads_finetuned_full_on_subimagenet',
    'vit_b_16_oads_finetuned_on_subimagenet',
    'vit_b_16_oads_finetuned_layer4_on_subimagenet',

    'vit_b_16_imagenet',
    'vit_b_16_subimagenet',

    'resnet50_imagenet_400x400_low_res',
    'resnet50_imagenet_no_crop_224x224',
    'resnet50_imagenet_no_crop_400x400',
    'resnet50_imagenet_600x600_crop_400x400',
    'resnet50_imagenet_350x350_crop_224x224',

    'resnet50_places365_finetuned_on_subimagenet',
    'resnet50_places365_finetuned_full_on_subimagenet',
    'resnet50_places365_finetuned_layer4_on_subimagenet',

    'fcn_resnet50_coco_finetuned_on_subimagenet',
    'fcn_resnet50_coco_finetuned_full_on_subimagenet',
    'fcn_resnet50_coco_finetuned_layer4_on_subimagenet',

    'fcn_resnet50_coco_oads_finetuned_on_subimagenet',
    'fcn_resnet50_coco_oads_finetuned_full_on_subimagenet',
    'fcn_resnet50_coco_oads_finetuned_layer4_on_subimagenet',


    'resnet50_coco_objects_finetuned_on_subimagenet',
    'resnet50_coco_objects_finetuned_full_on_subimagenet',
    'resnet50_coco_objects_finetuned_layer4_on_subimagenet',

    'resnet50_subimagenet_bounding_boxes',

    'resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p70',
    'resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100',

    'resnet50_subimagenet_bounding_boxes_p70',
    'resnet50_subimagenet_bounding_boxes_p100',
    'resnet50_subimagenet_bounding_boxes_p100_zoom_30',
    'resnet50_subimagenet_bounding_boxes_p100_zoom_50',
    'resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_100',
    'resnet50_subimagenet_bounding_boxes_p100_zoom_100',

    'resnet50_oads_zoom_80_finetuned_on_subimagenet',
    'resnet50_oads_zoom_100_finetuned_on_subimagenet',
    'resnet50_oads_zoom_150_finetuned_on_subimagenet',
    'resnet50_all_boxes_imagenet',


    'resnet50_ade20k_scenes_finetuned_on_subimagenet',
    'resnet50_ade20k_scenes_finetuned_layer4_on_subimagenet',
    'resnet50_ade20k_scenes_finetuned_full_on_subimagenet',

    'resnet50_ade20k_objects_zoom_0_finetuned_on_subimagenet',
    'resnet50_ade20k_objects_zoom_0_finetuned_layer4_on_subimagenet',
    'resnet50_ade20k_objects_zoom_0_finetuned_full_on_subimagenet',
    'resnet50_ade20k_objects_zoom_80_finetuned_on_subimagenet',
    'resnet50_ade20k_objects_zoom_80_finetuned_layer4_on_subimagenet',
    'resnet50_ade20k_objects_zoom_80_finetuned_full_on_subimagenet',
    'resnet50_ade20k_objects_zoom_150_finetuned_on_subimagenet',
    'resnet50_ade20k_objects_zoom_150_finetuned_layer4_on_subimagenet',
    'resnet50_ade20k_objects_zoom_150_finetuned_full_on_subimagenet',

    'resnet50_coco_objects_zoom_80_finetuned_on_subimagenet',
    'resnet50_coco_objects_zoom_80_finetuned_full_on_subimagenet',
    'resnet50_coco_objects_zoom_80_finetuned_layer4_on_subimagenet',
    'resnet50_coco_objects_zoom_150_finetuned_on_subimagenet',
    'resnet50_coco_objects_zoom_150_finetuned_full_on_subimagenet',
    'resnet50_coco_objects_zoom_150_finetuned_layer4_on_subimagenet',

    'resnet50_subimagenet_bounding_boxes_p100_zoom_150',
    'resnet50_subimagenet_bounding_boxes_p100_zoom_80',

    'vit_b_16_subimagenet_bounding_boxes_p100_zoom_150',
    'vit_b_16_subimagenet_bounding_boxes_p100_zoom_80',
    'vit_b_16_subimagenet_bounding_boxes_p100_zoom_0',
    'alexnet_subimagenet_bounding_boxes_p100_zoom_150',
    'alexnet_subimagenet_bounding_boxes_p100_zoom_80',
    'alexnet_subimagenet_bounding_boxes_p100_zoom_0',

    'resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_0',
    'resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_80',
    'resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_150',
    'alexnet_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_0',
    'alexnet_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_80',
    'alexnet_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_150',
    'vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_0',
    'vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_80',
    'vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_150',

    'resnet50_imagenet_finetuned_on_all_boxes_p50_zoom_0',

    'vit_b_16_coco_objects_zoom_0_finetuned_on_subimagenet',
    'vit_b_16_coco_objects_zoom_0_finetuned_full_on_subimagenet',
    'vit_b_16_coco_objects_zoom_0_finetuned_layer4_on_subimagenet',
    'vit_b_16_coco_objects_zoom_80_finetuned_on_subimagenet',
    'vit_b_16_coco_objects_zoom_80_finetuned_full_on_subimagenet',
    'vit_b_16_coco_objects_zoom_80_finetuned_layer4_on_subimagenet',
    'vit_b_16_coco_objects_zoom_150_finetuned_on_subimagenet',
    'vit_b_16_coco_objects_zoom_150_finetuned_full_on_subimagenet',
    'vit_b_16_coco_objects_zoom_150_finetuned_layer4_on_subimagenet',
    'alexnet_coco_objects_zoom_0_finetuned_on_subimagenet',
    'alexnet_coco_objects_zoom_0_finetuned_full_on_subimagenet',
    'alexnet_coco_objects_zoom_0_finetuned_layer4_on_subimagenet',
    'alexnet_coco_objects_zoom_80_finetuned_on_subimagenet',
    'alexnet_coco_objects_zoom_80_finetuned_full_on_subimagenet',
    'alexnet_coco_objects_zoom_80_finetuned_layer4_on_subimagenet',
    'alexnet_coco_objects_zoom_150_finetuned_on_subimagenet',
    'alexnet_coco_objects_zoom_150_finetuned_full_on_subimagenet',
    'alexnet_coco_objects_zoom_150_finetuned_layer4_on_subimagenet',
    'vit_b_16_ade20k_objects_zoom_0_finetuned_on_subimagenet',
    'vit_b_16_ade20k_objects_zoom_0_finetuned_full_on_subimagenet',
    'vit_b_16_ade20k_objects_zoom_0_finetuned_layer4_on_subimagenet',
    'vit_b_16_ade20k_objects_zoom_80_finetuned_on_subimagenet',
    'vit_b_16_ade20k_objects_zoom_80_finetuned_full_on_subimagenet',
    'vit_b_16_ade20k_objects_zoom_80_finetuned_layer4_on_subimagenet',
    'vit_b_16_ade20k_objects_zoom_150_finetuned_on_subimagenet',
    'vit_b_16_ade20k_objects_zoom_150_finetuned_full_on_subimagenet',
    'vit_b_16_ade20k_objects_zoom_150_finetuned_layer4_on_subimagenet',
    'alexnet_ade20k_objects_zoom_0_finetuned_on_subimagenet',
    'alexnet_ade20k_objects_zoom_0_finetuned_full_on_subimagenet',
    'alexnet_ade20k_objects_zoom_0_finetuned_layer4_on_subimagenet',
    'alexnet_ade20k_objects_zoom_80_finetuned_on_subimagenet',
    'alexnet_ade20k_objects_zoom_80_finetuned_full_on_subimagenet',
    'alexnet_ade20k_objects_zoom_80_finetuned_layer4_on_subimagenet',
    'alexnet_ade20k_objects_zoom_150_finetuned_on_subimagenet',
    'alexnet_ade20k_objects_zoom_150_finetuned_full_on_subimagenet',
    'alexnet_ade20k_objects_zoom_150_finetuned_layer4_on_subimagenet',

    'transformer_b16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_150',

    'resnet50_oads_zoom_150_finetuned_full_on_subimagenet',
    'resnet50_oads_zoom_80_finetuned_full_on_subimagenet',
    ]

for epoch in range(90):
    __all__.append(f'resnet50_julio_epoch_{epoch}')
    __all__.append(f'alexnet_julio_epoch_{epoch}')

model_paths = {
    'resnet50_oads_zoom_150_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/resnet50/rgb_zoom-150/reps/2024-05-01-211852_400x400/best_model_Mon_May_20_19:18:04_2024.pth',
    'resnet50_oads_zoom_80_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/resnet50/rgb_zoom-80/reps/2024-05-01-211514_400x400/best_model_Mon_May_20_19:49:19_2024.pth',

    # 'resnet50_subimagenet_bounding_boxes': '/home/nmuller/projects/fmg_storage/trained_models/imagenet_results/resnet50/25-04-24-091923_bounding_boxes/best_model_25-04-24-091923.pth',

    'resnet50_s2_imagenet': '/home/nmuller/projects/fmg_storage/trained_models/imagenet_results/resnet50_s2/16-04-24-140621/best_model_16-04-24-140621.pth',
    'resnet50_imagenet_subclasses': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/21-07-23-104903/best_model_21-07-23-104903.pth',
    'resnet50_imagenet_grayscale': '/home/nmuller/projects/fmg_storage/trained_models/imagenet_results/ResNet50_Gray_epoch60_BN_batchsize64_state_dict.pth',
    'resnet50_imagenet_112x112': '/home/nmuller/projects/fmg_storage/trained_models/imagenet_results/08-09-23-145142_112x112/best_model_08-09-23-145142.pth',
    'resnet50_imagenet_400x400': '/home/nmuller/projects/fmg_storage/trained_models/imagenet_results/29-08-23-154050_400x400/best_model_29-08-23-154050.pth',
    'resnet50_imagenet_500x500': '/home/nmuller/projects/fmg_storage/trained_models/imagenet_results/05-09-23-152434_500x500/best_model_05-09-23-152434.pth',
    'resnet50_imagenet_600x600': '/home/nmuller/projects/fmg_storage/trained_models/imagenet_results/05-09-23-154005_600x600/best_model_05-09-23-154005.pth',
    
    'resnet50_imagenet_10x10_to_224x224': '/home/nmuller/projects/fmg_storage/trained_models/imagenet_results/20-10-23-155519_pre_resize_10x10_then_224x224/best_model_20-10-23-155519.pth',
    'resnet50_imagenet_30x30_to_224x224': '/home/nmuller/projects/fmg_storage/trained_models/imagenet_results/20-10-23-155511_pre_resize_30x30_then_224x224/best_model_20-10-23-155511.pth',
    'resnet50_imagenet_80x80_to_224x224': '/home/nmuller/projects/fmg_storage/trained_models/imagenet_results/26-10-23-102423_pre_resize_80x80_then_224x224/best_model_26-10-23-102423.pth',

    'resnet50_imagenet_112x112_to_224x224': '/home/nmuller/projects/fmg_storage/trained_models/imagenet_results/11-09-23-170142_112x112_to_224x224/final_model_11-09-23-172037.pth',

    'resnet50_imagenet_no_crop_224x224': '/home/nmuller/projects/fmg_storage/trained_models/imagenet_results/09-10-23-160442_no_crop_224x224/best_model_09-10-23-160442.pth',
    'resnet50_imagenet_no_crop_400x400': '/home/nmuller/projects/fmg_storage/trained_models/imagenet_results/11-10-23-160119_no_crop_400x400/best_model_11-10-23-160119.pth',

    'resnet50_imagenet_600x600_crop_400x400': '/home/nmuller/projects/fmg_storage/trained_models/imagenet_results/09-10-23-161935_600x600_crop_400x400/best_model_09-10-23-161935.pth',

    'resnet50_imagenet_350x350_crop_224x224': '/home/nmuller/projects/fmg_storage/trained_models/imagenet_results/11-10-23-164632_350x350_crop_224x224/best_model_11-10-23-164632.pth',

    # 'resnet50_oads_rgb_finetuned_imagenetsize_on_imagenet': '',
    'resnet50_oads_normalized_rgb_jpeg_finetuned_imagenetsize_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned/resnet50/rgb/jpeg/2023-03-23-172559_imagenetsize/best_model_Fri_Jul_28_132020_2023.pth',
    # 'resnet50_oads_normalized_rgb_jpeg_finetuned_imagenetsize_on_sub_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned/resnet50/rgb/jpeg/2023-03-23-17:25:59_imagenetsize/best_model_Fri_Jul_28_16:16:28_2023.pth',

    'resnet18_oads_normalized_rgb_finetuned_layer4_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4/resnet18/rgb/2023-03-23-135528/best_model_Fri_Mar_24_092339_2023.pth',
    'resnet18_oads_normalized_rgb_jpeg_finetuned_layer4_imagenetsize_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4/resnet18/rgb/jpeg/2023-03-23-140446/best_model_Fri_Mar_24_092339_2023.pth',
    # 'resnet18_oads_normalized_coc_finetuned_layer4_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4/resnet50/rgb/jpeg/2023-03-23-17:25:59_imagenetsize/best_model_Fri_Mar_24_09:27:02_2023.pth',
    # 'resnet18_oads_normalized_coc_jpeg_finetuned_layer4_imagenetsize_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4/resnet50/rgb/jpeg/2023-03-23-17:25:59_imagenetsize/best_model_Fri_Mar_24_09:27:02_2023.pth',
    
    ################
    # 'resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_imagenetsize_on_imagenet': '',
    'resnet50_oads_normalized_rgb_jpeg_40_finetuned_layer4_imagenetsize_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4/resnet50/rgb/jpeg_40/2023-03-24-170238/best_model_Sun_Mar_26_122106_2023.pth',
    'resnet50_oads_normalized_rgb_jpeg_60_finetuned_layer4_imagenetsize_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4/resnet50/rgb/jpeg_60/2023-03-24-170238/best_model_Sun_Mar_26_122106_2023.pth',
    'resnet50_oads_normalized_rgb_jpeg_90_finetuned_layer4_imagenetsize_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4/resnet50/rgb/jpeg_90/2023-03-24-170708/best_model_Sun_Mar_26_122106_2023.pth',
    
    # 'resnet50_oads_normalized_rgb_jpeg_finetuned_full_imagenetsize_on_imagenet': '',
    'resnet50_oads_normalized_rgb_jpeg_40_finetuned_full_imagenetsize_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full/resnet50/rgb/jpeg_40/2023-03-24-170238/best_model_Sun_Mar_26_122420_2023.pth',
    'resnet50_oads_normalized_rgb_jpeg_60_finetuned_full_imagenetsize_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full/resnet50/rgb/jpeg_60/2023-03-24-170238/best_model_Sun_Mar_26_122420_2023.pth',
    'resnet50_oads_normalized_rgb_jpeg_90_finetuned_full_imagenetsize_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full/resnet50/rgb/jpeg_90/2023-03-24-170708/best_model_Wed_Mar_29_190532_2023.pth',

    'resnet50_oads_normalized_rgb_finetuned_layer4_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4/resnet50/rgb/2023-03-24-170946/best_model_Thu_Mar_30_124755_2023.pth',
    'resnet50_oads_normalized_rgb_finetuned_full_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full/resnet50/rgb/2023-03-24-170946/best_model_Wed_Mar_29_190447_2023.pth',
    'resnet50_oads_normalized_rgb_finetuned_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned/resnet50/rgb/2023-07-24-132028/best_model_Tue_Aug_15_172912_2023.pth',
    
    'resnet50_oads_normalized_rgb_finetuned_full_imagenetsize_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full/resnet50/rgb/2023-03-27-170131_imagenetsize/best_model_Wed_Mar_29_190447_2023.pth',
    'resnet50_oads_normalized_rgb_finetuned_layer4_imagenetsize_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4/resnet50/rgb/2023-03-27-170131_imagenetsize/best_model_Thu_Mar_30_124755_2023.pth',
    'resnet50_oads_normalized_rgb_finetuned_imagenetsize_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned/resnet50/rgb/2023-03-27-170131_imagenetsize/best_model_Mon_Aug_14_105606_2023.pth',

    'resnet50_oads_normalized_rgb_jpeg_finetuned_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned/resnet50/rgb/jpeg/2023-07-26-174512/best_model_Mon_Aug_14_125307_2023.pth',
    'resnet50_oads_normalized_rgb_jpeg_finetuned_full_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full/resnet50/rgb/jpeg/2023-07-26-174512/best_model_Tue_Aug_15_093711_2023.pth',
    'resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4/resnet50/rgb/jpeg/2023-07-26-174512/best_model_Mon_Aug_14_125307_2023.pth',
    

    # ImageNet 223 Subclasses
    'resnet50_oads_normalized_rgb_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/resnet50/rgb/2023-07-24-132028/best_model_Mon_Aug_14_174711_2023.pth',
    'resnet50_oads_normalized_rgb_jpeg_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/resnet50/rgb/jpeg/2023-07-26-174512/best_model_Mon_Aug_14_174942_2023.pth',
    'resnet50_oads_normalized_rgb_jpeg_finetuned_imagenetsize_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/resnet50/rgb/jpeg/2023-03-23-172559_imagenetsize/best_model_Mon_Aug_14_173655_2023.pth',
    'resnet50_oads_normalized_rgb_finetuned_imagenetsize_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/resnet50/rgb/2023-03-27-170131_imagenetsize/best_model_Mon_Aug_14_173921_2023.pth',
    
    'resnet50_oads_normalized_rgb_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/resnet50/rgb/2023-07-24-132028/best_model_Tue_Aug_15_093245_2023.pth',
    'resnet50_oads_normalized_rgb_jpeg_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/resnet50/rgb/jpeg/2023-07-26-174512/best_model_Tue_Aug_15_093245_2023.pth',
    'resnet50_oads_normalized_rgb_jpeg_finetuned_full_imagenetsize_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/resnet50/rgb/jpeg/2023-03-23-172559_imagenetsize/best_model_Tue_Aug_15_092937_2023.pth',
    'resnet50_oads_normalized_rgb_finetuned_full_imagenetsize_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/resnet50/rgb/2023-03-27-170131_imagenetsize/best_model_Tue_Aug_15_092937_2023.pth',
    
    'resnet50_oads_normalized_rgb_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/resnet50/rgb/2023-07-24-132028/best_model_Mon_Aug_14_174711_2023.pth',
    'resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/resnet50/rgb/jpeg/2023-07-26-174512/best_model_Mon_Aug_14_174942_2023.pth',
    'resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_imagenetsize_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/resnet50/rgb/jpeg/2023-03-23-172559_imagenetsize/best_model_Mon_Aug_14_173655_2023.pth',
    'resnet50_oads_normalized_rgb_finetuned_layer4_imagenetsize_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/resnet50/rgb/2023-03-27-170131_imagenetsize/best_model_Mon_Aug_14_173921_2023.pth',


    ####### PLACES365

    'resnet50_places365_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/resnet50_places365/rgb/04-09-23-154316/best_model_Mon_Sep__4_154312_2023.pth',
    'resnet50_places365_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/resnet50_places365/rgb/04-09-23-202816/best_model_Mon_Sep__4_154312_2023.pth',
    'resnet50_places365_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/resnet50_places365/rgb/04-09-23-180737/best_model_Mon_Sep__4_154312_2023.pth',


    ################ COCO

    # 'fasterrcnn_resnet50_fpn_coco_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/fasterrcnn_resnet50_fpn_coco/rgb/29-02-24-185145/best_model_Thu_Feb_29_18:51:41_2024.pth',
    'fasterrcnn_resnet50_fpn_coco_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/fasterrcnn_resnet50_fpn_coco/rgb/16-04-24-095917/final_model_Tue_Apr_16_09:59:12_2024_epoch_11.pth',
    'fasterrcnn_resnet50_fpn_coco_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/fasterrcnn_resnet50_fpn_coco/rgb/16-04-24-160554/best_model_Tue_Apr_16_16:05:52_2024.pth',
    'fasterrcnn_resnet50_fpn_coco_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/fasterrcnn_resnet50_fpn_coco/rgb/16-04-24-160426/best_model_Tue_Apr_16_16:04:22_2024.pth',

    'fcn_resnet50_coco_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/fcn_resnet50_coco/rgb/24-02-24-174626/best_model_Sat_Feb_24_17:46:22_2024.pth',
    'fcn_resnet50_coco_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/fcn_resnet50_coco/rgb/24-02-24-170629/best_model_Sat_Feb_24_17:06:23_2024.pth',
    'fcn_resnet50_coco_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/fcn_resnet50_coco/rgb/24-02-24-202007/best_model_Sat_Feb_24_17:46:22_2024.pth',

    ## COCO + OADS
    'fcn_resnet50_coco_oads_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/fcn_resnet50_coco_oads/rgb/17-04-24-180520/best_model_Wed_Apr_17_18:05:20_2024.pth',
    'fcn_resnet50_coco_oads_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/fcn_resnet50_coco_oads/rgb/18-04-24-083112/best_model_Thu_Apr_18_08:31:11_2024.pth',
    'fcn_resnet50_coco_oads_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/fcn_resnet50_coco_oads/rgb/18-04-24-083011/best_model_Thu_Apr_18_08:30:11_2024.pth',

    ## COCO Object Crops
    'resnet50_coco_objects_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/resnet50_coco_objects/rgb/reps/2024-04-23-093017_400x400/best_model_Tue_Apr_23_14:29:57_2024.pth',
    'resnet50_coco_objects_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/resnet50_coco_objects/rgb/reps/2024-04-23-093017_400x400/best_model_Tue_Apr_23_14:40:06_2024.pth',
    'resnet50_coco_objects_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/resnet50_coco_objects/rgb/reps/2024-04-23-093017_400x400/best_model_Tue_Apr_23_14:35:03_2024.pth',
    
    'resnet50_coco_objects_zoom_80_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/resnet50_coco_objects/rgb_zoom-80/reps/2024-05-05-005047_400x400/best_model_Sun_May__5_14:43:26_2024.pth',
    'resnet50_coco_objects_zoom_80_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/resnet50_coco_objects/rgb_zoom-80/reps/2024-05-05-005047_400x400/best_model_Sun_May__5_14:55:58_2024.pth',
    'resnet50_coco_objects_zoom_80_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/resnet50_coco_objects/rgb_zoom-80/reps/2024-05-05-005047_400x400/best_model_Sun_May__5_15:14:30_2024.pth',
    'resnet50_coco_objects_zoom_150_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/resnet50_coco_objects/rgb_zoom-150/reps/2024-05-05-010418_400x400/best_model_Sun_May__5_14:42:19_2024.pth',
    'resnet50_coco_objects_zoom_150_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/resnet50_coco_objects/rgb_zoom-150/reps/2024-05-05-010418_400x400/best_model_Sun_May__5_14:59:01_2024.pth',
    'resnet50_coco_objects_zoom_150_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/resnet50_coco_objects/rgb_zoom-150/reps/2024-05-05-010418_400x400/best_model_Sun_May__5_15:06:29_2024.pth',

    'vit_b_16_coco_objects_zoom_0_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/vit_b_16_coco_objects/rgb_zoom-0/reps/2024-05-09-112255_224x224/best_model_Tue_May_14_10:56:10_2024.pth',
    'vit_b_16_coco_objects_zoom_0_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/vit_b_16_coco_objects/rgb_zoom-0/reps/2024-05-09-112255_224x224/best_model_Fri_May_10_17:50:16_2024.pth',
    'vit_b_16_coco_objects_zoom_0_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/vit_b_16_coco_objects/rgb_zoom-0/reps/2024-05-09-112255_224x224/best_model_Fri_May_10_19:40:59_2024.pth',
    'vit_b_16_coco_objects_zoom_80_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/vit_b_16_coco_objects/rgb_zoom-80/reps/2024-05-09-121357_224x224/best_model_Tue_May_14_10:56:10_2024.pth',
    'vit_b_16_coco_objects_zoom_80_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/vit_b_16_coco_objects/rgb_zoom-80/reps/2024-05-09-121357_224x224/best_model_Fri_May_10_17:52:56_2024.pth',
    'vit_b_16_coco_objects_zoom_80_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/vit_b_16_coco_objects/rgb_zoom-80/reps/2024-05-09-121357_224x224/best_model_Fri_May_10_19:25:13_2024.pth',
    'vit_b_16_coco_objects_zoom_150_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/vit_b_16_coco_objects/rgb_zoom-150/reps/2024-05-09-144317_224x224/best_model_Tue_May_14_10:56:10_2024.pth',
    'vit_b_16_coco_objects_zoom_150_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/vit_b_16_coco_objects/rgb_zoom-150/reps/2024-05-09-144317_224x224/best_model_Fri_May_10_17:57:32_2024.pth',
    'vit_b_16_coco_objects_zoom_150_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/vit_b_16_coco_objects/rgb_zoom-150/reps/2024-05-09-144317_224x224/best_model_Fri_May_10_19:00:03_2024.pth',
    
    'alexnet_coco_objects_zoom_0_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/alexnet_coco_objects/rgb_zoom-0/reps/2024-05-10-031618_400x400/best_model_Tue_May_14_09:09:48_2024.pth',
    'alexnet_coco_objects_zoom_0_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/alexnet_coco_objects/rgb_zoom-0/reps/2024-05-10-031618_400x400/best_model_Fri_May_10_20:24:55_2024.pth',
    'alexnet_coco_objects_zoom_0_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/alexnet_coco_objects/rgb_zoom-0/reps/2024-05-10-031618_400x400/best_model_Fri_May_10_19:47:01_2024.pth',
    'alexnet_coco_objects_zoom_80_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/alexnet_coco_objects/rgb_zoom-80/reps/2024-05-10-031314_400x400/best_model_Tue_May_14_09:11:24_2024.pth',
    'alexnet_coco_objects_zoom_80_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/alexnet_coco_objects/rgb_zoom-80/reps/2024-05-10-031314_400x400/best_model_Fri_May_10_20:19:46_2024.pth',
    'alexnet_coco_objects_zoom_80_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/alexnet_coco_objects/rgb_zoom-80/reps/2024-05-10-031314_400x400/best_model_Fri_May_10_19:47:01_2024.pth',
    'alexnet_coco_objects_zoom_150_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/alexnet_coco_objects/rgb_zoom-150/reps/2024-05-10-031040_400x400/best_model_Tue_May_14_09:11:24_2024.pth',
    'alexnet_coco_objects_zoom_150_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/alexnet_coco_objects/rgb_zoom-150/reps/2024-05-10-031040_400x400/best_model_Fri_May_10_19:56:00_2024.pth',
    'alexnet_coco_objects_zoom_150_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/alexnet_coco_objects/rgb_zoom-150/reps/2024-05-10-031040_400x400/best_model_Fri_May_10_19:55:37_2024.pth',

    ################ ADE
    'resnet50_ade20k_scenes_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/resnet50_ade20k_scenes/rgb/02-05-24-171212/best_model_Thu_May__2_17:12:11_2024.pth',
    'resnet50_ade20k_scenes_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/resnet50_ade20k_scenes/rgb/02-05-24-171344/best_model_Thu_May__2_17:13:43_2024.pth',
    'resnet50_ade20k_scenes_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/resnet50_ade20k_scenes/rgb/02-05-24-155946/best_model_Thu_May__2_15:59:45_2024.pth',
    
    'resnet50_ade20k_objects_zoom_0_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/resnet50_ade20k_objects/rgb_zoom-0/reps/2024-05-02-193520_400x400/best_model_Fri_May__3_08:58:04_2024.pth',
    'resnet50_ade20k_objects_zoom_0_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/resnet50_ade20k_objects/rgb_zoom-0/reps/2024-05-02-193520_400x400/best_model_Fri_May__3_09:29:54_2024.pth',
    'resnet50_ade20k_objects_zoom_0_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/resnet50_ade20k_objects/rgb_zoom-0/reps/2024-05-02-193520_400x400/best_model_Fri_May__3_09:28:52_2024.pth',
    
    'resnet50_ade20k_objects_zoom_80_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/resnet50_ade20k_objects/rgb_zoom-80/reps/2024-05-02-193213_400x400/best_model_Fri_May__3_09:00:12_2024.pth',
    'resnet50_ade20k_objects_zoom_80_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/resnet50_ade20k_objects/rgb_zoom-80/reps/2024-05-02-193213_400x400/best_model_Fri_May__3_09:31:54_2024.pth',
    'resnet50_ade20k_objects_zoom_80_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/resnet50_ade20k_objects/rgb_zoom-80/reps/2024-05-02-193213_400x400/best_model_Fri_May__3_09:27:51_2024.pth',
    
    'resnet50_ade20k_objects_zoom_150_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/resnet50_ade20k_objects/rgb_zoom-150/reps/2024-05-02-185437_400x400/best_model_Fri_May__3_09:03:41_2024.pth',
    'resnet50_ade20k_objects_zoom_150_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/resnet50_ade20k_objects/rgb_zoom-150/reps/2024-05-02-185437_400x400/best_model_Fri_May__3_09:33:56_2024.pth',
    'resnet50_ade20k_objects_zoom_150_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/resnet50_ade20k_objects/rgb_zoom-150/reps/2024-05-02-185437_400x400/best_model_Fri_May__3_09:24:21_2024.pth',
    
    'vit_b_16_ade20k_objects_zoom_0_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/',
    'vit_b_16_ade20k_objects_zoom_0_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/vit_b_16_ade20k_objects/rgb_zoom-0/reps/2024-05-07-112904_224x224/best_model_Wed_May__8_14:13:58_2024.pth',
    'vit_b_16_ade20k_objects_zoom_0_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/vit_b_16_ade20k_objects/rgb_zoom-0/reps/2024-05-07-112904_224x224/best_model_Wed_May__8_14:18:10_2024.pth',
    'vit_b_16_ade20k_objects_zoom_80_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/',
    'vit_b_16_ade20k_objects_zoom_80_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/vit_b_16_ade20k_objects/rgb_zoom-80/reps/2024-05-07-112904_224x224/best_model_Wed_May__8_14:16:03_2024.pth',
    'vit_b_16_ade20k_objects_zoom_80_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/vit_b_16_ade20k_objects/rgb_zoom-80/reps/2024-05-07-112904_224x224/best_model_Wed_May__8_14:17:34_2024.pth',
    'vit_b_16_ade20k_objects_zoom_150_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/',
    'vit_b_16_ade20k_objects_zoom_150_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/vit_b_16_ade20k_objects/rgb_zoom-150/reps/2024-05-07-122345_224x224/best_model_Wed_May__8_15:43:25_2024.pth',
    'vit_b_16_ade20k_objects_zoom_150_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/vit_b_16_ade20k_objects/rgb_zoom-150/reps/2024-05-07-122345_224x224/best_model_Wed_May__8_14:17:40_2024.pth',
    
    'alexnet_ade20k_objects_zoom_0_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/',
    'alexnet_ade20k_objects_zoom_0_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/alexnet_ade20k_objects/rgb_zoom-0/reps/2024-05-07-234245_224x224/best_model_Wed_May__8_14:14:03_2024.pth',
    'alexnet_ade20k_objects_zoom_0_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/alexnet_ade20k_objects/rgb_zoom-0/reps/2024-05-07-234245_224x224/best_model_Wed_May__8_12:43:11_2024.pth',
    'alexnet_ade20k_objects_zoom_80_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/',
    'alexnet_ade20k_objects_zoom_80_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/alexnet_ade20k_objects/rgb_zoom-80/reps/2024-05-07-235007_224x224/best_model_Wed_May__8_14:12:30_2024.pth',
    'alexnet_ade20k_objects_zoom_80_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/alexnet_ade20k_objects/rgb_zoom-80/reps/2024-05-07-235007_224x224/best_model_Wed_May__8_12:43:04_2024.pth',
    'alexnet_ade20k_objects_zoom_150_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/',
    'alexnet_ade20k_objects_zoom_150_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/alexnet_ade20k_objects/rgb_zoom-150/reps/2024-05-07-235143_224x224/best_model_Wed_May__8_12:49:12_2024.pth',
    'alexnet_ade20k_objects_zoom_150_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/',
    ################ 

    'resnet18_oads_rgb_finetuned_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/finetuned/resnet18/rgb/2023-03-01-180417/best_model_Thu_Mar_16_173413_2023.pth',
    'resnet18_oads_coc_finetuned_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/finetuned/resnet18/coc/2023-03-01-180805/best_model_Thu_Mar_16_173413_2023.pth',
    'resnet18_oads_rgb_jpeg_finetuned_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/finetuned/resnet18/rgb/jpeg/2023-03-01-180732/best_model_Thu_Mar_16_173413_2023.pth',
    'resnet18_oads_coc_jpeg_finetuned_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/finetuned/resnet18/coc/jpeg/2023-03-01-180801/best_model_Thu_Mar_16_173413_2023.pth',

    'resnet18_oads_rgb_finetuned_full_on_imagenet': '',
    'resnet18_oads_coc_finetuned_full_on_imagenet': '',
    'resnet18_oads_rgb_jpeg_finetuned_full_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/finetuned_full/resnet18/rgb/jpeg/2023-03-01-180732/best_model_Thu_Mar_23_084928_2023.pth',
    'resnet18_oads_coc_jpeg_finetuned_full_on_imagenet': '',

    'resnet18_oads_rgb_finetuned_layer4_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/layer4/resnet18/rgb/2023-03-01-180417/final_model_Mon_Mar_13_152555_2023.pth',
    'resnet18_oads_coc_finetuned_layer4_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/layer4/resnet18/coc/2023-03-01-180805/best_model_Tue_Mar_21_114944_2023.pth',
    'resnet18_oads_rgb_jpeg_finetuned_layer4_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/layer4/resnet18/rgb/jpeg/2023-03-01-180732/best_model_Mon_Mar_13_152555_2023.pth',
    'resnet18_oads_coc_jpeg_finetuned_layer4_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/layer4/resnet18/coc/jpeg/2023-03-01-180801/best_model_Tue_Mar_21_114944_2023.pth',
    
    'resnet18_oads_rgb_jpeg_finetuned_layer4_imagenetsize_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/layer4/resnet18/rgb/jpeg/2023-03-17-16-55-42_imagenet_size/best_model_Tue_Mar_21_153240_2023.pth',

    'resnet50_oads_rgb_finetuned_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/finetuned/resnet50/rgb/2023-02-23-172111/best_model_Thu_Mar_16_173413_2023.pth',
    'resnet50_oads_coc_finetuned_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/finetuned/resnet50/coc/2023-02-23-172129/best_model_Thu_Mar_16_173413_2023.pth',
    'resnet50_oads_rgb_jpeg_finetuned_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/finetuned/resnet50/rgb/jpeg/2023-02-23-172111/best_model_Thu_Mar_16_173413_2023.pth',
    'resnet50_oads_coc_jpeg_finetuned_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/finetuned/resnet50/coc/jpeg/2023-02-23-172118/best_model_Thu_Mar_16_173413_2023.pth',

    'resnet50_oads_rgb_finetuned_full_on_imagenet': '',
    'resnet50_oads_coc_finetuned_full_on_imagenet': '',
    'resnet50_oads_rgb_jpeg_finetuned_full_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/finetuned_full/resnet50/rgb/jpeg/2023-02-23-172111/best_model_Thu_Mar_16_173413_2023.pth',
    'resnet50_oads_coc_jpeg_finetuned_full_on_imagenet': '',

    'resnet50_oads_rgb_finetuned_layer4_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/layer4/resnet50/rgb/2023-02-23-172111/best_model_Mon_Mar_13_152555_2023.pth',
    'resnet50_oads_coc_finetuned_layer4_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/layer4/resnet50/coc/2023-02-23-172129/final_model_Mon_Mar_13_152555_2023.pth',
    'resnet50_oads_rgb_jpeg_finetuned_layer4_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/layer4/resnet50/rgb/jpeg/2023-02-23-172111/best_model_Mon_Mar_13_152555_2023.pth',
    'resnet50_oads_coc_jpeg_finetuned_layer4_on_imagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/layer4/resnet50/coc/jpeg/2023-02-23-172118/final_model_Mon_Mar_13_152555_2023.pth',

    'resnet50_imagenet_400x400_low_res': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/18-09-23-114037_pre_resize_224x224_then_400x400/best_model_18-09-23-114037.pth',

    'vit_b_16_oads_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/vit_b_16/rgb/reps/2024-04-16-115415_224x224/best_model_Tue_Apr_16_16:15:49_2024.pth',
    'vit_b_16_oads_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/vit_b_16/rgb/reps/2024-04-16-115415_224x224/best_model_Tue_Apr_16_16:18:31_2024.pth',
    'vit_b_16_oads_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/vit_b_16/rgb/reps/2024-04-16-115415_224x224/best_model_Tue_Apr_16_16:16:57_2024.pth',
    
    'resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/resnet50/26-04-24-104221_IM_pretrained_bounding_boxes_p100/best_model_26-04-24-104221.pth',
    'resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p70': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/resnet50/26-04-24-104118_IM_pretrained_bounding_boxes_p70/best_model_26-04-24-104118.pth',
    
    'resnet50_subimagenet_bounding_boxes_p70': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/resnet50/26-04-24-211744_bounding_boxes_p70/best_model_26-04-24-211744.pth',
    'resnet50_subimagenet_bounding_boxes_p100': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/resnet50/26-04-24-191421_bounding_boxes_p100/best_model_26-04-24-191421.pth',
    'resnet50_subimagenet_bounding_boxes_p100_zoom_30': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/resnet50/26-04-24-212845_bounding_boxes_p100_zoom-30/best_model_26-04-24-212845.pth',
    'resnet50_subimagenet_bounding_boxes_p100_zoom_50': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/resnet50/26-04-24-232159_bounding_boxes_p100_zoom-50/best_model_26-04-24-232159.pth',
    'resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_100': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/resnet50/02-05-24-100434/best_model_02-05-24-100434.pth',
    'resnet50_subimagenet_bounding_boxes_p100_zoom_100': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/resnet50/02-05-24-181439_bounding_boxes_p100_zoom-100/best_model_02-05-24-181439.pth',

    'resnet50_subimagenet_bounding_boxes_p100_zoom_80': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/resnet50/04-05-24-023343_bounding_boxes_p100_zoom-80/best_model_04-05-24-023343.pth',
    'resnet50_subimagenet_bounding_boxes_p100_zoom_150': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/resnet50/04-05-24-034841_bounding_boxes_p100_zoom-150/best_model_04-05-24-034841.pth',

    'resnet50_oads_zoom_50_finetuned_full_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_full_subclasses/resnet50/rgb_zoom-50/reps/2024-04-23-173628_400x400/best_model_Tue_Apr_30_18:42:00_2024.pth',
    'resnet50_oads_zoom_50_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/resnet50/rgb_zoom-50/reps/2024-04-23-173628_400x400/final_model_Tue_Apr_30_16:35:21_2024.pth',
    'resnet50_oads_zoom_50_finetuned_layer4_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/layer4_subclasses/resnet50/rgb_zoom-50/reps/2024-04-23-173628_400x400/best_model_Wed_May__1_08:49:57_2024.pth',
    
    'resnet50_oads_zoom_80_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/resnet50/rgb_zoom-80/reps/2024-05-01-211514_400x400/best_model_Thu_May__2_08:06:35_2024.pth',
    'resnet50_oads_zoom_100_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/resnet50/rgb_zoom-100/reps/2024-05-01-211821_400x400/best_model_Thu_May__2_08:08:00_2024.pth',
    'resnet50_oads_zoom_150_finetuned_on_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/robustpy_finetuned/results/normalized/finetuned_subclasses/resnet50/rgb_zoom-150/reps/2024-05-01-211852_400x400/best_model_Thu_May__2_08:12:01_2024.pth',
    'resnet50_all_boxes_imagenet': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/resnet50/02-05-24-132729_IM_pretrained_all_boxes/best_model_02-05-24-132729.pth',

    'vit_b_16_subimagenet_bounding_boxes_p100_zoom_150': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/05-05-24-221805_vit_b_16_bounding_boxes_p100_zoom-150/best_model_05-05-24-221805.pth',
    'vit_b_16_subimagenet_bounding_boxes_p100_zoom_80': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/08-05-24-023732_vit_b_16_bounding_boxes_p100_zoom-80/best_model_08-05-24-023732.pth',
    'vit_b_16_subimagenet_bounding_boxes_p100_zoom_0': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/08-05-24-044009_vit_b_16_bounding_boxes_p100_zoom-0/best_model_08-05-24-044009.pth',
    'alexnet_subimagenet_bounding_boxes_p100_zoom_150': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/08-05-24-053840_alexnet_bounding_boxes_p100_zoom-150/best_model_08-05-24-053840.pth',
    'alexnet_subimagenet_bounding_boxes_p100_zoom_80': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/08-05-24-045800_alexnet_bounding_boxes_p100_zoom-80/best_model_08-05-24-045800.pth',
    'alexnet_subimagenet_bounding_boxes_p100_zoom_0': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/08-05-24-045129_alexnet_bounding_boxes_p100_zoom-0/best_model_08-05-24-045129.pth',

    'resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_0': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/09-05-24-034811_resnet50_IM_pretrained_bounding_boxes_p100_zoom-0/best_model_09-05-24-034811.pth',
    'resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_80': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/09-05-24-035601_resnet50_IM_pretrained_bounding_boxes_p100_zoom-80/best_model_09-05-24-035601.pth',
    'resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_150': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/09-05-24-035629_resnet50_IM_pretrained_bounding_boxes_p100_zoom-150/best_model_09-05-24-035629.pth',
    'alexnet_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_0': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/09-05-24-045357_alexnet_IM_pretrained_bounding_boxes_p100_zoom-0/best_model_09-05-24-045357.pth',
    'alexnet_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_80': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/09-05-24-042953_alexnet_IM_pretrained_bounding_boxes_p100_zoom-80/best_model_09-05-24-042953.pth',
    'alexnet_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_150': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/09-05-24-042208_alexnet_IM_pretrained_bounding_boxes_p100_zoom-150/best_model_09-05-24-042208.pth',
    # 'vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_0': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/',
    # 'vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_80': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/',
    # 'vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_150': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/',

    'vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_0': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/10-05-24-070509_vit_b_16_IM_pretrained_bounding_boxes_p100_zoom-0/best_model_10-05-24-070509.pth',
    'vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_80': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/10-05-24-071944_vit_b_16_IM_pretrained_bounding_boxes_p100_zoom-80/best_model_10-05-24-071944.pth',
    'vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_150': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/10-05-24-085242_vit_b_16_IM_pretrained_bounding_boxes_p100_zoom-150/best_model_10-05-24-085242.pth',
    
    'resnet50_imagenet_finetuned_on_all_boxes_p50_zoom_0': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/11-05-24-024537_resnet50_IM_pretrained_all_boxes_p50_zoom-0/best_model_11-05-24-024537.pth',

    'vit_b_16_imagenet': '',
    'vit_b_16_subimagenet': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/11-05-24-051231_vit_b_16/best_model_11-05-24-051231.pth',


    'transformer_b16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_150': f'{home_path}/projects/fmg_storage/trained_models/imagenet_results/18-05-24-100831_transformer_b16_IM_pretrained_bounding_boxes_p100_zoom-150/best_model_18-05-24-100831.pth',
    }

# for epoch in range(90):
    # model_paths[f'resnet50_julio_epoch_{epoch}'] = f'/home/Public/Models/ResNet50_SEED1/model_{epoch}.pth'
    # model_paths[f'alexnet_julio_epoch_{epoch}'] = f'/home/Public/Models/AlexNet_SEED1/model_{epoch}.pth'
    # model_paths[f'resnet50_julio_epoch_{epoch}'] = f'/home/Public/Models/ResNet50_SEED2/model_{epoch}.pth'
    # model_paths[f'alexnet_julio_epoch_{epoch}'] = f'/home/Public/Models/AlexNet_SEED2/model_{epoch}.pth'
    # model_paths[f'resnet50_julio_epoch_{epoch}'] = f'/home/Public/Models/ResNet50_SEED3/model_{epoch}.pth'
    # model_paths[f'alexnet_julio_epoch_{epoch}'] = f'/home/Public/Models/AlexNet_SEED3/model_{epoch}.pth'
    # model_paths[f'resnet50_julio_epoch_{epoch}'] = f'/home/Public/Models/ResNet50_SEED4/model_{epoch}.pth'
    # model_paths[f'alexnet_julio_epoch_{epoch}'] = f'/home/Public/Models/AlexNet_SEED4/model_{epoch}.pth'
    # model_paths[f'resnet50_julio_epoch_{epoch}'] = f'/home/Public/Models/ResNet50_SEED5/model_{epoch}.pth'
    # model_paths[f'alexnet_julio_epoch_{epoch}'] = f'/home/Public/Models/AlexNet_SEED5/model_{epoch}.pth'
model_paths['resnet50_julio_random'] = ''
model_paths['alexnet_julio_random'] = ''


def transformer_b16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_150():
    return _transformer_model('transformer_b16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_150', num_classes=1000)

def vit_b_16_ade20k_objects_zoom_150_finetuned_full_on_subimagenet():
    return _vit_model('vit_b_16_ade20k_objects_zoom_150_finetuned_full_on_subimagenet', num_classes=1000)
def vit_b_16_ade20k_objects_zoom_150_finetuned_on_subimagenet():
    return _vit_model('vit_b_16_ade20k_objects_zoom_150_finetuned_on_subimagenet', num_classes=1000)
def vit_b_16_ade20k_objects_zoom_80_finetuned_layer4_on_subimagenet():
    return _vit_model('vit_b_16_ade20k_objects_zoom_80_finetuned_layer4_on_subimagenet', num_classes=1000)
def vit_b_16_ade20k_objects_zoom_80_finetuned_full_on_subimagenet():
    return _vit_model('vit_b_16_ade20k_objects_zoom_80_finetuned_full_on_subimagenet', num_classes=1000)
def vit_b_16_ade20k_objects_zoom_80_finetuned_on_subimagenet():
    return _vit_model('vit_b_16_ade20k_objects_zoom_80_finetuned_on_subimagenet', num_classes=1000)
def vit_b_16_ade20k_objects_zoom_0_finetuned_layer4_on_subimagenet():
    return _vit_model('vit_b_16_ade20k_objects_zoom_0_finetuned_layer4_on_subimagenet', num_classes=1000)
def vit_b_16_ade20k_objects_zoom_0_finetuned_full_on_subimagenet():
    return _vit_model('vit_b_16_ade20k_objects_zoom_0_finetuned_full_on_subimagenet', num_classes=1000)
def vit_b_16_ade20k_objects_zoom_0_finetuned_on_subimagenet():
    return _vit_model('vit_b_16_ade20k_objects_zoom_0_finetuned_on_subimagenet', num_classes=1000)
def vit_b_16_ade20k_objects_zoom_150_finetuned_layer4_on_subimagenet():
    return _vit_model('vit_b_16_ade20k_objects_zoom_150_finetuned_layer4_on_subimagenet', num_classes=1000)



def alexnet_ade20k_objects_zoom_0_finetuned_on_subimagenet():
    return _model('alexnet_ade20k_objects_zoom_0_finetuned_on_subimagenet', alexnet, fc_channels=512)
def alexnet_ade20k_objects_zoom_150_finetuned_full_on_subimagenet():
    return _model('alexnet_ade20k_objects_zoom_150_finetuned_full_on_subimagenet', alexnet, fc_channels=512)
def alexnet_ade20k_objects_zoom_150_finetuned_on_subimagenet():
    return _model('alexnet_ade20k_objects_zoom_150_finetuned_on_subimagenet', alexnet, fc_channels=512)
def alexnet_ade20k_objects_zoom_80_finetuned_layer4_on_subimagenet():
    return _model('alexnet_ade20k_objects_zoom_80_finetuned_layer4_on_subimagenet', alexnet, fc_channels=512)
def alexnet_ade20k_objects_zoom_80_finetuned_full_on_subimagenet():
    return _model('alexnet_ade20k_objects_zoom_80_finetuned_full_on_subimagenet', alexnet, fc_channels=512)
def alexnet_ade20k_objects_zoom_80_finetuned_on_subimagenet():
    return _model('alexnet_ade20k_objects_zoom_80_finetuned_on_subimagenet', alexnet, fc_channels=512)
def alexnet_ade20k_objects_zoom_0_finetuned_layer4_on_subimagenet():
    return _model('alexnet_ade20k_objects_zoom_0_finetuned_layer4_on_subimagenet', alexnet, fc_channels=512)
def alexnet_ade20k_objects_zoom_0_finetuned_full_on_subimagenet():
    return _model('alexnet_ade20k_objects_zoom_0_finetuned_full_on_subimagenet', alexnet, fc_channels=512)
def alexnet_ade20k_objects_zoom_150_finetuned_layer4_on_subimagenet():
    return _model('alexnet_ade20k_objects_zoom_150_finetuned_layer4_on_subimagenet', alexnet, fc_channels=512)



def vit_b_16_coco_objects_zoom_150_finetuned_full_on_subimagenet():
    return _vit_model('vit_b_16_coco_objects_zoom_150_finetuned_full_on_subimagenet', num_classes=1000)
def vit_b_16_coco_objects_zoom_150_finetuned_on_subimagenet():
    return _vit_model('vit_b_16_coco_objects_zoom_150_finetuned_on_subimagenet', num_classes=1000)
def vit_b_16_coco_objects_zoom_80_finetuned_layer4_on_subimagenet():
    return _vit_model('vit_b_16_coco_objects_zoom_80_finetuned_layer4_on_subimagenet', num_classes=1000)
def vit_b_16_coco_objects_zoom_80_finetuned_full_on_subimagenet():
    return _vit_model('vit_b_16_coco_objects_zoom_80_finetuned_full_on_subimagenet', num_classes=1000)
def vit_b_16_coco_objects_zoom_80_finetuned_on_subimagenet():
    return _vit_model('vit_b_16_coco_objects_zoom_80_finetuned_on_subimagenet', num_classes=1000)
def vit_b_16_coco_objects_zoom_0_finetuned_layer4_on_subimagenet():
    return _vit_model('vit_b_16_coco_objects_zoom_0_finetuned_layer4_on_subimagenet', num_classes=1000)
def vit_b_16_coco_objects_zoom_0_finetuned_full_on_subimagenet():
    return _vit_model('vit_b_16_coco_objects_zoom_0_finetuned_full_on_subimagenet', num_classes=1000)
def vit_b_16_coco_objects_zoom_0_finetuned_on_subimagenet():
    return _vit_model('vit_b_16_coco_objects_zoom_0_finetuned_on_subimagenet', num_classes=1000)
def vit_b_16_coco_objects_zoom_150_finetuned_layer4_on_subimagenet():
    return _vit_model('vit_b_16_coco_objects_zoom_150_finetuned_layer4_on_subimagenet', num_classes=1000)



def alexnet_coco_objects_zoom_0_finetuned_on_subimagenet():
    return _model('alexnet_coco_objects_zoom_0_finetuned_on_subimagenet', alexnet, fc_channels=512)
def alexnet_coco_objects_zoom_150_finetuned_full_on_subimagenet():
    return _model('alexnet_coco_objects_zoom_150_finetuned_full_on_subimagenet', alexnet, fc_channels=512)
def alexnet_coco_objects_zoom_150_finetuned_on_subimagenet():
    return _model('alexnet_coco_objects_zoom_150_finetuned_on_subimagenet', alexnet, fc_channels=512)
def alexnet_coco_objects_zoom_80_finetuned_layer4_on_subimagenet():
    return _model('alexnet_coco_objects_zoom_80_finetuned_layer4_on_subimagenet', alexnet, fc_channels=512)
def alexnet_coco_objects_zoom_80_finetuned_full_on_subimagenet():
    return _model('alexnet_coco_objects_zoom_80_finetuned_full_on_subimagenet', alexnet, fc_channels=512)
def alexnet_coco_objects_zoom_80_finetuned_on_subimagenet():
    return _model('alexnet_coco_objects_zoom_80_finetuned_on_subimagenet', alexnet, fc_channels=512)
def alexnet_coco_objects_zoom_0_finetuned_layer4_on_subimagenet():
    return _model('alexnet_coco_objects_zoom_0_finetuned_layer4_on_subimagenet', alexnet, fc_channels=512)
def alexnet_coco_objects_zoom_0_finetuned_full_on_subimagenet():
    return _model('alexnet_coco_objects_zoom_0_finetuned_full_on_subimagenet', alexnet, fc_channels=512)
def alexnet_coco_objects_zoom_150_finetuned_layer4_on_subimagenet():
    return _model('alexnet_coco_objects_zoom_150_finetuned_layer4_on_subimagenet', alexnet, fc_channels=512)

def resnet50_imagenet_finetuned_on_all_boxes_p50_zoom_0():
    return _model('resnet50_imagenet_finetuned_on_all_boxes_p50_zoom_0', resnet50, fc_channels=2048)

def resnet50_oads_zoom_150_finetuned_full_on_subimagenet():
    return _model('resnet50_oads_zoom_150_finetuned_full_on_subimagenet', resnet50, fc_channels=2048)
def resnet50_oads_zoom_80_finetuned_full_on_subimagenet():
    return _model('resnet50_oads_zoom_80_finetuned_full_on_subimagenet', resnet50, fc_channels=2048)

def resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_80():
    return _model('resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_80', resnet50, fc_channels=2048)
def resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_0():
    return _model('resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_0', resnet50, fc_channels=2048)
def resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_150():
    return _model('resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_150', resnet50, fc_channels=2048)



def alexnet_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_80():
    return _model('alexnet_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_80', alexnet, fc_channels=512)
def alexnet_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_0():
    return _model('alexnet_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_0', alexnet, fc_channels=512)
def alexnet_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_150():
    return _model('alexnet_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_150', alexnet, fc_channels=512)


def vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_80():
    return _vit_model('vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_80', num_classes=1000)
def vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_0():
    return _vit_model('vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_0', num_classes=1000)
def vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_150():
    return _vit_model('vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_150', num_classes=1000)

def alexnet_subimagenet_bounding_boxes_p100_zoom_0():
    return _model('alexnet_subimagenet_bounding_boxes_p100_zoom_0', alexnet, fc_channels=512)
def alexnet_subimagenet_bounding_boxes_p100_zoom_80():
    return _model('alexnet_subimagenet_bounding_boxes_p100_zoom_80', alexnet, fc_channels=512)
def alexnet_subimagenet_bounding_boxes_p100_zoom_150():
    return _model('alexnet_subimagenet_bounding_boxes_p100_zoom_150', alexnet, fc_channels=512)
def vit_b_16_subimagenet_bounding_boxes_p100_zoom_0():
    return _vit_model('vit_b_16_subimagenet_bounding_boxes_p100_zoom_0', num_classes=1000)
def vit_b_16_subimagenet_bounding_boxes_p100_zoom_80():
    return _vit_model('vit_b_16_subimagenet_bounding_boxes_p100_zoom_80', num_classes=1000)
def vit_b_16_subimagenet_bounding_boxes_p100_zoom_150():
    return _vit_model('vit_b_16_subimagenet_bounding_boxes_p100_zoom_150', num_classes=1000)

def resnet50_subimagenet_bounding_boxes_p100_zoom_150():
    return _model('resnet50_subimagenet_bounding_boxes_p100_zoom_150', resnet50, fc_channels=2048)
def resnet50_subimagenet_bounding_boxes_p100_zoom_80():
    return _model('resnet50_subimagenet_bounding_boxes_p100_zoom_80', resnet50, fc_channels=2048)


def resnet50_coco_objects_zoom_150_finetuned_full_on_subimagenet():
    return _model('resnet50_coco_objects_zoom_150_finetuned_full_on_subimagenet', resnet50, fc_channels=2048)
def resnet50_coco_objects_zoom_150_finetuned_on_subimagenet():
    return _model('resnet50_coco_objects_zoom_150_finetuned_on_subimagenet', resnet50, fc_channels=2048)
def resnet50_coco_objects_zoom_80_finetuned_layer4_on_subimagenet():
    return _model('resnet50_coco_objects_zoom_80_finetuned_layer4_on_subimagenet', resnet50, fc_channels=2048)
def resnet50_coco_objects_zoom_80_finetuned_full_on_subimagenet():
    return _model('resnet50_coco_objects_zoom_80_finetuned_full_on_subimagenet', resnet50, fc_channels=2048)
def resnet50_coco_objects_zoom_80_finetuned_on_subimagenet():
    return _model('resnet50_coco_objects_zoom_80_finetuned_on_subimagenet', resnet50, fc_channels=2048)
def resnet50_coco_objects_zoom_150_finetuned_layer4_on_subimagenet():
    return _model('resnet50_coco_objects_zoom_150_finetuned_layer4_on_subimagenet', resnet50, fc_channels=2048)

def resnet50_ade20k_objects_zoom_0_finetuned_full_on_subimagenet():
    return _model('resnet50_ade20k_objects_zoom_0_finetuned_full_on_subimagenet', resnet50, fc_channels=2048)
def resnet50_ade20k_objects_zoom_0_finetuned_on_subimagenet():
    return _model('resnet50_ade20k_objects_zoom_0_finetuned_on_subimagenet', resnet50, fc_channels=2048)
def resnet50_ade20k_objects_zoom_0_finetuned_layer4_on_subimagenet():
    return _model('resnet50_ade20k_objects_zoom_0_finetuned_layer4_on_subimagenet', resnet50, fc_channels=2048)
def resnet50_ade20k_objects_zoom_80_finetuned_on_subimagenet():
    return _model('resnet50_ade20k_objects_zoom_80_finetuned_on_subimagenet', resnet50, fc_channels=2048)
def resnet50_ade20k_objects_zoom_80_finetuned_layer4_on_subimagenet():
    return _model('resnet50_ade20k_objects_zoom_80_finetuned_layer4_on_subimagenet', resnet50, fc_channels=2048)
def resnet50_ade20k_objects_zoom_80_finetuned_full_on_subimagenet():
    return _model('resnet50_ade20k_objects_zoom_80_finetuned_full_on_subimagenet', resnet50, fc_channels=2048)
def resnet50_ade20k_objects_zoom_150_finetuned_on_subimagenet():
    return _model('resnet50_ade20k_objects_zoom_150_finetuned_on_subimagenet', resnet50, fc_channels=2048)
def resnet50_ade20k_objects_zoom_150_finetuned_layer4_on_subimagenet():
    return _model('resnet50_ade20k_objects_zoom_150_finetuned_layer4_on_subimagenet', resnet50, fc_channels=2048)
def resnet50_ade20k_objects_zoom_150_finetuned_full_on_subimagenet():
    return _model('resnet50_ade20k_objects_zoom_150_finetuned_full_on_subimagenet', resnet50, fc_channels=2048)



def resnet50_oads_zoom_80_finetuned_on_subimagenet():
    return _model('resnet50_oads_zoom_80_finetuned_on_subimagenet', resnet50, fc_channels=2048)
def resnet50_oads_zoom_100_finetuned_on_subimagenet():
    return _model('resnet50_oads_zoom_100_finetuned_on_subimagenet', resnet50, fc_channels=2048)
def resnet50_oads_zoom_150_finetuned_on_subimagenet():
    return _model('resnet50_oads_zoom_150_finetuned_on_subimagenet', resnet50, fc_channels=2048)
def resnet50_all_boxes_imagenet():
    return _model('resnet50_all_boxes_imagenet', resnet50, fc_channels=2048)

def resnet50_oads_zoom_50_finetuned_on_subimagenet():
    return _model('resnet50_oads_zoom_50_finetuned_on_subimagenet', resnet50, fc_channels=2048)
def resnet50_oads_zoom_50_finetuned_full_on_subimagenet():
    return _model('resnet50_oads_zoom_50_finetuned_full_on_subimagenet', resnet50, fc_channels=2048)
def resnet50_oads_zoom_50_finetuned_layer4_on_subimagenet():
    return _model('resnet50_oads_zoom_50_finetuned_layer4_on_subimagenet', resnet50, fc_channels=2048)

def resnet50_subimagenet_bounding_boxes_p70():
    return _model('resnet50_subimagenet_bounding_boxes_p70', resnet50, fc_channels=2048)
def resnet50_subimagenet_bounding_boxes_p100():
    return _model('resnet50_subimagenet_bounding_boxes_p100', resnet50, fc_channels=2048)
def resnet50_subimagenet_bounding_boxes_p100_zoom_30():
    return _model('resnet50_subimagenet_bounding_boxes_p100_zoom_30', resnet50, fc_channels=2048)
def resnet50_subimagenet_bounding_boxes_p100_zoom_50():
    return _model('resnet50_subimagenet_bounding_boxes_p100_zoom_50', resnet50, fc_channels=2048)
def resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_100():
    return _model('resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_100', resnet50, fc_channels=2048)
def resnet50_subimagenet_bounding_boxes_p100_zoom_100():
    return _model('resnet50_subimagenet_bounding_boxes_p100_zoom_100', resnet50, fc_channels=2048)

def resnet50_julio_random():
    return _model('resnet50_julio_random', resnet50, fc_channels=2048)
def resnet50_julio_epoch_0():
    return _model('resnet50_julio_epoch_0', resnet50, fc_channels=2048)
def resnet50_julio_epoch_1():
    return _model('resnet50_julio_epoch_1', resnet50, fc_channels=2048)
def resnet50_julio_epoch_2():
    return _model('resnet50_julio_epoch_2', resnet50, fc_channels=2048)
def resnet50_julio_epoch_3():
    return _model('resnet50_julio_epoch_3', resnet50, fc_channels=2048)
def resnet50_julio_epoch_4():
    return _model('resnet50_julio_epoch_4', resnet50, fc_channels=2048)
def resnet50_julio_epoch_5():
    return _model('resnet50_julio_epoch_5', resnet50, fc_channels=2048)
def resnet50_julio_epoch_6():
    return _model('resnet50_julio_epoch_6', resnet50, fc_channels=2048)
def resnet50_julio_epoch_7():
    return _model('resnet50_julio_epoch_7', resnet50, fc_channels=2048)
def resnet50_julio_epoch_8():
    return _model('resnet50_julio_epoch_8', resnet50, fc_channels=2048)
def resnet50_julio_epoch_9():
    return _model('resnet50_julio_epoch_9', resnet50, fc_channels=2048)
def resnet50_julio_epoch_10():
    return _model('resnet50_julio_epoch_10', resnet50, fc_channels=2048)
def resnet50_julio_epoch_11():
    return _model('resnet50_julio_epoch_11', resnet50, fc_channels=2048)
def resnet50_julio_epoch_12():
    return _model('resnet50_julio_epoch_12', resnet50, fc_channels=2048)
def resnet50_julio_epoch_13():
    return _model('resnet50_julio_epoch_13', resnet50, fc_channels=2048)
def resnet50_julio_epoch_14():
    return _model('resnet50_julio_epoch_14', resnet50, fc_channels=2048)
def resnet50_julio_epoch_15():
    return _model('resnet50_julio_epoch_15', resnet50, fc_channels=2048)
def resnet50_julio_epoch_16():
    return _model('resnet50_julio_epoch_16', resnet50, fc_channels=2048)
def resnet50_julio_epoch_17():
    return _model('resnet50_julio_epoch_17', resnet50, fc_channels=2048)
def resnet50_julio_epoch_18():
    return _model('resnet50_julio_epoch_18', resnet50, fc_channels=2048)
def resnet50_julio_epoch_19():
    return _model('resnet50_julio_epoch_19', resnet50, fc_channels=2048)
def resnet50_julio_epoch_20():
    return _model('resnet50_julio_epoch_20', resnet50, fc_channels=2048)
def resnet50_julio_epoch_21():
    return _model('resnet50_julio_epoch_21', resnet50, fc_channels=2048)
def resnet50_julio_epoch_22():
    return _model('resnet50_julio_epoch_22', resnet50, fc_channels=2048)
def resnet50_julio_epoch_23():
    return _model('resnet50_julio_epoch_23', resnet50, fc_channels=2048)
def resnet50_julio_epoch_24():
    return _model('resnet50_julio_epoch_24', resnet50, fc_channels=2048)
def resnet50_julio_epoch_25():
    return _model('resnet50_julio_epoch_25', resnet50, fc_channels=2048)
def resnet50_julio_epoch_26():
    return _model('resnet50_julio_epoch_26', resnet50, fc_channels=2048)
def resnet50_julio_epoch_27():
    return _model('resnet50_julio_epoch_27', resnet50, fc_channels=2048)
def resnet50_julio_epoch_28():
    return _model('resnet50_julio_epoch_28', resnet50, fc_channels=2048)
def resnet50_julio_epoch_29():
    return _model('resnet50_julio_epoch_29', resnet50, fc_channels=2048)
def resnet50_julio_epoch_30():
    return _model('resnet50_julio_epoch_30', resnet50, fc_channels=2048)
def resnet50_julio_epoch_31():
    return _model('resnet50_julio_epoch_31', resnet50, fc_channels=2048)
def resnet50_julio_epoch_32():
    return _model('resnet50_julio_epoch_32', resnet50, fc_channels=2048)
def resnet50_julio_epoch_33():
    return _model('resnet50_julio_epoch_33', resnet50, fc_channels=2048)
def resnet50_julio_epoch_34():
    return _model('resnet50_julio_epoch_34', resnet50, fc_channels=2048)
def resnet50_julio_epoch_35():
    return _model('resnet50_julio_epoch_35', resnet50, fc_channels=2048)
def resnet50_julio_epoch_36():
    return _model('resnet50_julio_epoch_36', resnet50, fc_channels=2048)
def resnet50_julio_epoch_37():
    return _model('resnet50_julio_epoch_37', resnet50, fc_channels=2048)
def resnet50_julio_epoch_38():
    return _model('resnet50_julio_epoch_38', resnet50, fc_channels=2048)
def resnet50_julio_epoch_39():
    return _model('resnet50_julio_epoch_39', resnet50, fc_channels=2048)
def resnet50_julio_epoch_40():
    return _model('resnet50_julio_epoch_40', resnet50, fc_channels=2048)
def resnet50_julio_epoch_41():
    return _model('resnet50_julio_epoch_41', resnet50, fc_channels=2048)
def resnet50_julio_epoch_42():
    return _model('resnet50_julio_epoch_42', resnet50, fc_channels=2048)
def resnet50_julio_epoch_43():
    return _model('resnet50_julio_epoch_43', resnet50, fc_channels=2048)
def resnet50_julio_epoch_44():
    return _model('resnet50_julio_epoch_44', resnet50, fc_channels=2048)
def resnet50_julio_epoch_45():
    return _model('resnet50_julio_epoch_45', resnet50, fc_channels=2048)
def resnet50_julio_epoch_46():
    return _model('resnet50_julio_epoch_46', resnet50, fc_channels=2048)
def resnet50_julio_epoch_47():
    return _model('resnet50_julio_epoch_47', resnet50, fc_channels=2048)
def resnet50_julio_epoch_48():
    return _model('resnet50_julio_epoch_48', resnet50, fc_channels=2048)
def resnet50_julio_epoch_49():
    return _model('resnet50_julio_epoch_49', resnet50, fc_channels=2048)
def resnet50_julio_epoch_50():
    return _model('resnet50_julio_epoch_50', resnet50, fc_channels=2048)
def resnet50_julio_epoch_51():
    return _model('resnet50_julio_epoch_51', resnet50, fc_channels=2048)
def resnet50_julio_epoch_52():
    return _model('resnet50_julio_epoch_52', resnet50, fc_channels=2048)
def resnet50_julio_epoch_53():
    return _model('resnet50_julio_epoch_53', resnet50, fc_channels=2048)
def resnet50_julio_epoch_54():
    return _model('resnet50_julio_epoch_54', resnet50, fc_channels=2048)
def resnet50_julio_epoch_55():
    return _model('resnet50_julio_epoch_55', resnet50, fc_channels=2048)
def resnet50_julio_epoch_56():
    return _model('resnet50_julio_epoch_56', resnet50, fc_channels=2048)
def resnet50_julio_epoch_57():
    return _model('resnet50_julio_epoch_57', resnet50, fc_channels=2048)
def resnet50_julio_epoch_58():
    return _model('resnet50_julio_epoch_58', resnet50, fc_channels=2048)
def resnet50_julio_epoch_59():
    return _model('resnet50_julio_epoch_59', resnet50, fc_channels=2048)
def resnet50_julio_epoch_60():
    return _model('resnet50_julio_epoch_60', resnet50, fc_channels=2048)
def resnet50_julio_epoch_61():
    return _model('resnet50_julio_epoch_61', resnet50, fc_channels=2048)
def resnet50_julio_epoch_62():
    return _model('resnet50_julio_epoch_62', resnet50, fc_channels=2048)
def resnet50_julio_epoch_63():
    return _model('resnet50_julio_epoch_63', resnet50, fc_channels=2048)
def resnet50_julio_epoch_64():
    return _model('resnet50_julio_epoch_64', resnet50, fc_channels=2048)
def resnet50_julio_epoch_65():
    return _model('resnet50_julio_epoch_65', resnet50, fc_channels=2048)
def resnet50_julio_epoch_66():
    return _model('resnet50_julio_epoch_66', resnet50, fc_channels=2048)
def resnet50_julio_epoch_67():
    return _model('resnet50_julio_epoch_67', resnet50, fc_channels=2048)
def resnet50_julio_epoch_68():
    return _model('resnet50_julio_epoch_68', resnet50, fc_channels=2048)
def resnet50_julio_epoch_69():
    return _model('resnet50_julio_epoch_69', resnet50, fc_channels=2048)
def resnet50_julio_epoch_70():
    return _model('resnet50_julio_epoch_70', resnet50, fc_channels=2048)
def resnet50_julio_epoch_71():
    return _model('resnet50_julio_epoch_71', resnet50, fc_channels=2048)
def resnet50_julio_epoch_72():
    return _model('resnet50_julio_epoch_72', resnet50, fc_channels=2048)
def resnet50_julio_epoch_73():
    return _model('resnet50_julio_epoch_73', resnet50, fc_channels=2048)
def resnet50_julio_epoch_74():
    return _model('resnet50_julio_epoch_74', resnet50, fc_channels=2048)
def resnet50_julio_epoch_75():
    return _model('resnet50_julio_epoch_75', resnet50, fc_channels=2048)
def resnet50_julio_epoch_76():
    return _model('resnet50_julio_epoch_76', resnet50, fc_channels=2048)
def resnet50_julio_epoch_77():
    return _model('resnet50_julio_epoch_77', resnet50, fc_channels=2048)
def resnet50_julio_epoch_78():
    return _model('resnet50_julio_epoch_78', resnet50, fc_channels=2048)
def resnet50_julio_epoch_79():
    return _model('resnet50_julio_epoch_79', resnet50, fc_channels=2048)
def resnet50_julio_epoch_80():
    return _model('resnet50_julio_epoch_80', resnet50, fc_channels=2048)
def resnet50_julio_epoch_81():
    return _model('resnet50_julio_epoch_81', resnet50, fc_channels=2048)
def resnet50_julio_epoch_82():
    return _model('resnet50_julio_epoch_82', resnet50, fc_channels=2048)
def resnet50_julio_epoch_83():
    return _model('resnet50_julio_epoch_83', resnet50, fc_channels=2048)
def resnet50_julio_epoch_84():
    return _model('resnet50_julio_epoch_84', resnet50, fc_channels=2048)
def resnet50_julio_epoch_85():
    return _model('resnet50_julio_epoch_85', resnet50, fc_channels=2048)
def resnet50_julio_epoch_86():
    return _model('resnet50_julio_epoch_86', resnet50, fc_channels=2048)
def resnet50_julio_epoch_87():
    return _model('resnet50_julio_epoch_87', resnet50, fc_channels=2048)
def resnet50_julio_epoch_88():
    return _model('resnet50_julio_epoch_88', resnet50, fc_channels=2048)
def resnet50_julio_epoch_89():
    return _model('resnet50_julio_epoch_89', resnet50, fc_channels=2048)
def resnet50_julio_epoch_90():
    return _model('resnet50_julio_epoch_90', resnet50, fc_channels=2048)
def alexnet_julio_random():
    return _model('alexnet_julio_random', alexnet, fc_channels=512)
def alexnet_julio_epoch_0():
    return _model('alexnet_julio_epoch_0', alexnet, fc_channels=512)
def alexnet_julio_epoch_1():
    return _model('alexnet_julio_epoch_1', alexnet, fc_channels=512)
def alexnet_julio_epoch_2():
    return _model('alexnet_julio_epoch_2', alexnet, fc_channels=512)
def alexnet_julio_epoch_3():
    return _model('alexnet_julio_epoch_3', alexnet, fc_channels=512)
def alexnet_julio_epoch_4():
    return _model('alexnet_julio_epoch_4', alexnet, fc_channels=512)
def alexnet_julio_epoch_5():
    return _model('alexnet_julio_epoch_5', alexnet, fc_channels=512)
def alexnet_julio_epoch_6():
    return _model('alexnet_julio_epoch_6', alexnet, fc_channels=512)
def alexnet_julio_epoch_7():
    return _model('alexnet_julio_epoch_7', alexnet, fc_channels=512)
def alexnet_julio_epoch_8():
    return _model('alexnet_julio_epoch_8', alexnet, fc_channels=512)
def alexnet_julio_epoch_9():
    return _model('alexnet_julio_epoch_9', alexnet, fc_channels=512)
def alexnet_julio_epoch_10():
    return _model('alexnet_julio_epoch_10', alexnet, fc_channels=512)
def alexnet_julio_epoch_11():
    return _model('alexnet_julio_epoch_11', alexnet, fc_channels=512)
def alexnet_julio_epoch_12():
    return _model('alexnet_julio_epoch_12', alexnet, fc_channels=512)
def alexnet_julio_epoch_13():
    return _model('alexnet_julio_epoch_13', alexnet, fc_channels=512)
def alexnet_julio_epoch_14():
    return _model('alexnet_julio_epoch_14', alexnet, fc_channels=512)
def alexnet_julio_epoch_15():
    return _model('alexnet_julio_epoch_15', alexnet, fc_channels=512)
def alexnet_julio_epoch_16():
    return _model('alexnet_julio_epoch_16', alexnet, fc_channels=512)
def alexnet_julio_epoch_17():
    return _model('alexnet_julio_epoch_17', alexnet, fc_channels=512)
def alexnet_julio_epoch_18():
    return _model('alexnet_julio_epoch_18', alexnet, fc_channels=512)
def alexnet_julio_epoch_19():
    return _model('alexnet_julio_epoch_19', alexnet, fc_channels=512)
def alexnet_julio_epoch_20():
    return _model('alexnet_julio_epoch_20', alexnet, fc_channels=512)
def alexnet_julio_epoch_21():
    return _model('alexnet_julio_epoch_21', alexnet, fc_channels=512)
def alexnet_julio_epoch_22():
    return _model('alexnet_julio_epoch_22', alexnet, fc_channels=512)
def alexnet_julio_epoch_23():
    return _model('alexnet_julio_epoch_23', alexnet, fc_channels=512)
def alexnet_julio_epoch_24():
    return _model('alexnet_julio_epoch_24', alexnet, fc_channels=512)
def alexnet_julio_epoch_25():
    return _model('alexnet_julio_epoch_25', alexnet, fc_channels=512)
def alexnet_julio_epoch_26():
    return _model('alexnet_julio_epoch_26', alexnet, fc_channels=512)
def alexnet_julio_epoch_27():
    return _model('alexnet_julio_epoch_27', alexnet, fc_channels=512)
def alexnet_julio_epoch_28():
    return _model('alexnet_julio_epoch_28', alexnet, fc_channels=512)
def alexnet_julio_epoch_29():
    return _model('alexnet_julio_epoch_29', alexnet, fc_channels=512)
def alexnet_julio_epoch_30():
    return _model('alexnet_julio_epoch_30', alexnet, fc_channels=512)
def alexnet_julio_epoch_31():
    return _model('alexnet_julio_epoch_31', alexnet, fc_channels=512)
def alexnet_julio_epoch_32():
    return _model('alexnet_julio_epoch_32', alexnet, fc_channels=512)
def alexnet_julio_epoch_33():
    return _model('alexnet_julio_epoch_33', alexnet, fc_channels=512)
def alexnet_julio_epoch_34():
    return _model('alexnet_julio_epoch_34', alexnet, fc_channels=512)
def alexnet_julio_epoch_35():
    return _model('alexnet_julio_epoch_35', alexnet, fc_channels=512)
def alexnet_julio_epoch_36():
    return _model('alexnet_julio_epoch_36', alexnet, fc_channels=512)
def alexnet_julio_epoch_37():
    return _model('alexnet_julio_epoch_37', alexnet, fc_channels=512)
def alexnet_julio_epoch_38():
    return _model('alexnet_julio_epoch_38', alexnet, fc_channels=512)
def alexnet_julio_epoch_39():
    return _model('alexnet_julio_epoch_39', alexnet, fc_channels=512)
def alexnet_julio_epoch_40():
    return _model('alexnet_julio_epoch_40', alexnet, fc_channels=512)
def alexnet_julio_epoch_41():
    return _model('alexnet_julio_epoch_41', alexnet, fc_channels=512)
def alexnet_julio_epoch_42():
    return _model('alexnet_julio_epoch_42', alexnet, fc_channels=512)
def alexnet_julio_epoch_43():
    return _model('alexnet_julio_epoch_43', alexnet, fc_channels=512)
def alexnet_julio_epoch_44():
    return _model('alexnet_julio_epoch_44', alexnet, fc_channels=512)
def alexnet_julio_epoch_45():
    return _model('alexnet_julio_epoch_45', alexnet, fc_channels=512)
def alexnet_julio_epoch_46():
    return _model('alexnet_julio_epoch_46', alexnet, fc_channels=512)
def alexnet_julio_epoch_47():
    return _model('alexnet_julio_epoch_47', alexnet, fc_channels=512)
def alexnet_julio_epoch_48():
    return _model('alexnet_julio_epoch_48', alexnet, fc_channels=512)
def alexnet_julio_epoch_49():
    return _model('alexnet_julio_epoch_49', alexnet, fc_channels=512)
def alexnet_julio_epoch_50():
    return _model('alexnet_julio_epoch_50', alexnet, fc_channels=512)
def alexnet_julio_epoch_51():
    return _model('alexnet_julio_epoch_51', alexnet, fc_channels=512)
def alexnet_julio_epoch_52():
    return _model('alexnet_julio_epoch_52', alexnet, fc_channels=512)
def alexnet_julio_epoch_53():
    return _model('alexnet_julio_epoch_53', alexnet, fc_channels=512)
def alexnet_julio_epoch_54():
    return _model('alexnet_julio_epoch_54', alexnet, fc_channels=512)
def alexnet_julio_epoch_55():
    return _model('alexnet_julio_epoch_55', alexnet, fc_channels=512)
def alexnet_julio_epoch_56():
    return _model('alexnet_julio_epoch_56', alexnet, fc_channels=512)
def alexnet_julio_epoch_57():
    return _model('alexnet_julio_epoch_57', alexnet, fc_channels=512)
def alexnet_julio_epoch_58():
    return _model('alexnet_julio_epoch_58', alexnet, fc_channels=512)
def alexnet_julio_epoch_59():
    return _model('alexnet_julio_epoch_59', alexnet, fc_channels=512)
def alexnet_julio_epoch_60():
    return _model('alexnet_julio_epoch_60', alexnet, fc_channels=512)
def alexnet_julio_epoch_61():
    return _model('alexnet_julio_epoch_61', alexnet, fc_channels=512)
def alexnet_julio_epoch_62():
    return _model('alexnet_julio_epoch_62', alexnet, fc_channels=512)
def alexnet_julio_epoch_63():
    return _model('alexnet_julio_epoch_63', alexnet, fc_channels=512)
def alexnet_julio_epoch_64():
    return _model('alexnet_julio_epoch_64', alexnet, fc_channels=512)
def alexnet_julio_epoch_65():
    return _model('alexnet_julio_epoch_65', alexnet, fc_channels=512)
def alexnet_julio_epoch_66():
    return _model('alexnet_julio_epoch_66', alexnet, fc_channels=512)
def alexnet_julio_epoch_67():
    return _model('alexnet_julio_epoch_67', alexnet, fc_channels=512)
def alexnet_julio_epoch_68():
    return _model('alexnet_julio_epoch_68', alexnet, fc_channels=512)
def alexnet_julio_epoch_69():
    return _model('alexnet_julio_epoch_69', alexnet, fc_channels=512)
def alexnet_julio_epoch_70():
    return _model('alexnet_julio_epoch_70', alexnet, fc_channels=512)
def alexnet_julio_epoch_71():
    return _model('alexnet_julio_epoch_71', alexnet, fc_channels=512)
def alexnet_julio_epoch_72():
    return _model('alexnet_julio_epoch_72', alexnet, fc_channels=512)
def alexnet_julio_epoch_73():
    return _model('alexnet_julio_epoch_73', alexnet, fc_channels=512)
def alexnet_julio_epoch_74():
    return _model('alexnet_julio_epoch_74', alexnet, fc_channels=512)
def alexnet_julio_epoch_75():
    return _model('alexnet_julio_epoch_75', alexnet, fc_channels=512)
def alexnet_julio_epoch_76():
    return _model('alexnet_julio_epoch_76', alexnet, fc_channels=512)
def alexnet_julio_epoch_77():
    return _model('alexnet_julio_epoch_77', alexnet, fc_channels=512)
def alexnet_julio_epoch_78():
    return _model('alexnet_julio_epoch_78', alexnet, fc_channels=512)
def alexnet_julio_epoch_79():
    return _model('alexnet_julio_epoch_79', alexnet, fc_channels=512)
def alexnet_julio_epoch_80():
    return _model('alexnet_julio_epoch_80', alexnet, fc_channels=512)
def alexnet_julio_epoch_81():
    return _model('alexnet_julio_epoch_81', alexnet, fc_channels=512)
def alexnet_julio_epoch_82():
    return _model('alexnet_julio_epoch_82', alexnet, fc_channels=512)
def alexnet_julio_epoch_83():
    return _model('alexnet_julio_epoch_83', alexnet, fc_channels=512)
def alexnet_julio_epoch_84():
    return _model('alexnet_julio_epoch_84', alexnet, fc_channels=512)
def alexnet_julio_epoch_85():
    return _model('alexnet_julio_epoch_85', alexnet, fc_channels=512)
def alexnet_julio_epoch_86():
    return _model('alexnet_julio_epoch_86', alexnet, fc_channels=512)
def alexnet_julio_epoch_87():
    return _model('alexnet_julio_epoch_87', alexnet, fc_channels=512)
def alexnet_julio_epoch_88():
    return _model('alexnet_julio_epoch_88', alexnet, fc_channels=512)
def alexnet_julio_epoch_89():
    return _model('alexnet_julio_epoch_89', alexnet, fc_channels=512)
def alexnet_julio_epoch_90():
    return _model('alexnet_julio_epoch_90', alexnet, fc_channels=512)


def _model(identifier, model_fn, fc_channels, **kwargs):
    model = model_fn(**kwargs)

    state_dict_path = model_paths[identifier]
    # model.load_state_dict(torch.load(state_dict_path))

    in_features = fc_channels*2 if 's2' in identifier else fc_channels

    if 'resnet' in identifier:
        if 'fcn' not in identifier and 'fasterrcnn' not in identifier:
            model.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=model.conv1.out_channels, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
            model.fc = torch.nn.Linear(in_features=in_features, out_features=1000, bias=True)

        if 's2' in identifier:
            def get_features(self, x):
                # See note [TorchScript super()]
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                return x

            def _new_forward(self, x):
                # x = multiscale_forward(self, x, img_sizes=[224, 448], max_split_size=224, output_shape='bchw')
                x = multiscale_forward(self, x, img_sizes=[224, 448], max_split_size=224, output_shape='bchw')

                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)

                return x

            model.get_features = get_features.__get__(model, ResNet)
            model._forward_impl = _new_forward.__get__(model, ResNet)

    elif 'alexnet' in identifier:
        model.classifier[6] = torch.nn.Linear(4096, 1000, bias=True)
    
    if 'random' not in identifier:
        state_dict = torch.load(state_dict_path, map_location='cuda:0')

        if 'julio' in identifier:
            state_dict = state_dict['model']

    if 'random' not in identifier:
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            model = DataParallel(model) # , device_ids=[0]
            model.load_state_dict(state_dict)
            model = model.module

    model.eval()

    return DataParallel(model) # , device_ids=[0]

class ADE20K_ImageNetModel(torch.nn.Module):
    def __init__(self, output_channels):
        super().__init__()
        self.model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-resnet50-ade20k-full").model.pixel_level_module.encoder

        self.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.Flatten(1),
            torch.nn.Linear(in_features=2048, out_features=output_channels)
        )
    
    def forward(self, x):
        output = self.model(x)
        return self.head(output.feature_maps[3])

def _ade20k_resnet50(identifier, num_classes, **kwargs):
    model = ADE20K_ImageNetModel(num_classes)
    
    state_dict_path = model_paths[identifier]
    try:
        model.load_state_dict(torch.load(state_dict_path, map_location='cuda:0'))
    except RuntimeError:
        model = DataParallel(model, device_ids=[0])
        model.load_state_dict(torch.load(state_dict_path, map_location='cuda:0'))
        model = model.module

    model.eval()

    return DataParallel(model, device_ids=[0])

def _transformer_model(identifier, num_classes, **kwargs):
    from pytorch_pretrained_vit import ViT

    if 'transformer_b16_imagenet' in identifier:
        model = ViT('B_16_imagenet1k', pretrained=True)
    else:
        model = ViT('B_16', pretrained=True)

    model.fc = torch.nn.Linear(
        in_features=768, out_features=num_classes, bias=True)

    state_dict_path = model_paths[identifier]

    try:
        model.load_state_dict(torch.load(state_dict_path, map_location='cuda:0'))
    except (RuntimeError, KeyError):
        model = DataParallel(model, device_ids=[0])
        model.load_state_dict(torch.load(state_dict_path, map_location='cuda:0'))
        model = model.module

    model.eval()

    return DataParallel(model, device_ids=[0])

def _vit_model(identifier, num_classes, **kwargs):
    model = torch.hub.load("facebookresearch/swag", model="vit_b16_in1k" if "vit_b_16_imagenet" in identifier else "vit_b16")

    if identifier != "vit_b_16_imagenet":
        model.head = torch.nn.Linear(in_features=768, out_features=num_classes, bias=True)

        # print(model)

        state_dict_path = model_paths[identifier]

        try:
            model.load_state_dict(torch.load(state_dict_path, map_location='cuda:0'))
        except (RuntimeError, KeyError):
            model = DataParallel(model, device_ids=[0])
            model.load_state_dict(torch.load(state_dict_path, map_location='cuda:0'))
            model = model.module

    else:
        print('Running vit_b_16_imagenet')

    model.eval()

    return DataParallel(model, device_ids=[0])

def resnet50_ade20k_scenes_finetuned_layer4_on_subimagenet():
    return _ade20k_resnet50('resnet50_ade20k_scenes_finetuned_layer4_on_subimagenet', num_classes=1000)
def resnet50_ade20k_scenes_finetuned_full_on_subimagenet():
    return _ade20k_resnet50('resnet50_ade20k_scenes_finetuned_full_on_subimagenet', num_classes=1000)
def resnet50_ade20k_scenes_finetuned_on_subimagenet():
    return _ade20k_resnet50('resnet50_ade20k_scenes_finetuned_on_subimagenet', num_classes=1000)

def resnet50_ade20k_objects_finetuned_layer4_on_subimagenet():
    return _ade20k_resnet50('resnet50_ade20k_objects_finetuned_layer4_on_subimagenet', num_classes=1000)
def resnet50_ade20k_objects_finetuned_full_on_subimagenet():
    return _ade20k_resnet50('resnet50_ade20k_objects_finetuned_full_on_subimagenet', num_classes=1000)
def resnet50_ade20k_objects_finetuned_on_subimagenet():
    return _ade20k_resnet50('resnet50_ade20k_objects_finetuned_on_subimagenet', num_classes=1000)


def vit_b_16_imagenet():
    return _vit_model('vit_b_16_imagenet', num_classes=1000)
def vit_b_16_subimagenet():
    return _vit_model('vit_b_16_subimagenet', num_classes=1000)

def vit_b_16_oads_finetuned_on_subimagenet():
    return _vit_model('vit_b_16_oads_finetuned_on_subimagenet', num_classes=1000)
def vit_b_16_oads_finetuned_layer4_on_subimagenet():
    return _vit_model('vit_b_16_oads_finetuned_layer4_on_subimagenet', num_classes=1000)
def vit_b_16_oads_finetuned_full_on_subimagenet():
    return _vit_model('vit_b_16_oads_finetuned_full_on_subimagenet', num_classes=1000)


def resnet50_coco_objects_finetuned_on_subimagenet():
    return _model('resnet50_coco_objects_finetuned_on_subimagenet', resnet50, fc_channels=2048, num_classes=1000)
def resnet50_coco_objects_finetuned_full_on_subimagenet():
    return _model('resnet50_coco_objects_finetuned_full_on_subimagenet', resnet50, fc_channels=2048, num_classes=1000)
def resnet50_coco_objects_finetuned_layer4_on_subimagenet():
    return _model('resnet50_coco_objects_finetuned_layer4_on_subimagenet', resnet50, fc_channels=2048, num_classes=1000)


def fcn_resnet50_coco_oads_finetuned_on_subimagenet():
    return _model('fcn_resnet50_coco_oads_finetuned_on_subimagenet', own_fcn_resnet50, fc_channels=2048, num_classes=1000)
def fcn_resnet50_coco_oads_finetuned_full_on_subimagenet():
    return _model('fcn_resnet50_coco_oads_finetuned_full_on_subimagenet', own_fcn_resnet50, fc_channels=2048, num_classes=1000)
def fcn_resnet50_coco_oads_finetuned_layer4_on_subimagenet():
    return _model('fcn_resnet50_coco_oads_finetuned_layer4_on_subimagenet', own_fcn_resnet50, fc_channels=2048, num_classes=1000)

def resnet50_s2_imagenet():
    return _model('resnet50_s2_imagenet', resnet50, fc_channels=2048)

def fasterrcnn_resnet50_fpn_coco_finetuned_on_subimagenet():
    return _model('fasterrcnn_resnet50_fpn_coco_finetuned_on_subimagenet', fasterrcnn_resnet50_fpn_coco, fc_channels=2048, num_classes=1000)
def fasterrcnn_resnet50_fpn_coco_finetuned_full_on_subimagenet():
    return _model('fasterrcnn_resnet50_fpn_coco_finetuned_full_on_subimagenet', fasterrcnn_resnet50_fpn_coco, fc_channels=2048, num_classes=1000)
def fasterrcnn_resnet50_fpn_coco_finetuned_layer4_on_subimagenet():
    return _model('fasterrcnn_resnet50_fpn_coco_finetuned_layer4_on_subimagenet', fasterrcnn_resnet50_fpn_coco, fc_channels=2048, num_classes=1000)

def fcn_resnet50_coco_finetuned_on_subimagenet():
    return _model('fcn_resnet50_coco_finetuned_on_subimagenet', own_fcn_resnet50, fc_channels=2048, num_classes=1000)
def fcn_resnet50_coco_finetuned_full_on_subimagenet():
    return _model('fcn_resnet50_coco_finetuned_full_on_subimagenet', own_fcn_resnet50, fc_channels=2048, num_classes=1000)
def fcn_resnet50_coco_finetuned_layer4_on_subimagenet():
    return _model('fcn_resnet50_coco_finetuned_layer4_on_subimagenet', own_fcn_resnet50, fc_channels=2048, num_classes=1000)

def resnet50_places365_finetuned_on_subimagenet():
    return _model('resnet50_places365_finetuned_on_subimagenet', resnet50, fc_channels=2048)
def resnet50_places365_finetuned_full_on_subimagenet():
    return _model('resnet50_places365_finetuned_full_on_subimagenet', resnet50, fc_channels=2048)
def resnet50_places365_finetuned_layer4_on_subimagenet():
    return _model('resnet50_places365_finetuned_layer4_on_subimagenet', resnet50, fc_channels=2048)

def resnet50_imagenet_no_crop_224x224():
    return _model('resnet50_imagenet_no_crop_224x224', resnet50, fc_channels=2048)
def resnet50_imagenet_no_crop_400x400():
    return _model('resnet50_imagenet_no_crop_400x400', resnet50, fc_channels=2048)

def resnet50_imagenet_350x350_crop_224x224():
    return _model('resnet50_imagenet_350x350_crop_224x224', resnet50, fc_channels=2048)

def resnet50_imagenet_600x600_crop_400x400():
    return _model('resnet50_imagenet_600x600_crop_400x400', resnet50, fc_channels=2048)

def resnet50_imagenet_400x400():
    return _model('resnet50_imagenet_400x400', resnet50, fc_channels=2048)
def resnet50_imagenet_500x500():
    return _model('resnet50_imagenet_500x500', resnet50, fc_channels=2048)
def resnet50_imagenet_600x600():
    return _model('resnet50_imagenet_600x600', resnet50, fc_channels=2048)
    

def resnet50_imagenet_10x10_to_224x224():
    return _model('resnet50_imagenet_10x10_to_224x224', resnet50, fc_channels=2048)
def resnet50_imagenet_30x30_to_224x224():
    return _model('resnet50_imagenet_30x30_to_224x224', resnet50, fc_channels=2048)
def resnet50_imagenet_80x80_to_224x224():
    return _model('resnet50_imagenet_80x80_to_224x224', resnet50, fc_channels=2048)
def resnet50_imagenet_112x112_to_224x224():
    return _model('resnet50_imagenet_112x112_to_224x224', resnet50, fc_channels=2048)

def resnet50_imagenet_400x400_low_res():
    return _model('resnet50_imagenet_400x400_low_res', resnet50, fc_channels=2048)

def resnet50_oads_normalized_rgb_jpeg_finetuned_on_imagenet():
    return _model('resnet50_oads_normalized_rgb_jpeg_finetuned_on_imagenet', resnet50, fc_channels=2048)
    
def resnet50_oads_normalized_rgb_jpeg_finetuned_full_on_imagenet():
    return _model('resnet50_oads_normalized_rgb_jpeg_finetuned_full_on_imagenet', resnet50, fc_channels=2048)
    
def resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_on_imagenet():
    return _model('resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_on_imagenet', resnet50, fc_channels=2048)
    
def resnet50_oads_normalized_rgb_finetuned_on_subimagenet():
    return _model('resnet50_oads_normalized_rgb_finetuned_on_subimagenet', resnet50, fc_channels=2048)
    
def resnet50_oads_normalized_rgb_jpeg_finetuned_on_subimagenet():
    return _model('resnet50_oads_normalized_rgb_jpeg_finetuned_on_subimagenet', resnet50, fc_channels=2048)
    
def resnet50_oads_normalized_rgb_jpeg_finetuned_imagenetsize_on_subimagenet():
    return _model('resnet50_oads_normalized_rgb_jpeg_finetuned_imagenetsize_on_subimagenet', resnet50, fc_channels=2048)
    
def resnet50_oads_normalized_rgb_finetuned_imagenetsize_on_subimagenet():
    return _model('resnet50_oads_normalized_rgb_finetuned_imagenetsize_on_subimagenet', resnet50, fc_channels=2048)
    
def resnet50_oads_normalized_rgb_finetuned_full_on_subimagenet():
    return _model('resnet50_oads_normalized_rgb_finetuned_full_on_subimagenet', resnet50, fc_channels=2048)
    
def resnet50_oads_normalized_rgb_jpeg_finetuned_full_on_subimagenet():
    return _model('resnet50_oads_normalized_rgb_jpeg_finetuned_full_on_subimagenet', resnet50, fc_channels=2048)
    
def resnet50_oads_normalized_rgb_jpeg_finetuned_full_imagenetsize_on_subimagenet():
    return _model('resnet50_oads_normalized_rgb_jpeg_finetuned_full_imagenetsize_on_subimagenet', resnet50, fc_channels=2048)
    
def resnet50_oads_normalized_rgb_finetuned_full_imagenetsize_on_subimagenet():
    return _model('resnet50_oads_normalized_rgb_finetuned_full_imagenetsize_on_subimagenet', resnet50, fc_channels=2048)
    
def resnet50_oads_normalized_rgb_finetuned_layer4_on_subimagenet():
    return _model('resnet50_oads_normalized_rgb_finetuned_layer4_on_subimagenet', resnet50, fc_channels=2048)
    
def resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_on_subimagenet():
    return _model('resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_on_subimagenet', resnet50, fc_channels=2048)
    
def resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_imagenetsize_on_subimagenet():
    return _model('resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_imagenetsize_on_subimagenet', resnet50, fc_channels=2048)
    
def resnet50_oads_normalized_rgb_finetuned_layer4_imagenetsize_on_subimagenet():
    return _model('resnet50_oads_normalized_rgb_finetuned_layer4_imagenetsize_on_subimagenet', resnet50, fc_channels=2048)
    

def resnet18_oads_normalized_rgb_finetuned_layer4_on_imagenet():
    return _model('resnet18_oads_normalized_rgb_finetuned_layer4_on_imagenet', resnet18, fc_channels=512)

def resnet18_oads_normalized_rgb_jpeg_finetuned_layer4_imagenetsize_on_imagenet():
    return _model('resnet18_oads_normalized_rgb_jpeg_finetuned_layer4_imagenetsize_on_imagenet', resnet18, fc_channels=512)

def resnet18_oads_rgb_jpeg_finetuned_layer4_imagenetsize_on_imagenet():
    return _model('resnet18_oads_rgb_jpeg_finetuned_layer4_imagenetsize_on_imagenet', resnet18, fc_channels=512)

def resnet18_oads_rgb_finetuned_on_imagenet():
    return _model('resnet18_oads_rgb_finetuned_on_imagenet', resnet18, fc_channels=512)

def resnet18_oads_coc_finetuned_on_imagenet():
    return _model('resnet18_oads_coc_finetuned_on_imagenet', resnet18, fc_channels=512)

def resnet18_oads_rgb_jpeg_finetuned_on_imagenet():
    return _model('resnet18_oads_rgb_jpeg_finetuned_on_imagenet', resnet18, fc_channels=512)

def resnet18_oads_coc_jpeg_finetuned_on_imagenet():
    return _model('resnet18_oads_coc_jpeg_finetuned_on_imagenet', resnet18, fc_channels=512)

def resnet18_oads_rgb_finetuned_full_on_imagenet():
    return _model('resnet18_oads_rgb_finetuned_full_on_imagenet', resnet18, fc_channels=512)

def resnet18_oads_coc_finetuned_full_on_imagenet():
    return _model('resnet18_oads_coc_finetuned_full_on_imagenet', resnet18, fc_channels=512)

def resnet18_oads_rgb_jpeg_finetuned_full_on_imagenet():
    return _model('resnet18_oads_rgb_jpeg_finetuned_full_on_imagenet', resnet18, fc_channels=512)

def resnet18_oads_coc_jpeg_finetuned_full_on_imagenet():
    return _model('resnet18_oads_coc_jpeg_finetuned_full_on_imagenet', resnet18, fc_channels=512)

def resnet18_oads_rgb_finetuned_layer4_on_imagenet():
    return _model('resnet18_oads_rgb_finetuned_layer4_on_imagenet', resnet18, fc_channels=512)

def resnet18_oads_coc_finetuned_layer4_on_imagenet():
    return _model('resnet18_oads_coc_finetuned_layer4_on_imagenet', resnet18, fc_channels=512)

def resnet18_oads_rgb_jpeg_finetuned_layer4_on_imagenet():
    return _model('resnet18_oads_rgb_jpeg_finetuned_layer4_on_imagenet', resnet18, fc_channels=512)

def resnet18_oads_coc_jpeg_finetuned_layer4_on_imagenet():
    return _model('resnet18_oads_coc_jpeg_finetuned_layer4_on_imagenet', resnet18, fc_channels=512)

def resnet50_oads_rgb_finetuned_on_imagenet():
    return _model('resnet50_oads_rgb_finetuned_on_imagenet', resnet50, fc_channels=2048)

def resnet50_oads_coc_finetuned_on_imagenet():
    return _model('resnet50_oads_coc_finetuned_on_imagenet', resnet50, fc_channels=2048)

def resnet50_oads_rgb_jpeg_finetuned_on_imagenet():
    return _model('resnet50_oads_rgb_jpeg_finetuned_on_imagenet', resnet50, fc_channels=2048)

def resnet50_oads_coc_jpeg_finetuned_on_imagenet():
    return _model('resnet50_oads_coc_jpeg_finetuned_on_imagenet', resnet50, fc_channels=2048)

def resnet50_oads_rgb_finetuned_full_on_imagenet():
    return _model('resnet50_oads_rgb_finetuned_full_on_imagenet', resnet50, fc_channels=2048)

def resnet50_oads_coc_finetuned_full_on_imagenet():
    return _model('resnet50_oads_coc_finetuned_full_on_imagenet', resnet50, fc_channels=2048)

def resnet50_oads_rgb_jpeg_finetuned_full_on_imagenet():
    return _model('resnet50_oads_rgb_jpeg_finetuned_full_on_imagenet', resnet50, fc_channels=2048)

def resnet50_oads_coc_jpeg_finetuned_full_on_imagenet():
    return _model('resnet50_oads_coc_jpeg_finetuned_full_on_imagenet', resnet50, fc_channels=2048)

def resnet50_oads_rgb_finetuned_layer4_on_imagenet():
    return _model('resnet50_oads_rgb_finetuned_layer4_on_imagenet', resnet50, fc_channels=2048)

def resnet50_oads_coc_finetuned_layer4_on_imagenet():
    return _model('resnet50_oads_coc_finetuned_layer4_on_imagenet', resnet50, fc_channels=2048)

def resnet50_oads_rgb_jpeg_finetuned_layer4_on_imagenet():
    return _model('resnet50_oads_rgb_jpeg_finetuned_layer4_on_imagenet', resnet50, fc_channels=2048)

def resnet50_oads_coc_jpeg_finetuned_layer4_on_imagenet():
    return _model('resnet50_oads_coc_jpeg_finetuned_layer4_on_imagenet', resnet50, fc_channels=2048)

def resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_imagenetsize_on_imagenet():
    return _model('resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_imagenetsize_on_imagenet', resnet50, fc_channels=2048)
def resnet50_oads_normalized_rgb_jpeg_40_finetuned_layer4_imagenetsize_on_imagenet():
    return _model('resnet50_oads_normalized_rgb_jpeg_40_finetuned_layer4_imagenetsize_on_imagenet', resnet50, fc_channels=2048)
def resnet50_oads_normalized_rgb_jpeg_60_finetuned_layer4_imagenetsize_on_imagenet():
    return _model('resnet50_oads_normalized_rgb_jpeg_60_finetuned_layer4_imagenetsize_on_imagenet', resnet50, fc_channels=2048)
def resnet50_oads_normalized_rgb_jpeg_90_finetuned_layer4_imagenetsize_on_imagenet():
    return _model('resnet50_oads_normalized_rgb_jpeg_90_finetuned_layer4_imagenetsize_on_imagenet', resnet50, fc_channels=2048)
def resnet50_oads_normalized_rgb_finetuned_layer4_on_imagenet():
    return _model('resnet50_oads_normalized_rgb_finetuned_layer4_on_imagenet', resnet50, fc_channels=2048)

def resnet50_oads_normalized_rgb_jpeg_40_finetuned_full_imagenetsize_on_imagenet():
    return _model('resnet50_oads_normalized_rgb_jpeg_40_finetuned_full_imagenetsize_on_imagenet', resnet50, fc_channels=2048)
def resnet50_oads_normalized_rgb_jpeg_60_finetuned_full_imagenetsize_on_imagenet():
    return _model('resnet50_oads_normalized_rgb_jpeg_60_finetuned_full_imagenetsize_on_imagenet', resnet50, fc_channels=2048)
def resnet50_oads_normalized_rgb_jpeg_90_finetuned_full_imagenetsize_on_imagenet():
    return _model('resnet50_oads_normalized_rgb_jpeg_90_finetuned_full_imagenetsize_on_imagenet', resnet50, fc_channels=2048)
def resnet50_oads_normalized_rgb_finetuned_full_on_imagenet():
    return _model('resnet50_oads_normalized_rgb_finetuned_full_on_imagenet', resnet50, fc_channels=2048)

def resnet50_oads_normalized_rgb_finetuned_full_imagenetsize_on_imagenet():
    return _model('resnet50_oads_normalized_rgb_finetuned_full_imagenetsize_on_imagenet', resnet50, fc_channels=2048)

def resnet50_oads_normalized_rgb_finetuned_layer4_imagenetsize_on_imagenet():
    return _model('resnet50_oads_normalized_rgb_finetuned_layer4_imagenetsize_on_imagenet', resnet50, fc_channels=2048)

def resnet50_oads_normalized_rgb_jpeg_finetuned_imagenetsize_on_imagenet():
    return _model('resnet50_oads_normalized_rgb_jpeg_finetuned_imagenetsize_on_imagenet', resnet50, fc_channels=2048)

def resnet50_oads_normalized_rgb_finetuned_on_imagenet():
    return _model('resnet50_oads_normalized_rgb_finetuned_on_imagenet', resnet50, fc_channels=2048)

def resnet50_oads_normalized_rgb_finetuned_imagenetsize_on_imagenet():
    return _model('resnet50_oads_normalized_rgb_finetuned_imagenetsize_on_imagenet', resnet50, fc_channels=2048)

def resnet50_imagenet_subclasses():
    return _model('resnet50_imagenet_subclasses', resnet50, fc_channels=2048)

def resnet50_imagenet_112x112():
    return _model('resnet50_imagenet_112x112', resnet50, fc_channels=2048)

def resnet50_subimagenet_bounding_boxes():
    return _model('resnet50_subimagenet_bounding_boxes', resnet50, fc_channels=2048)
    
def resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100():
    return _model('resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100', resnet50, fc_channels=2048)
def resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p70():
    return _model('resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p70', resnet50, fc_channels=2048)

def resnet50_imagenet_grayscale():
    model = network_Gray_ResNet.resnet50()
    state_dict = torch.load(model_paths['resnet50_imagenet_grayscale'])
    model.load_state_dict(state_dict=state_dict)
    # return _model('resnet50_imagenet_grayscale', resnet50, fc_channels=2048)
    model.eval()
    
    return model