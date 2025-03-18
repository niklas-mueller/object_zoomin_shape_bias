import os
import torch
from torch.nn.parallel import DataParallel
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torchvision
from torch import nn
from transformers import MaskFormerForInstanceSegmentation
# import sys

# os.environ['MODELVSHUMANDIR'] = f'{os.path.dirname(os.path.abspath(__file__))}/../model-vs-human'

# sys.path.append(f'../model-vs-human')

class ADE20K_ImageNetModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-resnet50-ade20k-full").model.pixel_level_module.encoder

        self.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.Flatten(1),
            torch.nn.Linear(in_features=2048, out_features=num_classes)
        )
    
    def forward(self, x):
        output = self.model(x)
        return self.head(output.feature_maps[3])

def get_model_instance_segmentation(num_classes, pretrained:bool=False):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    # get number of input features for the classifier
    if not pretrained:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def fasterrcnn_resnet50_fpn_coco(num_classes, faster_ccn_model_path=None):
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
    if faster_ccn_model_path is None:
        faster_ccn_model_path = '/home/nmuller/projects/fmg_storage/trained_models/coco_results/27-02-24-173725/best_model_Tue_Feb_27_173803_2024.pth'

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


def load_model(model_name, model_fn, in_features, state_dict_path, num_classes, **kwargs):
    try:
        model = model_fn(**kwargs)
    except TypeError:
        model = model_fn(num_classes, **kwargs)

    if 'resnet' in model_name:
        if 'fcn' not in model_name and 'fasterrcnn' not in model_name and 'ade20k_scenes' not in model_name:
            model.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=model.conv1.out_channels, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
            model.fc = torch.nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    if type(state_dict_path) is str:
        state_dict = torch.load(state_dict_path, map_location='cuda:0')
    else:
        state_dict = state_dict_path


    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # print(e)
        model = DataParallel(model) # , device_ids=[0]
        model.load_state_dict(state_dict)
        model = model.module

    model.eval()

    return DataParallel(model)



# def load_model(model_name, n_input_channels, n_output_channels, device, weights:"None|dict"=None):

#     if 'fcn_resnet50' in model_name:
#         model = own_fcn_resnet50(num_classes=n_output_channels)

#     elif 'fasterrcnn_resnet50' in model_name:
#         model = get_faster_rcnn_resnet50_fpn_coco()

#     elif 'resnet50' in model_name:
#         model = resnet50()

#         in_features = 2048

#         # if weights is None:
#         model.conv1 = torch.nn.Conv2d(in_channels=n_input_channels, out_channels=model.conv1.out_channels, kernel_size=(
#             7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         model.fc = torch.nn.Linear(
#             in_features=in_features, out_features=n_output_channels, bias=True)

#     else:
#         raise Exception(f'Model {model_name} not found.')
    
#     if weights is not None:
#         if type(weights) is str:
#             try:
#                 model.load_state_dict(torch.load(weights))
#             except:
#                 model = torch.nn.DataParallel(model)
#                 model.load_state_dict(torch.load(weights))
#                 model = model.module
#         elif type(weights) is dict:
#             model.load_state_dict(weights)

#     model = model.to(device)

    
#     return model