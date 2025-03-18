import os
import torch
import sys
sys.path.append('/home/niklas/projects/ccnn')
from models.resnet import ResNet_image
from omegaconf import OmegaConf
from result_manager.result_manager import ResultManager

__all__ = [
    'flexconv_resnet_rgb_imagenet',
]

model_paths = {
    'flexconv_resnet_rgb_imagenet': '/home/niklas/projects/imagenet_results/10-07-23-13:42:00/best_model_10-07-23-13:42:00.pth',
    'flexconv_resnet_rgb_oads': '/home/niklas/projects/oads_results/flex_resnet50/rgb/2023-07-05-10:28:28/best_model_05-07-23-10:28:39.pth',
}

def _model(identifier, in_channels=3, out_channels=21, **kwargs):
    # model = model_fn(pretrained=False)

    state_dict_path = model_paths[identifier]

    result_path, model_ident = state_dict_path.split('best_model_')
    model_ident = model_ident.split('.pth')[0]

    result_manager = ResultManager(root=result_path)

    # info = result_manager.load_result(filename=f'fitting_description_{model_ident}.yaml')
    # cfg = OmegaConf.create(info['cfg'])
    try:
        cfg = OmegaConf.load(f'{os.path.expanduser("~")}/projects/ccnn/cfg/oads_config.yaml')
    except:
        cfg = OmegaConf.load(f'{os.path.expanduser("~")}/projects/oads_flexconv/cfg/oads_config.yaml')

    cfg.net.data_dim = 2
    cfg.net.no_hidden = 140

    cfg.kernel.size = 33
    cfg.kernel.no_hidden = 32
    cfg.kernel.no_layers = 3

    cfg.conv.padding = "same"
    cfg.conv.stride = 1

    model = ResNet_image(in_channels=in_channels,
    out_channels = out_channels,
    net_cfg=cfg.net,
    kernel_cfg= cfg.kernel,
    conv_cfg= cfg.conv,
    mask_cfg= cfg.mask,)

    try:
        model.load_state_dict(torch.load(state_dict_path, map_location='cuda:0'))
    except RuntimeError:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(state_dict_path, map_location='cuda:0'))
        model = model.module

    model.eval()

    return torch.nn.DataParallel(model)

def flexconv_resnet_rgb_imagenet():
    return _model('flexconv_resnet_rgb_imagenet', in_channels=3, out_channels=1000)

def flexconv_resnet_rgb_oads():
    return _model('flexconv_resnet_rgb_oads', in_channels=3, out_channels=21)