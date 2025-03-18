#!/usr/bin/env python3
import torch
from torchvision.models import resnet18, resnet50
from torch.nn.parallel import DataParallel
from ..registry import register_model
from ..wrappers.pytorch import PytorchModel, PyContrastPytorchModel, ClipPytorchModel, \
    ViTPytorchModel, EfficientNetPytorchModel, SwagPytorchModel

_PYTORCH_IMAGE_MODELS = "rwightman/pytorch-image-models"

_EFFICIENTNET_MODELS = "rwightman/gen-efficientnet-pytorch"


def model_pytorch(model_name, *args):
    import torchvision.models as zoomodels
    model = zoomodels.__dict__[model_name](pretrained=True)
    model = DataParallel(model)
    return PytorchModel(model, model_name, *args)

@register_model("pytorch")
def flexconv_resnet_rgb_imagenet(model_name, *args):
    from .flexconv_resnet.flexconv_resnet import flexconv_resnet_rgb_imagenet
    model = flexconv_resnet_rgb_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_imagenet_finetuned_on_all_boxes_p50_zoom_0(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_imagenet_finetuned_on_all_boxes_p50_zoom_0
    model = resnet50_imagenet_finetuned_on_all_boxes_p50_zoom_0()
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_zoom_150_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_zoom_150_finetuned_full_on_subimagenet
    model = resnet50_oads_zoom_150_finetuned_full_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
    
@register_model("pytorch")
def resnet50_oads_zoom_80_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_zoom_80_finetuned_full_on_subimagenet
    model = resnet50_oads_zoom_80_finetuned_full_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)





@register_model("pytorch")
def transformer_b16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_150(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import transformer_b16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_150
    model = transformer_b16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_150()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return ViTPytorchModel(model=model, model_name=model_name, img_size=(384, 384), *args)

@register_model("pytorch")
def vit_b_16_ade20k_objects_zoom_0_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_ade20k_objects_zoom_0_finetuned_on_subimagenet
    model = vit_b_16_ade20k_objects_zoom_0_finetuned_on_subimagenet()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return SwagPytorchModel(model=model, model_name=model_name, input_size=(224, 224), *args)
@register_model("pytorch")
def vit_b_16_ade20k_objects_zoom_0_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_ade20k_objects_zoom_0_finetuned_full_on_subimagenet
    model = vit_b_16_ade20k_objects_zoom_0_finetuned_full_on_subimagenet()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return SwagPytorchModel(model=model, model_name=model_name, input_size=(224, 224), *args)
@register_model("pytorch")
def vit_b_16_ade20k_objects_zoom_0_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_ade20k_objects_zoom_0_finetuned_layer4_on_subimagenet
    model = vit_b_16_ade20k_objects_zoom_0_finetuned_layer4_on_subimagenet()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return SwagPytorchModel(model=model, model_name=model_name, input_size=(224, 224), *args)
@register_model("pytorch")
def vit_b_16_ade20k_objects_zoom_80_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_ade20k_objects_zoom_80_finetuned_on_subimagenet
    model = vit_b_16_ade20k_objects_zoom_80_finetuned_on_subimagenet()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return SwagPytorchModel(model=model, model_name=model_name, input_size=(224, 224), *args)
@register_model("pytorch")
def vit_b_16_ade20k_objects_zoom_80_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_ade20k_objects_zoom_80_finetuned_full_on_subimagenet
    model = vit_b_16_ade20k_objects_zoom_80_finetuned_full_on_subimagenet()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return SwagPytorchModel(model=model, model_name=model_name, input_size=(224, 224), *args)
@register_model("pytorch")
def vit_b_16_ade20k_objects_zoom_80_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_ade20k_objects_zoom_80_finetuned_layer4_on_subimagenet
    model = vit_b_16_ade20k_objects_zoom_80_finetuned_layer4_on_subimagenet()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return SwagPytorchModel(model=model, model_name=model_name, input_size=(224, 224), *args)
@register_model("pytorch")
def vit_b_16_ade20k_objects_zoom_150_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_ade20k_objects_zoom_150_finetuned_on_subimagenet
    model = vit_b_16_ade20k_objects_zoom_150_finetuned_on_subimagenet()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return SwagPytorchModel(model=model, model_name=model_name, input_size=(224, 224), *args)
@register_model("pytorch")
def vit_b_16_ade20k_objects_zoom_150_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_ade20k_objects_zoom_150_finetuned_full_on_subimagenet
    model = vit_b_16_ade20k_objects_zoom_150_finetuned_full_on_subimagenet()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return SwagPytorchModel(model=model, model_name=model_name, input_size=(224, 224), *args)
@register_model("pytorch")
def vit_b_16_ade20k_objects_zoom_150_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_ade20k_objects_zoom_150_finetuned_layer4_on_subimagenet
    model = vit_b_16_ade20k_objects_zoom_150_finetuned_layer4_on_subimagenet()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return SwagPytorchModel(model=model, model_name=model_name, input_size=(224, 224), *args)


@register_model("pytorch")
def alexnet_ade20k_objects_zoom_0_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_ade20k_objects_zoom_0_finetuned_full_on_subimagenet
    model = alexnet_ade20k_objects_zoom_0_finetuned_full_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_ade20k_objects_zoom_0_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_ade20k_objects_zoom_0_finetuned_layer4_on_subimagenet
    model = alexnet_ade20k_objects_zoom_0_finetuned_layer4_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_ade20k_objects_zoom_80_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_ade20k_objects_zoom_80_finetuned_on_subimagenet
    model = alexnet_ade20k_objects_zoom_80_finetuned_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_ade20k_objects_zoom_80_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_ade20k_objects_zoom_80_finetuned_full_on_subimagenet
    model = alexnet_ade20k_objects_zoom_80_finetuned_full_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_ade20k_objects_zoom_80_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_ade20k_objects_zoom_80_finetuned_layer4_on_subimagenet
    model = alexnet_ade20k_objects_zoom_80_finetuned_layer4_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_ade20k_objects_zoom_150_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_ade20k_objects_zoom_150_finetuned_on_subimagenet
    model = alexnet_ade20k_objects_zoom_150_finetuned_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_ade20k_objects_zoom_150_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_ade20k_objects_zoom_150_finetuned_full_on_subimagenet
    model = alexnet_ade20k_objects_zoom_150_finetuned_full_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_ade20k_objects_zoom_0_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_ade20k_objects_zoom_0_finetuned_on_subimagenet
    model = alexnet_ade20k_objects_zoom_0_finetuned_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_ade20k_objects_zoom_150_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_ade20k_objects_zoom_150_finetuned_layer4_on_subimagenet
    model = alexnet_ade20k_objects_zoom_150_finetuned_layer4_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def vit_b_16_coco_objects_zoom_0_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_coco_objects_zoom_0_finetuned_on_subimagenet
    model = vit_b_16_coco_objects_zoom_0_finetuned_on_subimagenet()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return SwagPytorchModel(model=model, model_name=model_name, input_size=(224, 224), *args)
@register_model("pytorch")
def vit_b_16_coco_objects_zoom_0_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_coco_objects_zoom_0_finetuned_full_on_subimagenet
    model = vit_b_16_coco_objects_zoom_0_finetuned_full_on_subimagenet()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return SwagPytorchModel(model=model, model_name=model_name, input_size=(224, 224), *args)
@register_model("pytorch")
def vit_b_16_coco_objects_zoom_0_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_coco_objects_zoom_0_finetuned_layer4_on_subimagenet
    model = vit_b_16_coco_objects_zoom_0_finetuned_layer4_on_subimagenet()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return SwagPytorchModel(model=model, model_name=model_name, input_size=(224, 224), *args)
@register_model("pytorch")
def vit_b_16_coco_objects_zoom_80_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_coco_objects_zoom_80_finetuned_on_subimagenet
    model = vit_b_16_coco_objects_zoom_80_finetuned_on_subimagenet()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return SwagPytorchModel(model=model, model_name=model_name, input_size=(224, 224), *args)
@register_model("pytorch")
def vit_b_16_coco_objects_zoom_80_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_coco_objects_zoom_80_finetuned_full_on_subimagenet
    model = vit_b_16_coco_objects_zoom_80_finetuned_full_on_subimagenet()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return SwagPytorchModel(model=model, model_name=model_name, input_size=(224, 224), *args)
@register_model("pytorch")
def vit_b_16_coco_objects_zoom_80_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_coco_objects_zoom_80_finetuned_layer4_on_subimagenet
    model = vit_b_16_coco_objects_zoom_80_finetuned_layer4_on_subimagenet()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return SwagPytorchModel(model=model, model_name=model_name, input_size=(224, 224), *args)
@register_model("pytorch")
def vit_b_16_coco_objects_zoom_150_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_coco_objects_zoom_150_finetuned_on_subimagenet
    model = vit_b_16_coco_objects_zoom_150_finetuned_on_subimagenet()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return SwagPytorchModel(model=model, model_name=model_name, input_size=(224, 224), *args)
@register_model("pytorch")
def vit_b_16_coco_objects_zoom_150_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_coco_objects_zoom_150_finetuned_full_on_subimagenet
    model = vit_b_16_coco_objects_zoom_150_finetuned_full_on_subimagenet()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return SwagPytorchModel(model=model, model_name=model_name, input_size=(224, 224), *args)
@register_model("pytorch")
def vit_b_16_coco_objects_zoom_150_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_coco_objects_zoom_150_finetuned_layer4_on_subimagenet
    model = vit_b_16_coco_objects_zoom_150_finetuned_layer4_on_subimagenet()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return SwagPytorchModel(model=model, model_name=model_name, input_size=(224, 224), *args)


@register_model("pytorch")
def alexnet_coco_objects_zoom_0_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_coco_objects_zoom_0_finetuned_full_on_subimagenet
    model = alexnet_coco_objects_zoom_0_finetuned_full_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_coco_objects_zoom_0_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_coco_objects_zoom_0_finetuned_layer4_on_subimagenet
    model = alexnet_coco_objects_zoom_0_finetuned_layer4_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_coco_objects_zoom_80_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_coco_objects_zoom_80_finetuned_on_subimagenet
    model = alexnet_coco_objects_zoom_80_finetuned_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_coco_objects_zoom_80_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_coco_objects_zoom_80_finetuned_full_on_subimagenet
    model = alexnet_coco_objects_zoom_80_finetuned_full_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_coco_objects_zoom_80_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_coco_objects_zoom_80_finetuned_layer4_on_subimagenet
    model = alexnet_coco_objects_zoom_80_finetuned_layer4_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_coco_objects_zoom_150_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_coco_objects_zoom_150_finetuned_on_subimagenet
    model = alexnet_coco_objects_zoom_150_finetuned_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_coco_objects_zoom_150_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_coco_objects_zoom_150_finetuned_full_on_subimagenet
    model = alexnet_coco_objects_zoom_150_finetuned_full_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_coco_objects_zoom_0_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_coco_objects_zoom_0_finetuned_on_subimagenet
    model = alexnet_coco_objects_zoom_0_finetuned_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_coco_objects_zoom_150_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_coco_objects_zoom_150_finetuned_layer4_on_subimagenet
    model = alexnet_coco_objects_zoom_150_finetuned_layer4_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)


@register_model("pytorch")
def alexnet_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_150(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_150
    model = alexnet_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_150()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_80(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_80
    model = alexnet_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_80()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_0(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_0
    model = alexnet_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_0()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_150(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_150
    model = resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_150()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_80(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_80
    model = resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_80()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_0(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_0
    model = resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_0()
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_0(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_0
    model = vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_0()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return SwagPytorchModel(model=model, model_name=model_name, input_size=(384, 384), *args)
@register_model("pytorch")
def vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_80(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_80
    model = vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_80()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return SwagPytorchModel(model=model, model_name=model_name, input_size=(384, 384), *args)
@register_model("pytorch")
def vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_150(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_150
    model = vit_b_16_imagenet_finetuned_on_subimagenet_bounding_boxes_384x384_p100_zoom_150()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return SwagPytorchModel(model=model, model_name=model_name, input_size=(384, 384), *args)

@register_model("pytorch")
def vit_b_16_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_imagenet
    model = vit_b_16_imagenet()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return SwagPytorchModel(model=model, model_name=model_name, input_size=(384, 384), *args)
@register_model("pytorch")
def vit_b_16_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_subimagenet
    model = vit_b_16_subimagenet()
    # return PytorchModel(model=model, model_name=model_name, *args)
    return SwagPytorchModel(model=model, model_name=model_name, input_size=(224, 224), *args)


@register_model("pytorch")
def alexnet_subimagenet_bounding_boxes_p100_zoom_0(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_subimagenet_bounding_boxes_p100_zoom_0
    model = alexnet_subimagenet_bounding_boxes_p100_zoom_0()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_subimagenet_bounding_boxes_p100_zoom_80(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_subimagenet_bounding_boxes_p100_zoom_80
    model = alexnet_subimagenet_bounding_boxes_p100_zoom_80()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_subimagenet_bounding_boxes_p100_zoom_150(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_subimagenet_bounding_boxes_p100_zoom_150
    model = alexnet_subimagenet_bounding_boxes_p100_zoom_150()
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def vit_b_16_subimagenet_bounding_boxes_p100_zoom_0(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_subimagenet_bounding_boxes_p100_zoom_0
    model = vit_b_16_subimagenet_bounding_boxes_p100_zoom_0()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def vit_b_16_subimagenet_bounding_boxes_p100_zoom_80(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_subimagenet_bounding_boxes_p100_zoom_80
    model = vit_b_16_subimagenet_bounding_boxes_p100_zoom_80()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def vit_b_16_subimagenet_bounding_boxes_p100_zoom_150(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_subimagenet_bounding_boxes_p100_zoom_150
    model = vit_b_16_subimagenet_bounding_boxes_p100_zoom_150()
    return PytorchModel(model=model, model_name=model_name, *args)


@register_model("pytorch")
def resnet50_subimagenet_bounding_boxes_p100_zoom_80(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_subimagenet_bounding_boxes_p100_zoom_80
    model = resnet50_subimagenet_bounding_boxes_p100_zoom_80()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_subimagenet_bounding_boxes_p100_zoom_150(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_subimagenet_bounding_boxes_p100_zoom_150
    model = resnet50_subimagenet_bounding_boxes_p100_zoom_150()
    return PytorchModel(model=model, model_name=model_name, *args)


@register_model("pytorch")
def resnet50_coco_objects_zoom_80_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_coco_objects_zoom_80_finetuned_on_subimagenet
    model = resnet50_coco_objects_zoom_80_finetuned_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_coco_objects_zoom_80_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_coco_objects_zoom_80_finetuned_full_on_subimagenet
    model = resnet50_coco_objects_zoom_80_finetuned_full_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_coco_objects_zoom_80_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_coco_objects_zoom_80_finetuned_layer4_on_subimagenet
    model = resnet50_coco_objects_zoom_80_finetuned_layer4_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_coco_objects_zoom_150_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_coco_objects_zoom_150_finetuned_on_subimagenet
    model = resnet50_coco_objects_zoom_150_finetuned_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_coco_objects_zoom_150_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_coco_objects_zoom_150_finetuned_full_on_subimagenet
    model = resnet50_coco_objects_zoom_150_finetuned_full_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_coco_objects_zoom_150_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_coco_objects_zoom_150_finetuned_layer4_on_subimagenet
    model = resnet50_coco_objects_zoom_150_finetuned_layer4_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_subimagenet_bounding_boxes_p100_zoom_100(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_subimagenet_bounding_boxes_p100_zoom_100
    model = resnet50_subimagenet_bounding_boxes_p100_zoom_100()
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_ade20k_objects_zoom_0_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_ade20k_objects_zoom_0_finetuned_full_on_subimagenet
    model = resnet50_ade20k_objects_zoom_0_finetuned_full_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_ade20k_objects_zoom_0_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_ade20k_objects_zoom_0_finetuned_on_subimagenet
    model = resnet50_ade20k_objects_zoom_0_finetuned_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_ade20k_objects_zoom_0_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_ade20k_objects_zoom_0_finetuned_layer4_on_subimagenet
    model = resnet50_ade20k_objects_zoom_0_finetuned_layer4_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_ade20k_objects_zoom_80_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_ade20k_objects_zoom_80_finetuned_on_subimagenet
    model = resnet50_ade20k_objects_zoom_80_finetuned_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_ade20k_objects_zoom_150_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_ade20k_objects_zoom_150_finetuned_full_on_subimagenet
    model = resnet50_ade20k_objects_zoom_150_finetuned_full_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_ade20k_objects_zoom_150_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_ade20k_objects_zoom_150_finetuned_on_subimagenet
    model = resnet50_ade20k_objects_zoom_150_finetuned_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_ade20k_objects_zoom_150_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_ade20k_objects_zoom_150_finetuned_layer4_on_subimagenet
    model = resnet50_ade20k_objects_zoom_150_finetuned_layer4_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_ade20k_objects_zoom_80_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_ade20k_objects_zoom_80_finetuned_full_on_subimagenet
    model = resnet50_ade20k_objects_zoom_80_finetuned_full_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_ade20k_objects_zoom_80_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_ade20k_objects_zoom_80_finetuned_layer4_on_subimagenet
    model = resnet50_ade20k_objects_zoom_80_finetuned_layer4_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)


@register_model("pytorch")
def resnet50_ade20k_scenes_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_ade20k_scenes_finetuned_full_on_subimagenet
    model = resnet50_ade20k_scenes_finetuned_full_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_ade20k_objects_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_ade20k_objects_finetuned_on_subimagenet
    model = resnet50_ade20k_objects_finetuned_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_ade20k_objects_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_ade20k_objects_finetuned_layer4_on_subimagenet
    model = resnet50_ade20k_objects_finetuned_layer4_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_ade20k_objects_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_ade20k_objects_finetuned_full_on_subimagenet
    model = resnet50_ade20k_objects_finetuned_full_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_ade20k_scenes_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_ade20k_scenes_finetuned_layer4_on_subimagenet
    model = resnet50_ade20k_scenes_finetuned_layer4_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_ade20k_scenes_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_ade20k_scenes_finetuned_on_subimagenet
    model = resnet50_ade20k_scenes_finetuned_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)



@register_model("pytorch")
def resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_100(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_100
    model = resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100_zoom_100()
    return PytorchModel(model=model, model_name=model_name, *args)


@register_model("pytorch")
def resnet50_oads_zoom_150_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_zoom_150_finetuned_on_subimagenet
    model = resnet50_oads_zoom_150_finetuned_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_oads_zoom_100_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_zoom_100_finetuned_on_subimagenet
    model = resnet50_oads_zoom_100_finetuned_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_oads_zoom_80_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_zoom_80_finetuned_on_subimagenet
    model = resnet50_oads_zoom_80_finetuned_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_all_boxes_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_all_boxes_imagenet
    model = resnet50_all_boxes_imagenet()
    return PytorchModel(model=model, model_name=model_name, *args)


@register_model("pytorch")
def resnet50_subimagenet_bounding_boxes_p70(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_subimagenet_bounding_boxes_p70
    model = resnet50_subimagenet_bounding_boxes_p70()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_subimagenet_bounding_boxes_p100(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_subimagenet_bounding_boxes_p100
    model = resnet50_subimagenet_bounding_boxes_p100()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_subimagenet_bounding_boxes_p100_zoom_30(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_subimagenet_bounding_boxes_p100_zoom_30
    model = resnet50_subimagenet_bounding_boxes_p100_zoom_30()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_subimagenet_bounding_boxes_p100_zoom_50(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_subimagenet_bounding_boxes_p100_zoom_50
    model = resnet50_subimagenet_bounding_boxes_p100_zoom_50()
    return PytorchModel(model=model, model_name=model_name, *args)


@register_model("pytorch")
def resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100
    model = resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p100()
    return PytorchModel(model=model, model_name=model_name, *args)
    
@register_model("pytorch")
def resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p70(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p70
    model = resnet50_imagenet_finetuned_on_subimagenet_bounding_boxes_p70()
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_zoom_50_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_zoom_50_finetuned_on_subimagenet
    model = resnet50_oads_zoom_50_finetuned_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_oads_zoom_50_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_zoom_50_finetuned_full_on_subimagenet
    model = resnet50_oads_zoom_50_finetuned_full_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_oads_zoom_50_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_zoom_50_finetuned_layer4_on_subimagenet
    model = resnet50_oads_zoom_50_finetuned_layer4_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)

##############################
@register_model("pytorch")
def resnet50_julio_random(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_random
    model = resnet50_julio_random()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_0(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_0
    model = resnet50_julio_epoch_0()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_1(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_1
    model = resnet50_julio_epoch_1()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_2(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_2
    model = resnet50_julio_epoch_2()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_3(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_3
    model = resnet50_julio_epoch_3()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_4(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_4
    model = resnet50_julio_epoch_4()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_5(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_5
    model = resnet50_julio_epoch_5()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_6(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_6
    model = resnet50_julio_epoch_6()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_7(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_7
    model = resnet50_julio_epoch_7()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_8(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_8
    model = resnet50_julio_epoch_8()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_9(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_9
    model = resnet50_julio_epoch_9()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_10(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_10
    model = resnet50_julio_epoch_10()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_11(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_11
    model = resnet50_julio_epoch_11()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_12(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_12
    model = resnet50_julio_epoch_12()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_13(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_13
    model = resnet50_julio_epoch_13()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_14(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_14
    model = resnet50_julio_epoch_14()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_15(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_15
    model = resnet50_julio_epoch_15()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_16(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_16
    model = resnet50_julio_epoch_16()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_17(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_17
    model = resnet50_julio_epoch_17()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_18(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_18
    model = resnet50_julio_epoch_18()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_19(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_19
    model = resnet50_julio_epoch_19()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_20(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_20
    model = resnet50_julio_epoch_20()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_21(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_21
    model = resnet50_julio_epoch_21()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_22(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_22
    model = resnet50_julio_epoch_22()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_23(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_23
    model = resnet50_julio_epoch_23()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_24(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_24
    model = resnet50_julio_epoch_24()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_25(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_25
    model = resnet50_julio_epoch_25()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_26(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_26
    model = resnet50_julio_epoch_26()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_27(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_27
    model = resnet50_julio_epoch_27()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_28(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_28
    model = resnet50_julio_epoch_28()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_29(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_29
    model = resnet50_julio_epoch_29()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_30(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_30
    model = resnet50_julio_epoch_30()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_31(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_31
    model = resnet50_julio_epoch_31()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_32(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_32
    model = resnet50_julio_epoch_32()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_33(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_33
    model = resnet50_julio_epoch_33()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_34(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_34
    model = resnet50_julio_epoch_34()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_35(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_35
    model = resnet50_julio_epoch_35()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_36(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_36
    model = resnet50_julio_epoch_36()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_37(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_37
    model = resnet50_julio_epoch_37()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_38(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_38
    model = resnet50_julio_epoch_38()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_39(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_39
    model = resnet50_julio_epoch_39()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_40(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_40
    model = resnet50_julio_epoch_40()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_41(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_41
    model = resnet50_julio_epoch_41()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_42(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_42
    model = resnet50_julio_epoch_42()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_43(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_43
    model = resnet50_julio_epoch_43()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_44(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_44
    model = resnet50_julio_epoch_44()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_45(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_45
    model = resnet50_julio_epoch_45()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_46(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_46
    model = resnet50_julio_epoch_46()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_47(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_47
    model = resnet50_julio_epoch_47()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_48(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_48
    model = resnet50_julio_epoch_48()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_49(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_49
    model = resnet50_julio_epoch_49()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_50(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_50
    model = resnet50_julio_epoch_50()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_51(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_51
    model = resnet50_julio_epoch_51()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_52(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_52
    model = resnet50_julio_epoch_52()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_53(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_53
    model = resnet50_julio_epoch_53()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_54(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_54
    model = resnet50_julio_epoch_54()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_55(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_55
    model = resnet50_julio_epoch_55()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_56(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_56
    model = resnet50_julio_epoch_56()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_57(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_57
    model = resnet50_julio_epoch_57()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_58(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_58
    model = resnet50_julio_epoch_58()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_59(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_59
    model = resnet50_julio_epoch_59()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_60(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_60
    model = resnet50_julio_epoch_60()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_61(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_61
    model = resnet50_julio_epoch_61()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_62(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_62
    model = resnet50_julio_epoch_62()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_63(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_63
    model = resnet50_julio_epoch_63()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_64(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_64
    model = resnet50_julio_epoch_64()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_65(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_65
    model = resnet50_julio_epoch_65()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_66(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_66
    model = resnet50_julio_epoch_66()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_67(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_67
    model = resnet50_julio_epoch_67()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_68(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_68
    model = resnet50_julio_epoch_68()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_69(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_69
    model = resnet50_julio_epoch_69()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_70(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_70
    model = resnet50_julio_epoch_70()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_71(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_71
    model = resnet50_julio_epoch_71()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_72(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_72
    model = resnet50_julio_epoch_72()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_73(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_73
    model = resnet50_julio_epoch_73()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_74(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_74
    model = resnet50_julio_epoch_74()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_75(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_75
    model = resnet50_julio_epoch_75()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_76(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_76
    model = resnet50_julio_epoch_76()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_77(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_77
    model = resnet50_julio_epoch_77()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_78(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_78
    model = resnet50_julio_epoch_78()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_79(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_79
    model = resnet50_julio_epoch_79()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_80(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_80
    model = resnet50_julio_epoch_80()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_81(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_81
    model = resnet50_julio_epoch_81()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_82(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_82
    model = resnet50_julio_epoch_82()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_83(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_83
    model = resnet50_julio_epoch_83()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_84(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_84
    model = resnet50_julio_epoch_84()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_85(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_85
    model = resnet50_julio_epoch_85()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_86(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_86
    model = resnet50_julio_epoch_86()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_87(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_87
    model = resnet50_julio_epoch_87()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_88(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_88
    model = resnet50_julio_epoch_88()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_89(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_89
    model = resnet50_julio_epoch_89()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_89(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_89
    model = resnet50_julio_epoch_89()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_julio_epoch_90(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_julio_epoch_90
    model = resnet50_julio_epoch_90()
    return PytorchModel(model=model, model_name=model_name, *args)


@register_model("pytorch")
def alexnet_julio_random(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_random
    model = alexnet_julio_random()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_0(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_0
    model = alexnet_julio_epoch_0()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_1(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_1
    model = alexnet_julio_epoch_1()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_2(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_2
    model = alexnet_julio_epoch_2()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_3(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_3
    model = alexnet_julio_epoch_3()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_4(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_4
    model = alexnet_julio_epoch_4()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_5(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_5
    model = alexnet_julio_epoch_5()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_6(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_6
    model = alexnet_julio_epoch_6()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_7(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_7
    model = alexnet_julio_epoch_7()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_8(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_8
    model = alexnet_julio_epoch_8()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_9(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_9
    model = alexnet_julio_epoch_9()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_10(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_10
    model = alexnet_julio_epoch_10()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_11(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_11
    model = alexnet_julio_epoch_11()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_12(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_12
    model = alexnet_julio_epoch_12()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_13(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_13
    model = alexnet_julio_epoch_13()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_14(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_14
    model = alexnet_julio_epoch_14()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_15(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_15
    model = alexnet_julio_epoch_15()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_16(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_16
    model = alexnet_julio_epoch_16()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_17(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_17
    model = alexnet_julio_epoch_17()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_18(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_18
    model = alexnet_julio_epoch_18()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_19(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_19
    model = alexnet_julio_epoch_19()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_20(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_20
    model = alexnet_julio_epoch_20()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_21(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_21
    model = alexnet_julio_epoch_21()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_22(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_22
    model = alexnet_julio_epoch_22()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_23(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_23
    model = alexnet_julio_epoch_23()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_24(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_24
    model = alexnet_julio_epoch_24()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_25(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_25
    model = alexnet_julio_epoch_25()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_26(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_26
    model = alexnet_julio_epoch_26()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_27(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_27
    model = alexnet_julio_epoch_27()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_28(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_28
    model = alexnet_julio_epoch_28()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_29(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_29
    model = alexnet_julio_epoch_29()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_30(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_30
    model = alexnet_julio_epoch_30()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_31(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_31
    model = alexnet_julio_epoch_31()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_32(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_32
    model = alexnet_julio_epoch_32()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_33(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_33
    model = alexnet_julio_epoch_33()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_34(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_34
    model = alexnet_julio_epoch_34()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_35(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_35
    model = alexnet_julio_epoch_35()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_36(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_36
    model = alexnet_julio_epoch_36()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_37(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_37
    model = alexnet_julio_epoch_37()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_38(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_38
    model = alexnet_julio_epoch_38()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_39(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_39
    model = alexnet_julio_epoch_39()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_40(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_40
    model = alexnet_julio_epoch_40()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_41(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_41
    model = alexnet_julio_epoch_41()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_42(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_42
    model = alexnet_julio_epoch_42()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_43(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_43
    model = alexnet_julio_epoch_43()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_44(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_44
    model = alexnet_julio_epoch_44()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_45(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_45
    model = alexnet_julio_epoch_45()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_46(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_46
    model = alexnet_julio_epoch_46()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_47(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_47
    model = alexnet_julio_epoch_47()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_48(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_48
    model = alexnet_julio_epoch_48()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_49(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_49
    model = alexnet_julio_epoch_49()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_50(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_50
    model = alexnet_julio_epoch_50()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_51(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_51
    model = alexnet_julio_epoch_51()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_52(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_52
    model = alexnet_julio_epoch_52()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_53(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_53
    model = alexnet_julio_epoch_53()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_54(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_54
    model = alexnet_julio_epoch_54()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_55(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_55
    model = alexnet_julio_epoch_55()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_56(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_56
    model = alexnet_julio_epoch_56()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_57(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_57
    model = alexnet_julio_epoch_57()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_58(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_58
    model = alexnet_julio_epoch_58()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_59(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_59
    model = alexnet_julio_epoch_59()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_60(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_60
    model = alexnet_julio_epoch_60()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_61(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_61
    model = alexnet_julio_epoch_61()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_62(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_62
    model = alexnet_julio_epoch_62()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_63(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_63
    model = alexnet_julio_epoch_63()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_64(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_64
    model = alexnet_julio_epoch_64()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_65(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_65
    model = alexnet_julio_epoch_65()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_66(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_66
    model = alexnet_julio_epoch_66()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_67(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_67
    model = alexnet_julio_epoch_67()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_68(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_68
    model = alexnet_julio_epoch_68()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_69(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_69
    model = alexnet_julio_epoch_69()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_70(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_70
    model = alexnet_julio_epoch_70()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_71(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_71
    model = alexnet_julio_epoch_71()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_72(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_72
    model = alexnet_julio_epoch_72()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_73(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_73
    model = alexnet_julio_epoch_73()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_74(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_74
    model = alexnet_julio_epoch_74()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_75(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_75
    model = alexnet_julio_epoch_75()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_76(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_76
    model = alexnet_julio_epoch_76()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_77(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_77
    model = alexnet_julio_epoch_77()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_78(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_78
    model = alexnet_julio_epoch_78()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_79(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_79
    model = alexnet_julio_epoch_79()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_80(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_80
    model = alexnet_julio_epoch_80()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_81(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_81
    model = alexnet_julio_epoch_81()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_82(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_82
    model = alexnet_julio_epoch_82()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_83(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_83
    model = alexnet_julio_epoch_83()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_84(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_84
    model = alexnet_julio_epoch_84()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_85(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_85
    model = alexnet_julio_epoch_85()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_86(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_86
    model = alexnet_julio_epoch_86()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_87(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_87
    model = alexnet_julio_epoch_87()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_88(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_88
    model = alexnet_julio_epoch_88()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_89(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_89
    model = alexnet_julio_epoch_89()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def alexnet_julio_epoch_90(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import alexnet_julio_epoch_90
    model = alexnet_julio_epoch_90()
    return PytorchModel(model=model, model_name=model_name, *args)

##############################


@register_model("pytorch")
def vit_b_16_oads_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_oads_finetuned_on_subimagenet
    model = vit_b_16_oads_finetuned_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def vit_b_16_oads_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_oads_finetuned_layer4_on_subimagenet
    model = vit_b_16_oads_finetuned_layer4_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def vit_b_16_oads_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import vit_b_16_oads_finetuned_full_on_subimagenet
    model = vit_b_16_oads_finetuned_full_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def fasterrcnn_resnet50_fpn_coco_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import fasterrcnn_resnet50_fpn_coco_finetuned_layer4_on_subimagenet
    model = fasterrcnn_resnet50_fpn_coco_finetuned_layer4_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def fasterrcnn_resnet50_fpn_coco_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import fasterrcnn_resnet50_fpn_coco_finetuned_full_on_subimagenet
    model = fasterrcnn_resnet50_fpn_coco_finetuned_full_on_subimagenet()
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def fasterrcnn_resnet50_fpn_coco_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import fasterrcnn_resnet50_fpn_coco_finetuned_on_subimagenet
    model = fasterrcnn_resnet50_fpn_coco_finetuned_on_subimagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_coco_objects_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_coco_objects_finetuned_on_subimagenet
    model = resnet50_coco_objects_finetuned_on_subimagenet()

    return PytorchModel(model=model, model_name=model_name, *args)
    
@register_model("pytorch")
def resnet50_coco_objects_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_coco_objects_finetuned_full_on_subimagenet
    model = resnet50_coco_objects_finetuned_full_on_subimagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_coco_objects_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_coco_objects_finetuned_layer4_on_subimagenet
    model = resnet50_coco_objects_finetuned_layer4_on_subimagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_subimagenet_bounding_boxes(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_subimagenet_bounding_boxes
    model = resnet50_subimagenet_bounding_boxes()

    return PytorchModel(model=model, model_name=model_name, *args)




@register_model("pytorch")
def fcn_resnet50_coco_oads_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import fcn_resnet50_coco_oads_finetuned_full_on_subimagenet
    model = fcn_resnet50_coco_oads_finetuned_full_on_subimagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def fcn_resnet50_coco_oads_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import fcn_resnet50_coco_oads_finetuned_layer4_on_subimagenet
    model = fcn_resnet50_coco_oads_finetuned_layer4_on_subimagenet()

    return PytorchModel(model=model, model_name=model_name, *args)
    
@register_model("pytorch")
def fcn_resnet50_coco_oads_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import fcn_resnet50_coco_oads_finetuned_on_subimagenet
    model = fcn_resnet50_coco_oads_finetuned_on_subimagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def fcn_resnet50_coco_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import fcn_resnet50_coco_finetuned_on_subimagenet
    model = fcn_resnet50_coco_finetuned_on_subimagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def fcn_resnet50_coco_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import fcn_resnet50_coco_finetuned_full_on_subimagenet
    model = fcn_resnet50_coco_finetuned_full_on_subimagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def fcn_resnet50_coco_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import fcn_resnet50_coco_finetuned_layer4_on_subimagenet
    model = fcn_resnet50_coco_finetuned_layer4_on_subimagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_places365_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_places365_finetuned_on_subimagenet
    model = resnet50_places365_finetuned_on_subimagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_places365_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_places365_finetuned_full_on_subimagenet
    model = resnet50_places365_finetuned_full_on_subimagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_places365_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_places365_finetuned_layer4_on_subimagenet
    model = resnet50_places365_finetuned_layer4_on_subimagenet()

    return PytorchModel(model=model, model_name=model_name, *args)
    
@register_model("pytorch")
def resnet50_imagenet_no_crop_224x224(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_imagenet_no_crop_224x224
    model = resnet50_imagenet_no_crop_224x224()

    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_imagenet_no_crop_400x400(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_imagenet_no_crop_400x400
    model = resnet50_imagenet_no_crop_400x400()

    return PytorchModel(model=model, model_name=model_name, *args)
    
@register_model("pytorch")
def resnet50_imagenet_350x350_crop_224x224(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_imagenet_350x350_crop_224x224
    model = resnet50_imagenet_350x350_crop_224x224()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_imagenet_600x600_crop_400x400(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_imagenet_600x600_crop_400x400
    model = resnet50_imagenet_600x600_crop_400x400()

    return PytorchModel(model=model, model_name=model_name, *args)


@register_model("pytorch")
def resnet50_imagenet_400x400(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_imagenet_400x400
    model = resnet50_imagenet_400x400()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_imagenet_500x500(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_imagenet_500x500
    model = resnet50_imagenet_500x500()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_imagenet_600x600(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_imagenet_600x600
    model = resnet50_imagenet_600x600()

    return PytorchModel(model=model, model_name=model_name, *args)


@register_model("pytorch")
def resnet50_imagenet_80x80_to_224x224(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_imagenet_80x80_to_224x224
    model = resnet50_imagenet_80x80_to_224x224()

    return PytorchModel(model=model, model_name=model_name, *args)
    
@register_model("pytorch")
def resnet50_imagenet_30x30_to_224x224(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_imagenet_30x30_to_224x224
    model = resnet50_imagenet_30x30_to_224x224()

    return PytorchModel(model=model, model_name=model_name, *args)

    
@register_model("pytorch")
def resnet50_imagenet_10x10_to_224x224(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_imagenet_10x10_to_224x224
    model = resnet50_imagenet_10x10_to_224x224()

    return PytorchModel(model=model, model_name=model_name, *args)
@register_model("pytorch")
def resnet50_imagenet_112x112_to_224x224(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_imagenet_112x112_to_224x224
    model = resnet50_imagenet_112x112_to_224x224()

    return PytorchModel(model=model, model_name=model_name, *args)


@register_model("pytorch")
def resnet50_imagenet_400x400_low_res(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_imagenet_400x400_low_res
    model = resnet50_imagenet_400x400_low_res()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_imagenet_112x112(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_imagenet_112x112
    model = resnet50_imagenet_112x112()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_s2_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_s2_imagenet
    model = resnet50_s2_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def random_resnet18(model_name, *args):

    return PytorchModel(model=resnet18(weights=None), model_name=model_name, *args)
@register_model("pytorch")
def random_resnet50(model_name, *args):

    return PytorchModel(model=resnet50(weights=None), model_name=model_name, *args)


@register_model("pytorch")
def resnet50_imagenet_subclasses(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_imagenet_subclasses
    model = resnet50_imagenet_subclasses()
    
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_imagenet_grayscale(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_imagenet_grayscale
    model = resnet50_imagenet_grayscale()
    
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_finetuned_imagenetsize_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_finetuned_imagenetsize_on_imagenet
    model = resnet50_oads_normalized_rgb_finetuned_imagenetsize_on_imagenet()
    
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_finetuned_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_finetuned_on_imagenet
    model = resnet50_oads_normalized_rgb_finetuned_on_imagenet()
    
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_jpeg_finetuned_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_jpeg_finetuned_on_imagenet
    model = resnet50_oads_normalized_rgb_jpeg_finetuned_on_imagenet()
    
    return PytorchModel(model=model, model_name=model_name, *args)


@register_model("pytorch")
def resnet50_oads_normalized_rgb_jpeg_finetuned_full_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_jpeg_finetuned_full_on_imagenet
    model = resnet50_oads_normalized_rgb_jpeg_finetuned_full_on_imagenet()
    
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_on_imagenet
    model = resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_on_imagenet()
    
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_finetuned_on_subimagenet
    model = resnet50_oads_normalized_rgb_finetuned_on_subimagenet()
    
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_jpeg_finetuned_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_jpeg_finetuned_on_subimagenet
    model = resnet50_oads_normalized_rgb_jpeg_finetuned_on_subimagenet()
    
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_jpeg_finetuned_imagenetsize_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_jpeg_finetuned_imagenetsize_on_subimagenet
    model = resnet50_oads_normalized_rgb_jpeg_finetuned_imagenetsize_on_subimagenet()
    
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_finetuned_imagenetsize_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_finetuned_imagenetsize_on_subimagenet
    model = resnet50_oads_normalized_rgb_finetuned_imagenetsize_on_subimagenet()
    
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_finetuned_full_on_subimagenet
    model = resnet50_oads_normalized_rgb_finetuned_full_on_subimagenet()
    
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_jpeg_finetuned_full_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_jpeg_finetuned_full_on_subimagenet
    model = resnet50_oads_normalized_rgb_jpeg_finetuned_full_on_subimagenet()
    
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_jpeg_finetuned_full_imagenetsize_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_jpeg_finetuned_full_imagenetsize_on_subimagenet
    model = resnet50_oads_normalized_rgb_jpeg_finetuned_full_imagenetsize_on_subimagenet()
    
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_finetuned_full_imagenetsize_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_finetuned_full_imagenetsize_on_subimagenet
    model = resnet50_oads_normalized_rgb_finetuned_full_imagenetsize_on_subimagenet()
    
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_finetuned_layer4_on_subimagenet
    model = resnet50_oads_normalized_rgb_finetuned_layer4_on_subimagenet()
    
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_on_subimagenet
    model = resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_on_subimagenet()
    
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_imagenetsize_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_imagenetsize_on_subimagenet
    model = resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_imagenetsize_on_subimagenet()
    
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_finetuned_layer4_imagenetsize_on_subimagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_finetuned_layer4_imagenetsize_on_subimagenet
    model = resnet50_oads_normalized_rgb_finetuned_layer4_imagenetsize_on_subimagenet()
    
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_jpeg_finetuned_imagenetsize_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_jpeg_finetuned_imagenetsize_on_imagenet
    model = resnet50_oads_normalized_rgb_jpeg_finetuned_imagenetsize_on_imagenet()
    
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_jpeg_40_finetuned_layer4_imagenetsize_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_jpeg_40_finetuned_layer4_imagenetsize_on_imagenet
    model = resnet50_oads_normalized_rgb_jpeg_40_finetuned_layer4_imagenetsize_on_imagenet()
    
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_jpeg_60_finetuned_layer4_imagenetsize_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_jpeg_60_finetuned_layer4_imagenetsize_on_imagenet
    model = resnet50_oads_normalized_rgb_jpeg_60_finetuned_layer4_imagenetsize_on_imagenet()
    
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_jpeg_90_finetuned_layer4_imagenetsize_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_jpeg_90_finetuned_layer4_imagenetsize_on_imagenet
    model = resnet50_oads_normalized_rgb_jpeg_90_finetuned_layer4_imagenetsize_on_imagenet()
    
    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_imagenetsize_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_imagenetsize_on_imagenet
    model = resnet50_oads_normalized_rgb_jpeg_finetuned_layer4_imagenetsize_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_jpeg_60_finetuned_full_imagenetsize_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_jpeg_60_finetuned_full_imagenetsize_on_imagenet
    model = resnet50_oads_normalized_rgb_jpeg_60_finetuned_full_imagenetsize_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_jpeg_40_finetuned_full_imagenetsize_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_jpeg_40_finetuned_full_imagenetsize_on_imagenet
    model = resnet50_oads_normalized_rgb_jpeg_40_finetuned_full_imagenetsize_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_jpeg_90_finetuned_full_imagenetsize_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_jpeg_90_finetuned_full_imagenetsize_on_imagenet
    model = resnet50_oads_normalized_rgb_jpeg_90_finetuned_full_imagenetsize_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_finetuned_full_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_finetuned_full_on_imagenet
    model = resnet50_oads_normalized_rgb_finetuned_full_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_finetuned_layer4_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_finetuned_layer4_on_imagenet
    model = resnet50_oads_normalized_rgb_finetuned_layer4_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet18_oads_normalized_rgb_finetuned_layer4_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet18_oads_normalized_rgb_finetuned_layer4_on_imagenet
    model = resnet18_oads_normalized_rgb_finetuned_layer4_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet18_oads_normalized_rgb_jpeg_finetuned_layer4_imagenetsize_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet18_oads_normalized_rgb_jpeg_finetuned_layer4_imagenetsize_on_imagenet
    model = resnet18_oads_normalized_rgb_jpeg_finetuned_layer4_imagenetsize_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet18_oads_rgb_finetuned_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet18_oads_rgb_finetuned_on_imagenet
    model = resnet18_oads_rgb_finetuned_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet18_oads_rgb_jpeg_finetuned_layer4_imagenetsize_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet18_oads_rgb_jpeg_finetuned_layer4_imagenetsize_on_imagenet
    model = resnet18_oads_rgb_jpeg_finetuned_layer4_imagenetsize_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet18_oads_coc_finetuned_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet18_oads_coc_finetuned_on_imagenet
    model = resnet18_oads_coc_finetuned_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet18_oads_rgb_jpeg_finetuned_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet18_oads_rgb_jpeg_finetuned_on_imagenet
    model = resnet18_oads_rgb_jpeg_finetuned_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet18_oads_coc_jpeg_finetuned_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet18_oads_coc_jpeg_finetuned_on_imagenet
    model = resnet18_oads_coc_jpeg_finetuned_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet18_oads_rgb_finetuned_full_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet18_oads_rgb_finetuned_full_on_imagenet
    model = resnet18_oads_rgb_finetuned_full_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet18_oads_coc_finetuned_full_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet18_oads_coc_finetuned_full_on_imagenet
    model = resnet18_oads_coc_finetuned_full_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet18_oads_rgb_jpeg_finetuned_full_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet18_oads_rgb_jpeg_finetuned_full_on_imagenet
    model = resnet18_oads_rgb_jpeg_finetuned_full_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet18_oads_coc_jpeg_finetuned_full_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet18_oads_coc_jpeg_finetuned_full_on_imagenet
    model = resnet18_oads_coc_jpeg_finetuned_full_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet18_oads_rgb_finetuned_layer4_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet18_oads_rgb_finetuned_layer4_on_imagenet
    model = resnet18_oads_rgb_finetuned_layer4_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet18_oads_coc_finetuned_layer4_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet18_oads_coc_finetuned_layer4_on_imagenet
    model = resnet18_oads_coc_finetuned_layer4_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet18_oads_rgb_jpeg_finetuned_layer4_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet18_oads_rgb_jpeg_finetuned_layer4_on_imagenet
    model = resnet18_oads_rgb_jpeg_finetuned_layer4_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet18_oads_coc_jpeg_finetuned_layer4_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet18_oads_coc_jpeg_finetuned_layer4_on_imagenet
    model = resnet18_oads_coc_jpeg_finetuned_layer4_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_rgb_finetuned_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_rgb_finetuned_on_imagenet
    model = resnet50_oads_rgb_finetuned_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_coc_finetuned_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_coc_finetuned_on_imagenet
    model = resnet50_oads_coc_finetuned_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_rgb_jpeg_finetuned_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_rgb_jpeg_finetuned_on_imagenet
    model = resnet50_oads_rgb_jpeg_finetuned_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_coc_jpeg_finetuned_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_coc_jpeg_finetuned_on_imagenet
    model = resnet50_oads_coc_jpeg_finetuned_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_rgb_finetuned_layer4_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_rgb_finetuned_layer4_on_imagenet
    model = resnet50_oads_rgb_finetuned_layer4_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_coc_finetuned_layer4_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_coc_finetuned_layer4_on_imagenet
    model = resnet50_oads_coc_finetuned_layer4_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_rgb_jpeg_finetuned_layer4_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_rgb_jpeg_finetuned_layer4_on_imagenet
    model = resnet50_oads_rgb_jpeg_finetuned_layer4_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_coc_jpeg_finetuned_layer4_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_coc_jpeg_finetuned_layer4_on_imagenet
    model = resnet50_oads_coc_jpeg_finetuned_layer4_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_rgb_finetuned_full_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_rgb_finetuned_full_on_imagenet
    model = resnet50_oads_rgb_finetuned_full_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_finetuned_full_imagenetsize_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_finetuned_full_imagenetsize_on_imagenet
    model = resnet50_oads_normalized_rgb_finetuned_full_imagenetsize_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_normalized_rgb_finetuned_layer4_imagenetsize_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_normalized_rgb_finetuned_layer4_imagenetsize_on_imagenet
    model = resnet50_oads_normalized_rgb_finetuned_layer4_imagenetsize_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)


@register_model("pytorch")
def resnet50_oads_coc_finetuned_full_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_coc_finetuned_full_on_imagenet
    model = resnet50_oads_coc_finetuned_full_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_rgb_jpeg_finetuned_full_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_rgb_jpeg_finetuned_full_on_imagenet
    model = resnet50_oads_rgb_jpeg_finetuned_full_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)

@register_model("pytorch")
def resnet50_oads_coc_jpeg_finetuned_full_on_imagenet(model_name, *args):
    from .resnet_oads_finetuned_on_imagenet.resnet_oads_models import resnet50_oads_coc_jpeg_finetuned_full_on_imagenet
    model = resnet50_oads_coc_jpeg_finetuned_full_on_imagenet()

    return PytorchModel(model=model, model_name=model_name, *args)


@register_model("pytorch")
def resnet50_trained_on_SIN(model_name, *args):
    from .shapenet import texture_shape_models as tsm
    model = tsm.load_model(model_name)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_trained_on_SIN_and_IN(model_name, *args):
    from .shapenet import texture_shape_models as tsm
    model = tsm.load_model(model_name)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN(model_name, *args):
    from .shapenet import texture_shape_models as tsm
    model = tsm.load_model(model_name)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def bagnet9(model_name, *args):
    from .bagnets.pytorchnet import bagnet9
    model = bagnet9(pretrained=True)
    model = DataParallel(model)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def bagnet17(model_name, *args):
    from .bagnets.pytorchnet import bagnet17
    model = bagnet17(pretrained=True)
    model = DataParallel(model)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def bagnet33(model_name, *args):
    from .bagnets.pytorchnet import bagnet33
    model = bagnet33(pretrained=True)
    model = DataParallel(model)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def simclr_resnet50x1_supervised_baseline(model_name, *args):
    from .simclr import simclr_resnet50x1_supervised_baseline
    model = simclr_resnet50x1_supervised_baseline(pretrained=True, use_data_parallel=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def simclr_resnet50x4_supervised_baseline(model_name, *args):
    from .simclr import simclr_resnet50x4_supervised_baseline
    model = simclr_resnet50x4_supervised_baseline(pretrained=True, use_data_parallel=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def simclr_resnet50x1(model_name, *args):
    from .simclr import simclr_resnet50x1
    model = simclr_resnet50x1(pretrained=True, use_data_parallel=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def simclr_resnet50x2(model_name, *args):
    from .simclr import simclr_resnet50x2
    model = simclr_resnet50x2(pretrained=True, use_data_parallel=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def simclr_resnet50x4(model_name, *args):
    from .simclr import simclr_resnet50x4
    model = simclr_resnet50x4(pretrained=True,
                              use_data_parallel=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def InsDis(model_name, *args):
    from .pycontrast.pycontrast_resnet50 import InsDis
    model, classifier = InsDis(pretrained=True)
    return PyContrastPytorchModel(*(model, classifier), model_name, *args)


@register_model("pytorch")
def MoCo(model_name, *args):
    from .pycontrast.pycontrast_resnet50 import MoCo
    model, classifier = MoCo(pretrained=True)
    return PyContrastPytorchModel(*(model, classifier), model_name, *args)


@register_model("pytorch")
def MoCoV2(model_name, *args):
    from .pycontrast.pycontrast_resnet50 import MoCoV2
    model, classifier = MoCoV2(pretrained=True)
    return PyContrastPytorchModel(*(model, classifier), model_name, *args)


@register_model("pytorch")
def PIRL(model_name, *args):
    from .pycontrast.pycontrast_resnet50 import PIRL
    model, classifier = PIRL(pretrained=True)
    return PyContrastPytorchModel(*(model, classifier), model_name, *args)


@register_model("pytorch")
def InfoMin(model_name, *args):
    from .pycontrast.pycontrast_resnet50 import InfoMin
    model, classifier = InfoMin(pretrained=True)
    return PyContrastPytorchModel(*(model, classifier), model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps0
    model = resnet50_l2_eps0()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_01(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_01
    model = resnet50_l2_eps0_01()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_03(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_03
    model = resnet50_l2_eps0_03()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_05(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_05
    model = resnet50_l2_eps0_05()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_1(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_1
    model = resnet50_l2_eps0_1()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_25(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_25
    model = resnet50_l2_eps0_25()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_5(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_5
    model = resnet50_l2_eps0_5()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps1(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps1
    model = resnet50_l2_eps1()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps3(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps3
    model = resnet50_l2_eps3()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps5(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps5
    model = resnet50_l2_eps5()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def efficientnet_b0(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def efficientnet_es(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def efficientnet_b0_noisy_student(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           "tf_efficientnet_b0_ns",
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def efficientnet_l2_noisy_student_475(model_name, *args):
    model = torch.hub.load(_EFFICIENTNET_MODELS,
                           "tf_efficientnet_l2_ns_475",
                           pretrained=True)
    return EfficientNetPytorchModel(model, model_name, *args)


@register_model("pytorch")
def transformer_B16_IN21K(model_name, *args):
    from pytorch_pretrained_vit import ViT
    model = ViT('B_16_imagenet1k', pretrained=True)
    return ViTPytorchModel(model, model_name, *args)


@register_model("pytorch")
def transformer_B32_IN21K(model_name, *args):
    from pytorch_pretrained_vit import ViT
    model = ViT('B_32_imagenet1k', pretrained=True)
    return ViTPytorchModel(model, model_name, *args)


@register_model("pytorch")
def transformer_L16_IN21K(model_name, *args):
    from pytorch_pretrained_vit import ViT
    model = ViT('L_16_imagenet1k', pretrained=True)
    return ViTPytorchModel(model, model_name, *args)


@register_model("pytorch")
def transformer_L32_IN21K(model_name, *args):
    from pytorch_pretrained_vit import ViT
    model = ViT('L_32_imagenet1k', pretrained=True)
    return ViTPytorchModel(model, model_name, *args)


@register_model("pytorch")
def vit_small_patch16_224(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    img_size = (224, 224)
    return ViTPytorchModel(model, model_name, img_size, *args)


@register_model("pytorch")
def vit_base_patch16_224(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    img_size = (224, 224)
    return ViTPytorchModel(model, model_name, img_size, *args)


@register_model("pytorch")
def vit_large_patch16_224(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    img_size = (224, 224)
    return ViTPytorchModel(model, model_name, img_size, *args)


@register_model("pytorch")
def cspresnet50(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cspresnext50(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cspdarknet53(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def darknet53(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn68(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn68b(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn92(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn98(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn131(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn107(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w18_small(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w18_small(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w18_small_v2(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w18(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w30(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w40(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w44(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w48(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w64(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def selecsls42(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def selecsls84(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def selecsls42b(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def selecsls60(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def selecsls60b(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def clip(model_name, *args):
    import clip
    model, _ = clip.load("ViT-B/32")
    return ClipPytorchModel(model, model_name, *args)


@register_model("pytorch")
def clipRN50(model_name, *args):
    import clip
    model, _ = clip.load("RN50")
    return ClipPytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_swsl(model_name, *args):
    model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models',
                           'resnet50_swsl')
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def ResNeXt101_32x16d_swsl(model_name, *args):
    model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models',
                           'resnext101_32x16d_swsl')
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_50x1(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           "resnetv2_50x1_bitm",
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_50x3(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           "resnetv2_50x3_bitm",
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_101x1(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           "resnetv2_101x1_bitm",
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_101x3(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           "resnetv2_101x3_bitm",
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_152x2(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           "resnetv2_152x2_bitm",
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_152x4(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           "resnetv2_152x4_bitm",
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_clip_hard_labels(model_name, *args):
    import torchvision.models as zoomodels
    model = zoomodels.__dict__["resnet50"](pretrained=False)
    model = DataParallel(model)
    checkpoint = torch.hub.load_state_dict_from_url("https://github.com/bethgelab/model-vs-human/releases/download/v0.3"
                                                    "/ResNet50_clip_hard_labels.pth",map_location='cpu')
    model.load_state_dict(checkpoint["state_dict"])
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_clip_soft_labels(model_name, *args):
    import torchvision.models as zoomodels
    model = zoomodels.__dict__["resnet50"](pretrained=False)
    model = DataParallel(model)
    checkpoint = torch.hub.load_state_dict_from_url("https://github.com/bethgelab/model-vs-human/releases/download/v0.3"
                                                    "/ResNet50_clip_soft_labels.pth", map_location='cpu')
    model.load_state_dict(checkpoint["state_dict"])
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def swag_regnety_16gf_in1k(model_name, *args):
    model = torch.hub.load("facebookresearch/swag", model="regnety_16gf_in1k")
    return SwagPytorchModel(model, model_name, input_size=384, *args)


@register_model("pytorch")
def swag_regnety_32gf_in1k(model_name, *args):
    model = torch.hub.load("facebookresearch/swag", model="regnety_32gf_in1k")
    return SwagPytorchModel(model, model_name, input_size=384, *args)


@register_model("pytorch")
def swag_regnety_128gf_in1k(model_name, *args):
    model = torch.hub.load("facebookresearch/swag", model="regnety_128gf_in1k")
    return SwagPytorchModel(model, model_name, input_size=384, *args)


@register_model("pytorch")
def swag_vit_b16_in1k(model_name, *args):
    model = torch.hub.load("facebookresearch/swag", model="vit_b16_in1k")
    return SwagPytorchModel(model, model_name, input_size=384, *args)


@register_model("pytorch")
def swag_vit_l16_in1k(model_name, *args):
    model = torch.hub.load("facebookresearch/swag", model="vit_l16_in1k")
    return SwagPytorchModel(model, model_name, input_size=512, *args)


@register_model("pytorch")
def swag_vit_h14_in1k(model_name, *args):
    model = torch.hub.load("facebookresearch/swag", model="vit_h14_in1k")
    return SwagPytorchModel(model, model_name, input_size=518, *args)
