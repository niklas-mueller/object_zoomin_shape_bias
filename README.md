# \<TITLE\>

This repository contains the code for the paper "".

## Create dataset

In this paper, we created zoom-in versions of MS-COCO, ADE20K, and our custom, high resolution Open Amsterdam Data Set (OADS). To do so, we make use of publicly available annotations (object bounding boxes for MS-COCO and OADS; segmentation maps for ADE20K) to create object crops. We systematically vary the amount of surrounding background information that is included in the object crop from 0\%, to 80\%, to 150\%.

Scripts: ```make_datasets```:
- ```make_coco_objects.py``` - Script to create object crop version of MS-COCO
- ```make_oads_objects.py``` - Script to create object crop version of OADS
- For ADE20K we do not save the object crop version to files but instead create the object crops on the fly during training (see ```train_models/train_dnn_ade20k_objects.py```)
- ```make_oads_cue_conflict_images.py``` - Script to create the new cue-conflict dataset for OADS tight crops using neural style transfer

![coco_crops.png]

## Network training

We train 4 ResNet50 instances, using 4 different, random seeds, on each of the 9 object crop datasets for 30 epochs. After training, we finetune each network on ImageNet for 15 epochs. We finetune each network instance multiple times using varying amount of parameters that are kept fixed during finetuning.

Scripts: ```train_models```:
- ```train_dnn_ade20k_objects.py``` - Train a network on the object crop version of ADE20K
- ```train_dnn_coco_objects.py``` - Train a network on the object crop version of MS-COCO
- ```train_dnn_oads_objects.py``` - Train a network on the object crop version of OADS

As control models, we also include models trained on scene segmentation on ADE20K and on object detection on MS-COCO:
- ```train_dnn_coco_scenes.py``` - Train a network on object detection on MS-COCO
<!-- - ```train_ade20k_scene_segmentation.py``` - Train a network on scene segmentation on ADE20K -->



## Texture-bias assessment

Scripts ```texture_shape_bias```:
- ```assess.py``` - Scripts to assess texture-shape-bias for ImageNet and OADS, using cue-conflict images from Geirhos et al. (2018) or using our newly created OADS cue-conflict images. This script uses an adapted version of the ```models-vs-human``` repository (we essentially removed the tensorflow part of the package)