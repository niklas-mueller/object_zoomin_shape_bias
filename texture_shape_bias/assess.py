import os
import yaml
import csv
import torch
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


os.environ['MODELVSHUMANDIR'] = f'{os.path.dirname(os.path.abspath(__file__))}/../model-vs-human'
sys.path.append(f'../model-vs-human')

from modelvshuman import Evaluate
from modelvshuman.models.wrappers.pytorch import PytorchModel

from base.models import ADE20K_ImageNetModel, fasterrcnn_resnet50_fpn_coco, load_model
from base.datasets import OADSCueConflict


def assess_model_on_oads(model_name, model_path):
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results_journal', 'human_class_responses', 'oads_400x400_cue-conflict')
    home_path = os.path.expanduser('~')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    use_human_classes = True

    batch_size = 8
    num_workers = 8
    image_size = 400

    if os.path.exists(os.path.join(result_dir, f'{model_name}_responses.yml')):
        return
    
    if 'fasterrcnn' in model_name:
        model_fn = fasterrcnn_resnet50_fpn_coco
    elif 'ade20k_scenes' in model_name:
        model_fn = ADE20K_ImageNetModel
    else:
        model_fn = resnet50

    model = load_model(model_name, model_fn, in_features=2048, state_dict_path=model_path, num_classes=21)
    model.to(device)
    
    oads_root = f'{home_path}/projects/fmg_storage/oads_texture_shape_images/larger/1.0'
    
    if hasattr(model, 'eval'):
        model.eval()
            
    mean = [0.3410, 0.3123, 0.2787]
    std = [0.2362, 0.2252, 0.2162]
    transform_list = [
        transforms.Resize((image_size, image_size))
    ]
    transform_list.append(transforms.ToTensor()) # type: ignore
    transform_list.append(transforms.Normalize(mean, std))  # type: ignore # transform it into a torch tensor
    oads_transform = transforms.Compose(transform_list)

    index_label_mapping =  {0: 'MASK',
                            1: 'Xtra Class 1',
                            2: 'Xtra Class 2',
                            3: 'Bin',
                            4: 'Compact car',
                            5: 'Scooter',
                            6: 'Bollard',
                            7: 'Balcony door',
                            8: 'Van',
                            9: 'Oma fiets',
                            10: 'Carrier bike',
                            11: 'SUV',
                            12: 'Traffic sign',
                            13: 'Lamppost',
                            14: 'Traffic light',
                            15: 'Bench',
                            16: 'Tree',
                            17: 'Front door',
                            18: 'Truck'}
    
    if use_human_classes:
        class_mapping = {
            'Carrier bike': 'Bike',
            'Oma fiets': 'Bike',
            'Compact car': 'Car',
            'SUV': 'Car',
            'Van': 'Truck',
            'Front door': 'Door',
            'Balcony door': 'Door',
        }

        index_label_mapping = {
            key: class_mapping[value] if value in class_mapping else value for key, value in index_label_mapping.items()
        }
    else:
        class_mapping = {}

    

    n_total = 0
    any_correct = 0
    texture_correct = 0
    shape_correct = 0

    per_class = {
        'n_total': {c: 0 for c in index_label_mapping.values()},
        'any_correct': {c: 0 for c in index_label_mapping.values()},
        'texture_correct': {c: 0 for c in index_label_mapping.values()},
        'shape_correct': {c: 0 for c in index_label_mapping.values()},
    }

    # model = model.to(device)

    dataset = OADSCueConflict(root=oads_root, transform=oads_transform)
    dataloader = DataLoader(dataset=dataset, batch_size=8, num_workers=8)

    responses = {}

    with tqdm(dataloader) as t:
        for item in t:
            input, image_names, idx = item
            input = input.to(device)            

            # for model_name, model in models:

            scores = model(input)
            _, predictions = scores.max(1)

            for pred, image_name in zip(predictions, image_names):

                image_name = image_name.split('/')[-1]
                shape_desc, texture_desc = image_name.split('-')
                shape_class, shape_image_id, shape_index = shape_desc.split('_')
                texture_class, texture_image_id, texture_index = texture_desc.split('_')
                texture_index = texture_index.split('.')[0]

                if texture_class in class_mapping:
                    texture_class = class_mapping[texture_class]
                
                if shape_class in class_mapping:
                    shape_class = class_mapping[shape_class]
                

                pred_class = index_label_mapping[int(pred.cpu().numpy())]

                if pred_class == shape_class:
                    any_correct += 1
                    shape_correct += 1
                    
                    if pred_class in per_class['any_correct']:
                        per_class['any_correct'][pred_class] += 1
                    else:
                        per_class['any_correct'][pred_class] = 1
                        
                    if pred_class in per_class['shape_correct']:
                        per_class['shape_correct'][pred_class] += 1
                    else:
                        per_class['shape_correct'][pred_class] = 1

                elif pred_class == texture_class:
                    any_correct += 1
                    texture_correct += 1


                    if pred_class in per_class['any_correct']:
                        per_class['any_correct'][pred_class] += 1
                    else:
                        per_class['any_correct'][pred_class] = 1

                    if pred_class in per_class['texture_correct']:
                        per_class['texture_correct'][pred_class] += 1
                    else:
                        per_class['texture_correct'][pred_class] = 1
                else:
                    pass



                n_total += 1

                if shape_class in per_class['n_total']:
                    per_class['n_total'][shape_class] += 1
                else:
                    per_class['n_total'][shape_class] = 1

                if texture_class in per_class['n_total']:
                    per_class['n_total'][texture_class] += 1
                else:
                    per_class['n_total'][texture_class] = 1

                responses[image_name] = {'prediction': pred_class, 'shape_class': shape_class, 'texture_class': texture_class, 'image_name': image_name}


    # for model_name in per_class['n_total'].keys():
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f'{model_name}_responses.yml'), 'w') as f:
        res = {
                'shape_bias': (shape_correct / any_correct) if any_correct > 0 else 0,
                'any_correct': any_correct,
                'per_class': {c: {
                                'shape_bias': (per_class['shape_correct'][c] / per_class['any_correct'][c]) if per_class['any_correct'][c] > 0 else 0,
                                'any_correct': per_class["any_correct"][c],
                                'n_total': per_class["n_total"][c],
                            } for c in per_class['n_total'].keys()
                },
                'responses': responses,
            }
        yaml.dump(data=res, stream=f)

    print(f'Save responses for {model_name}')

def assess_model_on_imagenet(model_name, model_path):
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results_journal', 'human_class_responses', 'imagenet_224x224_cue-conflict')

    batch_size = 8
    num_workers = 8
    image_size = 224

    if os.path.exists(os.path.join(result_dir, f'{model_name}_responses.yml')):
        return

    os.makedirs(result_dir, exist_ok=True)
    datasets = ["cue-conflict"]
    params = {"batch_size": batch_size, "print_predictions": True, "num_workers": num_workers, 'image_size': image_size}

    if 'fasterrcnn' in model_name:
        model_fn = fasterrcnn_resnet50_fpn_coco
    elif 'ade20k_scenes' in model_name:
        model_fn = ADE20K_ImageNetModel
    else:
        model_fn = resnet50
        
    model = load_model(model_name, model_fn, in_features=2048, state_dict_path=model_path, num_classes=1000)

    model = PytorchModel(model, model_name)

    Evaluate().assess(model_name=model_name, model=model, dataset_names=datasets, **params)


    # for model_name in models:
    with open(f'{os.path.dirname(os.path.abspath(__file__))}/../model-vs-human/raw-data/{datasets[0]}/{datasets[0]}_{model_name.replace("_", "-")}_session-1.csv', 'r') as f:
        reader = csv.reader(f)
        lines = []
        for row in reader:
            lines.append(row)

    cols = lines[0]
    rows = lines[1:]

    responses = {}

    n_total = 0
    any_correct = 0
    texture_correct = 0
    shape_correct = 0

    per_class = {
        'n_total': {},
        'any_correct': {},
        'texture_correct': {},
        'shape_correct': {},
        'accuracy': {},
    }

    for row in rows:
        # print(row)
        _, session, trial, rt, pred_class, category, condition, imagename = row
        shape_class = category
        texture_class = imagename.split('-')[-1].split('.')[0][:-1]
        if '_' in texture_class:
            texture_class = texture_class.split('_')[0]

        for _c in [shape_class, texture_class, pred_class]:
            if _c not in per_class['n_total'].keys():
                per_class['n_total'][_c] = 0
            elif _c not in per_class['any_correct'].keys():
                per_class['any_correct'][_c] = 0
            elif _c not in per_class['shape_correct'].keys():
                per_class['shape_correct'][_c] = 0
            elif _c not in per_class['texture_correct'].keys():
                per_class['texture_correct'][_c] = 0

        # print(texture_class)
        responses[imagename] = {'prediction': pred_class, 'shape_class': shape_class, 'texture_class': texture_class, 'image_name': imagename}

        if pred_class == shape_class:
            any_correct += 1
            shape_correct += 1

            if pred_class in per_class['any_correct']:
                per_class['any_correct'][pred_class] += 1
            else:
                per_class['any_correct'][pred_class] = 1
                
            if pred_class in per_class['shape_correct']:
                per_class['shape_correct'][pred_class] += 1
            else:
                per_class['shape_correct'][pred_class] = 1

        elif pred_class == texture_class:
            any_correct += 1
            texture_correct += 1

            if pred_class in per_class['any_correct']:
                per_class['any_correct'][pred_class] += 1
            else:
                per_class['any_correct'][pred_class] = 1
                
            if pred_class in per_class['texture_correct']:
                per_class['texture_correct'][pred_class] += 1
            else:
                per_class['texture_correct'][pred_class] = 1

        n_total += 1

        if shape_class in per_class['n_total']:
            per_class['n_total'][shape_class] += 1
        else:
            per_class['n_total'][shape_class] = 1

        if texture_class in per_class['n_total']:
            per_class['n_total'][texture_class] += 1
        else:
            per_class['n_total'][texture_class] = 1

    res = {
        'shape_bias': (shape_correct / any_correct) if any_correct > 0 else 0,
        'any_correct': any_correct,
        'per_class': {c: {
                        'shape_bias': (per_class['shape_correct'][c] / per_class['any_correct'][c]) if per_class['any_correct'][c] > 0 else 0,
                        'any_correct': per_class["any_correct"][c],
                        'n_total': per_class["n_total"][c],
                        'accuracy': (per_class['any_correct'][c] / per_class['n_total'][c]) if per_class['n_total'][c] > 0 else 0,
                    } for c in per_class['n_total'].keys()
        },
        'responses': responses,
        'accuracy': (any_correct / n_total) if n_total > 0 else 0
    }

    with open(os.path.join(result_dir, f'{model_name}_responses.yml'), 'w') as f:
        yaml.dump(data=res, stream=f)


if __name__ == "__main__":
    
    yml_path = './model_overview.yaml'
    
    with open(yml_path, 'r') as file:
        model_overview = yaml.load(file, Loader=yaml.FullLoader)


    # Assess native ImageNet model
    assess_model_on_imagenet('resnet50_imagenet', ResNet50_Weights.IMAGENET1K_V1.get_state_dict(False))

    for model_type in ['resnet50', 'resnet50_imagenet', 'resnet50_coco_objects', 'resnet50_ade20k_objects', 'fasterrcnn_resnet50_fpn_coco_scenes', 'resnet50_ade20k_scenes']:
        for image_representation in ['rgb', 'rgb_zoom-0', 'rgb_zoom-80', 'rgb_zoom-150', 'rgb_zoom-in-100', 'rgb_zoom-in-150']:
            
            pretraining_dataset = 'oads'
            if 'coco_objects' in model_type:
                pretraining_dataset = 'coco_objects'
            elif 'ade20k_objects' in model_type:
                pretraining_dataset = 'ade20k_objects'
            elif 'imagenet' in model_type:
                pretraining_dataset = 'imagenet'
            elif 'coco_scenes' in model_type:
                pretraining_dataset = 'coco_object_detection'
            elif 'ade20k_scenes' in model_type:
                pretraining_dataset = 'ade20k_scene_segmentation'

            zoom = image_representation.split('_')[-1] if 'zoom' in image_representation else 'original'

            if zoom not in model_overview[pretraining_dataset].keys():
                continue

            for reps in model_overview[pretraining_dataset][zoom]['reps'].keys():
                

                pretrained_model_path = model_overview[pretraining_dataset][zoom]['reps'][reps]['path'].replace('training_results', 'best_model').replace('.yml', '.pth')
                if pretraining_dataset == 'oads':
                    model_name = '_'.join([model_type, pretraining_dataset, zoom, str(reps), 'on_oads'])
                    model_path = pretrained_model_path.replace(':', '')


                    if model_path != '':
                        # continue

                        if not os.path.exists(model_path):
                            model_path = model_path.replace('best_', 'final_')

                        ## Assess shape-bias on OADS
                        try:
                            assess_model_on_oads(model_name, model_path)
                        except Exception as e:
                            print(e)
                            continue

                elif pretraining_dataset == 'imagenet':
                    model_name = '_'.join([model_type, pretraining_dataset, zoom, str(reps), 'on_subimagenet'])
                    model_path = pretrained_model_path.replace(':', '')


                    if model_path != '':
                        # continue

                        if not os.path.exists(model_path):
                            model_path = model_path.replace('best_', 'final_')

                        ## Assess shape-bias on ImageNet
                        try:
                            assess_model_on_imagenet(model_name, model_path)
                        except Exception as e:
                            print(e)
                            continue

                if 'finetuned' not in model_overview[pretraining_dataset][zoom]['reps'][reps].keys():
                    continue
                    
                if 'imagenet' in model_overview[pretraining_dataset][zoom]['reps'][reps]['finetuned'].keys():

                    for finetuning in model_overview[pretraining_dataset][zoom]['reps'][reps]['finetuned']['imagenet'].keys():
                        model_name = '_'.join([model_type, pretraining_dataset, zoom, str(reps), finetuning, 'on_subimagenet'])
                        
                        model_path = model_overview[pretraining_dataset][zoom]['reps'][reps]['finetuned']['imagenet'][finetuning].replace('training_results', 'best_model').replace('.yml', '.pth')
                        model_path = model_path.replace(':', '')


                        if model_path != '':
                            # continue

                            if not os.path.exists(model_path):
                                model_path = model_path.replace('best_', 'final_')

                            try:
                                assess_model_on_imagenet(model_name, model_path)
                            except Exception as e:
                                print(e)
                                continue

                elif 'oads' in model_overview[pretraining_dataset][zoom]['reps'][reps]['finetuned'].keys():
                    for finetuning in model_overview[pretraining_dataset][zoom]['reps'][reps]['finetuned']['oads'].keys():
                        model_name = '_'.join([model_type, pretraining_dataset, zoom, str(reps), finetuning, 'on_oads'])
                        
                        model_path = model_overview[pretraining_dataset][zoom]['reps'][reps]['finetuned']['oads'][finetuning].replace('training_results', 'best_model').replace('.yml', '.pth')
                        model_path = model_path.replace(':', '')


                        if model_path != '':
                            # continue

                            if not os.path.exists(model_path):
                                model_path = model_path.replace('best_', 'final_')

                            try:
                                assess_model_on_oads(model_name, model_path)
                            except Exception as e:
                                print(e)
                                continue