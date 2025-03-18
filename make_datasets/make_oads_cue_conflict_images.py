import os
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.autograd import Variable

from torchvision.models import resnet50
from PIL import Image

from base.oads_style_transfer import PytorchNeuralStyleTransfer

def get_oads_model(device):
    n_input_channels = 3
    output_channels = 19

    model_path = 'trained_models/oads/zoom-0/rep-0/best_model_29-08-23-160144.pth'
    model = resnet50()
    model.conv1 = nn.Conv2d(in_channels=n_input_channels, out_channels=model.conv1.out_channels, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
    model.fc = nn.Linear(
        in_features=2048, out_features=output_channels, bias=True)

    model = DataParallel(model,)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device=device)

    model.eval()

    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    if not torch.cuda.is_available():
        exit(1)

    home_path = os.path.expanduser('~')

    style_weight_factor = 1.0

    result_dir = f'{home_path}/projects/oads_texture_shape_images_zoom-0/{str(style_weight_factor)}'
    os.makedirs(result_dir, exist_ok=True)


    basedir = f'{home_path}/projects/data/oads/'

    # mean and std of the OADS dataset
    mean = [0.3410, 0.3123, 0.2787]
    std = [0.2362, 0.2252, 0.2162]

    imsize = (400, 400)  # use small size if no gpu

    model = get_oads_model(device=device)

    filenames = {}

    with open('filenames.yaml', 'r') as f:
        filenames = yaml.safe_load(f)
    with open('remaining_filenames.yaml', 'r') as f:
        remaining_filenames = yaml.safe_load(f)


    filenames.update(remaining_filenames)

    style_transfer = PytorchNeuralStyleTransfer(img_size=imsize, device=device, mean=mean)

    style_transfer.weights = [x * style_weight_factor if i < len(style_transfer.weights)-1 else x for i, x in enumerate(style_transfer.weights)]

    verbose = False
    save_to_file = True
    # results = {}

    max_n_images = 10

    counter = 0
    for c, tuples in tqdm(filenames.items(), desc='Classes'):

        # results[c] = {}
        for image, index in tqdm(tuples, desc='Images', leave=False):

            # results[c][f'{image}_{index}'] = {}

            for other_c, other_tuples in tqdm(filenames.items(), leave=False, desc='Other classes'):
                counter = 0
                # results[c][f'{image}_{index}'][other_c] = {}

                if other_c == c:
                    continue
                for other_image, other_index in tqdm(other_tuples, leave=False, desc='Other images'):
                    if os.path.exists(os.path.join(result_dir, f'{c}_{image}_{index}-{other_c}_{other_image}_{other_index}.png')):
                        continue

                    if counter > max_n_images:
                        break
                    
                    # results[c][f'{image}_{index}'][other_c][f'{other_image}_{other_index}'] = None
                    
                    img = Image.open(os.path.join(basedir, 'oads_arw', 'crops', 'ML', c, f'{image}_{index}.tiff'))
                    texture = Image.open(os.path.join(basedir, 'oads_arw', 'crops', 'ML', other_c, f'{other_image}_{other_index}.tiff'))

                    style_image = Variable(style_transfer.prep(texture).unsqueeze(0)).to(device)
                    content_image = Variable(style_transfer.prep(img).unsqueeze(0)).to(device)

                    output = style_transfer.run(style_image=style_image, content_image=content_image, max_iter=500, verbose=verbose)
                    
                    
                    # results[c][f'{image}_{index}'][other_c][f'{other_image}_{other_index}'] = output
                    
                    counter += 1
                    
                    if save_to_file:
                        output.save(os.path.join(result_dir, f'{c}_{image}_{index}-{other_c}_{other_image}_{other_index}.png'))


    print("Done creating images.")