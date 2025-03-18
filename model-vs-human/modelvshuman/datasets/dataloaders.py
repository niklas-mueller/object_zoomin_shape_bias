import torch
from torchvision import transforms
import torchvision.datasets as datasets
from . import info_mappings
import numpy as np
from PIL import Image

def rgb_to_opponent_space(img, normalize=False):
    """rgb_to_opponent_space

    Convert a image in RBG space to color opponent space, i.e., Intensity, Blue-Yellow (BY) opponent, Red-Green (RG) opponent.

    Parameters
    ----------
    img: ndarray, list, PIL Image
        Image to be converted
    normalize: bool, optional
        Whether to normalize pixel values by the maximum, by default False

    Returns
    ----------
    ndarray
        Array of length 3, with image converted to Intensity, BY, RG opponent, respectively.

    Example
    ----------
    >>> 
    """
    o1 = 0.3 * img[:, :, 0] + 0.58 * img[:, :, 1] + \
        0.11 * img[:, :, 2]   # Intensity/Luminance
    o2 = 0.25 * img[:, :, 0] + 0.25 * img[:, :, 1] - \
        0.5 * img[:, :, 2]   # BY opponent
    o3 = 0.5 * img[:, :, 0] - 0.5 * \
        img[:, :, 1]                        # RG opponent

    if normalize:
        ret = []
        for _x in [o1, o2, o3]:
            _max = _x.max()
            ret.append(_x / _max)
        return np.array(ret)

    return np.array([o1, o2, o3])

class ToOpponentChannel(object):

    def __init__(self, dtype=np.float32):
        super().__init__()
        self.dtype = dtype

    def __call__(self, sample):
        ops = rgb_to_opponent_space(np.array(sample, dtype=np.uint8))

        return ops.transpose((1,2,0)).astype(self.dtype)
    

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder

    Adapted from:
    https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    def __init__(self, *args, **kwargs):

        if "info_mapping" in kwargs.keys():
            self.info_mapping = kwargs["info_mapping"]
            del kwargs["info_mapping"]
        else:
            self.info_mapping = info_mappings.ImageNetInfoMapping()

        super(ImageFolderWithPaths, self).__init__(*args, **kwargs)


    def __getitem__(self, index):
        """override the __getitem__ method. This is the method that dataloader calls."""
        # this is what ImageFolder normally returns
        (sample, target) = super(ImageFolderWithPaths, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]
        _, _, _, new_target = self.info_mapping(path)
        original_tuple = (sample, new_target)

        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class PytorchLoader(object):
    """Pytorch Data loader"""

    def __call__(self, path, resize, batch_size, num_workers,
                 info_mapping=None, grayscale:bool=None):
        """
        Data loader for pytorch models
        :param path:
        :param resize:
        :param batch_size:
        :param num_workers:
        :return:
        """
        if grayscale:
            normalize = transforms.Normalize(mean=[0.5,],
                                            std=[0.5,])
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

        print(resize)
        if resize:
            transform_list = []
            if grayscale:
                # transform_list.append(transforms.Grayscale())
                transform_list.append(transforms.Lambda(lambd=lambda img: img.convert('L').resize((224, 224), Image.ANTIALIAS)))
            else:
                transform_list.append(transforms.Resize((384, 384)))
                # transform_list.append(transforms.Resize(112))
                # transform_list.append(transforms.Resize(256))
                # transform_list.append(transforms.CenterCrop(224))
            transform_list.append(transforms.ToTensor())
            transform_list.append(normalize)
            transformations = transforms.Compose(transform_list)
            # transformations = transforms.Compose([
            #     transforms.Resize(256),
            #     transforms.CenterCrop(224),
            #     # ToOpponentChannel(),
            #     transforms.ToTensor(),
            #     normalize,
            # ])
        else:
            transform_list = []
            if grayscale:
                # transform_list.append(transforms.Grayscale())
                transform_list.append(transforms.Lambda(lambd=lambda img: img.convert('L')))
            transform_list.append(transforms.ToTensor())
            transform_list.append(normalize)
            transformations = transforms.Compose(transform_list)
            # transformations = transforms.Compose([
            #     # ToOpponentChannel(),
            #     transforms.ToTensor(),
            #     normalize,
            # ])

        loader = torch.utils.data.DataLoader(
            ImageFolderWithPaths(path, transformations,
                                 info_mapping=info_mapping),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        return loader
