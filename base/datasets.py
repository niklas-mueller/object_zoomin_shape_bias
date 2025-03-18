import numpy as np
import torch
from PIL import Image
import os
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import json
from tqdm import tqdm

class OADS_Objects(Dataset):
    def __init__(self, root, item_ids: list, transform=None, target_transform=None,
                 device='cuda:0', target: str = 'label', return_index:bool=False) -> None:
        super().__init__()

        # self.oads_access = oads_access
        self.root = root
        self.item_ids = item_ids

        self.transform = transform
        self.target_transform = target_transform
        self.device = device

        self.target = target

        self.return_index = return_index

        self.label_index_mapping =  {
                            'MASK': 0,
                            'Xtra Class 1': 1,
                            'Xtra Class 2': 2,
                            'Bin': 3,
                            'Compact car': 4,
                            'Scooter': 5,
                            'Bollard': 6,
                            'Balcony door': 7,
                            'Van': 8,
                            'Oma fiets': 9,
                            'Carrier bike': 10,
                            'SUV': 11,
                            'Traffic sign': 12,
                            'Lamppost': 13,
                            'Traffic light': 14,
                            'Bench': 15,
                            'Tree': 16,
                            'Front door': 17,
                            'Truck': 18,
                            }

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx):
        image_name, label = self.item_ids[idx]
        crop_path = os.path.join(self.root, image_name)
        crop = Image.open(crop_path)

        tup = (crop, label)


        if tup is None:
            return None

        img, label = tup
        del tup

        if img is None or label is None:
            return None

        if self.transform:
            img = self.transform(img)

        img = img.float()

        label = self.label_index_mapping[label]
        # if self.target == 'label':
        #     label = label['classId']
        #     if self.class_index_mapping is not None:
        #         label = self.class_index_mapping[label]

        # elif self.target == 'image':
        #     label = img

        # else:
        #     label = np.array([])

        if self.target_transform:
            label = self.target_transform(label)

        if self.return_index:
            return (img, label, idx)
        
        return (img, label)


class ImageNetCueConflict(Dataset):
    def __init__(self, root, split, transform=None, return_index:bool=True, root_extension="ILSVRC/Data/CLS-LOC", val_label_filepath=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        self.syn_to_classname = {}

        self.return_index = return_index

        self.use_classes =   [
                            'n03041632', 'n03085013', 'n04505470', 'n02504013', 'n02504458', 'n02835271', 'n03792782', 'n02690373', 'n03955296', 'n13861050', 'n13941806', 'n02708093', 'n03196217', 'n04548280', 
                            'n03259401', 'n04111414', 'n04111531', 'n02791124', 'n03376595', 'n04099969', 'n00605023', 'n04429376', 'n02132136', 'n02133161', 'n02134084', 'n02134418', 'n02951358', 'n03344393', 
                            'n03662601', 'n04273569', 'n04612373', 'n04612504', 'n02823428', 'n03937543', 'n03983396', 'n04557648', 'n04560804', 'n04579145', 'n04591713', 'n03345487', 'n03417042', 'n03770679', 
                            'n03796401', 'n00319176', 'n01016201', 'n03930630', 'n03930777', 'n05061003', 'n06547832', 'n10432053', 'n03977966', 'n04461696', 'n04467665', 'n02814533', 'n03100240', 'n03100346', 
                            'n13419325', 'n04285008', 'n01321123', 'n01514859', 'n01792640', 'n07646067', 'n01530575', 'n01531178', 'n01532829', 'n01534433', 'n01537544', 'n01558993', 'n01562265', 'n01560419', 
                            'n01582220', 'n10281276', 'n01592084', 'n01601694', 'n01614925', 'n01616318', 'n01622779', 'n01795545', 'n01796340', 'n01797886', 'n01798484', 'n01817953', 'n01818515', 'n01819313', 
                            'n01820546', 'n01824575', 'n01828970', 'n01829413', 'n01833805', 'n01843065', 'n01843383', 'n01855032', 'n01855672', 'n07646821', 'n01860187', 'n02002556', 'n02002724', 'n02006656', 
                            'n02007558', 'n02009229', 'n02009912', 'n02011460', 'n02013706', 'n02017213', 'n02018207', 'n02018795', 'n02025239', 'n02027492', 'n02028035', 'n02033041', 'n02037110', 'n02051845', 
                            'n02056570', 'n02085782', 'n02085936', 'n02086079', 'n02086240', 'n02086646', 'n02086910', 'n02087046', 'n02087394', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02088632', 
                            'n02089078', 'n02089867', 'n02089973', 'n02090379', 'n02090622', 'n02090721', 'n02091032', 'n02091134', 'n02091244', 'n02091467', 'n02091635', 'n02091831', 'n02092002', 'n02092339', 
                            'n02093256', 'n02093428', 'n02093647', 'n02093754', 'n02093859', 'n02093991', 'n02094114', 'n02094258', 'n02094433', 'n02095314', 'n02095570', 'n02095889', 'n02096051', 'n02096294', 
                            'n02096437', 'n02096585', 'n02097047', 'n02097130', 'n02097209', 'n02097298', 'n02097474', 'n02097658', 'n02098105', 'n02098286', 'n02099267', 'n02099429', 'n02099601', 'n02099712', 
                            'n02099849', 'n02100236', 'n02100583', 'n02100735', 'n02100877', 'n02101006', 'n02101388', 'n02101556', 'n02102040', 'n02102177', 'n02102318', 'n02102480', 'n02102973', 'n02104029', 
                            'n02104365', 'n02105056', 'n02105162', 'n02105251', 'n02105505', 'n02105641', 'n02105855', 'n02106030', 'n02106166', 'n02106382', 'n02106550', 'n02106662', 'n02107142', 'n02107312', 
                            'n02107574', 'n02107683', 'n02107908', 'n02108000', 'n02108422', 'n02108551', 'n02108915', 'n02109047', 'n02109525', 'n02109961', 'n02110063', 'n02110185', 'n02110627', 'n02110806', 
                            'n02110958', 'n02111129', 'n02111277', 'n08825211', 'n02111500', 'n02112018', 'n02112350', 'n02112706', 'n02113023', 'n02113624', 'n02113712', 'n02113799', 'n02113978'
                        ]

        if val_label_filepath is None:
            val_label_filepath = root
        with open(os.path.join(val_label_filepath, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            # class_ids_counter = 0
            for class_id, v in json_file.items():
                # if self.use_classes is None or v[0] in self.use_classes:
                self.syn_to_class[v[0]] = int(class_id)
                # class_ids_counter += 1
                self.syn_to_classname[v[0]] = v[1]
        with open(os.path.join(val_label_filepath, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, root_extension, split)
        sample_filenames = os.listdir(samples_dir)
    
        for entry in tqdm(sample_filenames, total=len(sample_filenames)):
            if split == "train":
                syn_id = entry
                if self.use_classes is None or syn_id in self.use_classes:
                    target = self.syn_to_class[syn_id]
                    syn_folder = os.path.join(samples_dir, syn_id)
                    for sample in os.listdir(syn_folder):
                        sample_path = os.path.join(syn_folder, sample)
                        self.samples.append(sample_path)
                        self.targets.append(target)


            elif split == "val" or split == 'validation':
                syn_id = self.val_to_syn[entry]
                if self.use_classes is None or syn_id in self.use_classes:
                    target = self.syn_to_class[syn_id]
                    sample_path = os.path.join(samples_dir, entry)
                    self.samples.append(sample_path)
                    self.targets.append(target)  

                
    def __len__(self):
        return len(self.samples)    
    
    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        x = Image.open(sample_path).convert("RGB")

        if self.transform:
            x = self.transform(x)
        
        if self.return_index:
            return x, self.targets[idx], idx
        
        return x, self.targets[idx]

class ADE20K_Objects(Dataset):
    def __init__(self, images, other_images=None, transform=None, class_to_index=None, zoom_fraction=0.0):
        self.images = images
        self.transform = transform
        self.zoom_fraction = zoom_fraction

        self.index_to_key = {index: key for index, key in enumerate(self.images.keys())}

        self.targets = [info['label'] for info in self.images.values()]

        if other_images is not None:
            self.other_targets = [info['label'] for info in other_images.values()]
            self.classes = list(set(self.targets + self.other_targets))
        else:
            self.classes = list(set(self.targets))

        if class_to_index is None:
            self.class_to_index = {label: index for index, label in enumerate(self.classes)}
        else:
            self.class_to_index = class_to_index

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        key = self.index_to_key[idx]
        info = self.images[key]
        image = Image.open(info['path'].replace('.json', '.jpg')).convert('RGB')

        if self.zoom_fraction > 0.0:
            left, top, right, bottom = info['bbox']
            width = right - left
            height = bottom - top

            margin = ((width + height) / 2) * self.zoom_fraction
            
            crop_area = [left - margin, top - margin, right + margin, bottom + margin]

            if crop_area[0] < 0:
                crop_area[0] = 0
            if crop_area[1] < 0:
                crop_area[1] = 0
            if crop_area[2] > image.size[0]:
                crop_area[2] = image.size[0]
            if crop_area[3] > image.size[1]:
                crop_area[3] = image.size[1]
        else:
            crop_area = info['bbox']
        
        image = image.crop(crop_area)

        if self.transform:
            image = self.transform(image)

        label = self.targets[idx]
        label = self.class_to_index[label]

        return image, label

class CocoScenes(Dataset):
    def __init__(self, root, annotation, transforms=None, fraction_ids:float=1.0):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

        idx_max = int(len(self.ids) * fraction_ids)
        self.ids = self.ids[:idx_max]

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # print(f'image: {img.size}')

        # number of objects in the image
        num_objs = len(coco_annotation)

        if num_objs == 0:
            return img, None

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]

            if xmin == xmax or ymin == ymax:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(coco_annotation[i]['category_id'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        # print(f'tensor: {img.shape}')
        return img, my_annotation

    def __len__(self):
        return len(self.ids)



class COCO_Objects(Dataset):
    def __init__(self, images, other_images=None, transform=None, class_to_index=None, zoom_fraction=0.0):
        self.images = images
        self.transform = transform
        self.zoom_fraction = zoom_fraction

        self.index_to_key = {index: key for index, key in enumerate(self.images.keys())}

        self.targets = [info['label'] for info in self.images.values()]

        if other_images is not None:
            self.other_targets = [info['label'] for info in other_images.values()]
            self.classes = list(set(self.targets + self.other_targets))
        else:
            self.classes = list(set(self.targets))

        if class_to_index is None:
            self.class_to_index = {label: index for index, label in enumerate(self.classes)}
        else:
            self.class_to_index = class_to_index

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        key = self.index_to_key[idx]
        info = self.images[key]
        image = Image.open(info['path']).convert('RGB')

        if self.zoom_fraction > 0.0:
            left, top, right, bottom = info['bbox']
            width = right - left
            height = bottom - top

            margin = ((width + height) / 2) * self.zoom_fraction
            
            crop_area = [left - margin, top - margin, right + margin, bottom + margin]

            if crop_area[0] < 0:
                crop_area[0] = 0
            if crop_area[1] < 0:
                crop_area[1] = 0
            if crop_area[2] > image.size[0]:
                crop_area[2] = image.size[0]
            if crop_area[3] > image.size[1]:
                crop_area[3] = image.size[1]
        else:
            crop_area = info['bbox']
        
        image = image.crop(crop_area)

        if self.transform:
            image = self.transform(image)

        label = self.targets[idx]
        label = self.class_to_index[label]

        return image, label


class OADSCueConflict(Dataset):
    def __init__(self, root, transform) -> None:
        super().__init__()

        self.root = root
        self.image_names = [x for x in os.listdir(root) if x.endswith('.png') or os.path.isdir(os.path.join(root, x))]

        # If there are subfolder present, fill them accordingly
        for i in range(len(self.image_names)):
            if os.path.isdir(os.path.join(root, self.image_names[i])):
                c = self.image_names[i]
                c_image_names = os.listdir(os.path.join(root, c))
                self.image_names[i] = os.path.join(c, c_image_names[0])
                for j in range(1, len(c_image_names)):
                    if c_image_names[j].endswith('.png'):
                        self.image_names.append(os.path.join(c, c_image_names[j]))

        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.image_names[idx])).convert("RGB")
        img = self.transform(img)
        return img, self.image_names[idx], idx