import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from toolbox.datasets.augmentations import Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale


scales_range = '0.5 2.0'
city_crop_size = '1280 720'
brightness = 0.5
contrast = 0.5
saturation = 0.5
p = 0.5


class Cityscapes(Dataset):

    def __init__(self, root='', mode='train'):

        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        scale_range = tuple(float(i) for i in scales_range.split(' '))
        crop_size = tuple(int(i) for i in city_crop_size.split(' '))

        self.aug = Compose([
            ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation),
            RandomHorizontalFlip(p),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])

        
        # The values associated with the 35 classes
        self.full_classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                             17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                             32, 33, -1)
        # The values above are remapped to the following
        self.new_classes = (0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 5, 0, 0, 0, 6, 0, 7,
                            8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 17, 18, 19, 0)

        self.class_weight = np.array([9.36879574, 3.05247989, 14.15357162, 5.58607574, 37.43976198, 36.06995197,
                                      32.03698029, 46.44198507, 40.61829658, 7.04938217, 33.43519928, 20.94698721,
                                      29.2455243, 45.64950437, 11.10225935, 43.37297876, 45.06890502, 45.21967197,
                                      47.53300041, 41.05393685, ])
        
        self.mode = mode
        self.root = root
        self.files = {}
        self.images_base = os.path.join(self.root, 'leftImg8bit', self.mode)
        self.annotations_base = os.path.join(self.root, 'gtFine', self.mode)
        self.files[mode] = self.recursive_glob(rootdir=self.images_base, suffix='.png')
      
    def __len__(self):
        return len(self.files[self.mode])

    def __getitem__(self, index):

        img_path = self.files[self.mode][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')

        image = Image.open(img_path).convert('RGB')
        lbl = Image.open(lbl_path)
        label = self.remap(lbl, self.full_classes, self.new_classes)

        sample = {
            'image': image,
            'label': label,
        }

        if self.mode == 'train':  # 只对训练集增强
            sample = self.aug(sample)

        sample['image'] = self.im_to_tensor(sample['image'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()

        return sample

    def recursive_glob(self, rootdir='.', suffix=''):
        """使用给定的后缀和根目录执行递归
            :param rootdir 根目录
            :param suffix 要搜索的后缀
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]
    
    # 35类别转化为19类别
    def remap(self, image, old_values, new_values):
        assert isinstance(image, Image.Image) or isinstance(
            image, np.ndarray), "image must be of type PIL.Image or numpy.ndarray"
        assert type(new_values) is tuple, "new_values must be of type tuple"
        assert type(old_values) is tuple, "old_values must be of type tuple"
        assert len(new_values) == len(
            old_values), "new_values and old_values must have the same length"

        # If image is a PIL.Image convert it to a numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Replace old values by the new ones
        tmp = np.zeros_like(image)
        for old, new in zip(old_values, new_values):
            # Since tmp is already initialized as zeros we can skip new values
            # equal to 0
            if new != 0:
                tmp[image == old] = new

        return Image.fromarray(tmp)
