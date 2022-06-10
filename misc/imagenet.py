import torch
import torchvision
import os
from torchvision.io import read_image
import PIL
import pandas


class ImageNet(torch.utils.data.Dataset):
    """
        Homemade ImageNet dataset class to handle the ILSVRC2012 dataset
        dataset: [https://www.kaggle.com/samfc10/ilsvrc2012-validation-set]
     """

    def __init__(self, root_dir, transform=None, target_transform=None):
        self.f_name = 'ILSVRC2012_val_'
        self.root_dir = root_dir
        self.transform = transform
        self.img_labels = pandas.read_csv(os.path.join(root_dir, 'Labels', 'labels.txt'))
        self.fill = 8
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return 50000

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = idx + 1
        item_str = str(item).zfill(self.fill)
        img_name = os.path.join(self.root_dir, 'Images', 'imagenet', self.f_name + item_str + '.JPEG')
        image = PIL.Image.open(img_name).convert('RGB')
        label = self.img_labels.iloc[idx, 0]-1

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
