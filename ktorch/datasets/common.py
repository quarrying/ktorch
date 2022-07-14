import os

import torch
import khandy

__all__ = ['ImageAndLabelDataset', 'PathImageAndLabelDataset', 'PathAndImageDataset']


class ImageAndLabelDataset(torch.utils.data.Dataset):
    def __init__(self, filename, transform=None):
        self.records = khandy.load_list(filename)
        self.transform = transform

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        parts = record.split(',')
        path = ','.join(parts[:-1])
        label = int(parts[-1])
        image = khandy.imread_pil(path)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class PathImageAndLabelDataset(torch.utils.data.Dataset):
    def __init__(self, filename, transform=None):
        self.records = khandy.load_list(filename)
        self.transform = transform

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        parts = record.split(',')
        path = ','.join(parts[:-1])
        label = int(parts[-1])
        image = khandy.imread_pil(path)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return path, image, label


class PathAndImageDataset(torch.utils.data.Dataset):
    def __init__(self, filename, transform=None):
        if isinstance(filename, str):
            if os.path.isfile(filename):
                self.records = khandy.load_list(filename)
            elif os.path.isdir(filename):
                self.records = khandy.get_all_filenames(filename)
            else:
                raise ValueError('file or dir does not exist!')
        elif isinstance(filename, (list, tuple)):
            self.records = filename
        else:
            raise ValueError('unsupported type!')
        self.transform = transform

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        path = self.records[idx]
        image = khandy.imread_pil(path)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return path, image
        
        