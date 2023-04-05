import os

import torch
import khandy

__all__ = ['RecordDataset', 'ImageAndLabelDataset', 'PathImageAndLabelDataset', 'PathAndImageDataset']


class RecordDataset(torch.utils.data.Dataset):
    def __init__(self, filename, transform=None):
        self.records = khandy.load_list(filename)
        self.transform = transform
        
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        return self.transform(record)
    

class ImageAndLabelDataset(torch.utils.data.Dataset):
    def __init__(self, filename_or_else, transform=None, label_transform=int):
        if isinstance(filename_or_else, str):
            self.records = khandy.load_list(filename_or_else)
        elif isinstance(filename_or_else, (list, tuple)):
            self.records = filename_or_else
        else:
            raise ValueError('unsupported type!')
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        path, label = record.rsplit(',', maxsplit=1)
        image = khandy.imread_pil(path, 'RGB')
        if self.transform:
            image = self.transform(image)
        if self.label_transform is not None:
            label = self.label_transform(label)
        return image, label


class PathImageAndLabelDataset(torch.utils.data.Dataset):
    def __init__(self, filename_or_else, transform=None):
        if isinstance(filename_or_else, str):
            self.records = khandy.load_list(filename_or_else)
        elif isinstance(filename_or_else, (list, tuple)):
            self.records = filename_or_else
        else:
            raise ValueError('unsupported type!')
        self.transform = transform

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        parts = record.split(',')
        path = ','.join(parts[:-1])
        label = int(parts[-1])
        image = khandy.imread_pil(path, 'RGB')
        if self.transform:
            image = self.transform(image)
        return path, image, label


class PathAndImageDataset(torch.utils.data.Dataset):
    def __init__(self, filename_or_else, transform=None):
        if isinstance(filename_or_else, str):
            if os.path.isfile(filename_or_else):
                self.paths = khandy.load_list(filename_or_else)
            elif os.path.isdir(filename_or_else):
                self.paths = khandy.get_all_filenames(filename_or_else)
            else:
                raise ValueError('file or dir does not exist!')
        elif isinstance(filename_or_else, (list, tuple)):
            self.paths = filename_or_else
        else:
            raise ValueError('unsupported type!')
        self.transform = transform

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        image = khandy.imread_pil(path, 'RGB')
        if self.transform:
            image = self.transform(image)
        return path, image
        
        