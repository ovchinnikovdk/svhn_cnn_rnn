from torch.utils.data import Dataset
import torch
import torchvision
import h5py
import cv2
import os
import numpy as np


class HouseNumberTrainDataset(Dataset):
    def __init__(self, path, hdf5_path, transform_func=None):
        super(HouseNumberTrainDataset, self).__init__()
        self.meta_file = h5py.File(hdf5_path)
        self.path = path
        if callable(transform_func):
            self.transform = transform_func()
        else:
            self.transform = None

    def __len__(self):
        # return min(12000, len(self.meta_file['digitStruct/name']))
        return len(self.meta_file['digitStruct/name'])

    def __getitem__(self, idx):
        filename = self._get_name(idx)
        labels = self._get_labels(idx)
        img = cv2.imread(os.path.join(self.path, filename))
        img = cv2.resize(img, (128, 128))
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torchvision.transforms.ToTensor()(img)
        lab = torch.zeros(8, 12)
        for i in range(len(labels), 8):
            lab[i][-1] = 1
        for i, l in enumerate(labels):
            lab[i][int(l)] = 1
        return (img, lab), lab

    def _get_name(self, index):
        name = self.meta_file['/digitStruct/name']
        return ''.join([chr(v[0]) for v in self.meta_file[name[index][0]].value])

    def _get_labels(self, index):
        item = self.meta_file['digitStruct']['bbox'][index].item()
        label = self.meta_file[item]['label']
        return [self.meta_file[label.value[i].item()].value[0][0]
                for i in range(len(label))] if len(label) > 1 else [label.value[0][0]]


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def get_loader(path, hdf5_path, transform_func=None, batch_size=64, shuffle=True):
    dataset = HouseNumberTrainDataset(path, hdf5_path, transform_func)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle)
                                              # num_workers=num_workers)
                                              # collate_fn=collate_fn)
    return data_loader
