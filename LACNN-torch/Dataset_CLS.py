import os
import optparse as op
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
# from models.mEfficientNet import EfficientNet
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from glob import glob


name2label = {'CNV': 0, 'DRUSEN': 1, 'DME': 2, 'NORMAL': 3}


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val]
    return weight


def config_dataset(img_size, batch_size, train_dir, test_dir, valid_dir, save_dir):

    transform_train = transforms.Compose([
        transforms.RandomAffine(degrees=[-15, 15], translate=(0.2, 0.2), scale=(0.8, 1.2), shear=(0, 0, 0, 0), fill=0),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(size=[img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test_val = transforms.Compose([
        transforms.Resize(size=[img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_image_files = sorted(glob(train_dir + '/*/*.jpeg'))
    all_labels = [name2label[path.split('/')[-2]] for path in all_image_files]
    file_label_train, file_label_test = train_test_split(all_image_files, all_labels, save_dir)

    files_valid = sorted(glob(valid_dir + '/*/*.jpeg'))
    labels_valid = [name2label[path.split('/')[-2]] for path in files_valid]

    # For unbalanced dataset we create a weighted sampler
    weights = make_weights_for_balanced_classes(file_label_train[1], 4)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_dataset = OCT_Dataset(data=file_label_train, transform=transform_train)
    test_dataset = OCT_Dataset(data=file_label_test, transform=transform_test_val)
    valid_dataset = OCT_Dataset(data=(files_valid, labels_valid), transform=transform_test_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler, drop_last=False, num_workers=8)    #shuffle=True
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)

    print('Number of train images: {}'.format(len(train_dataset)))
    print('Number of test images: {}'.format(len(test_dataset)))
    print('Number of valid images: {}'.format(len(valid_dataset)))
    print('')

    return train_loader, test_loader, valid_loader


def train_test_split(all_image_files, all_labels, save_dir):
    all_image_files = np.array(all_image_files)
    all_labels = np.array(all_labels)

    cnv_image_files = all_image_files[all_labels == 0]
    drusen_image_files = all_image_files[all_labels == 1]
    dme_image_files = all_image_files[all_labels == 2]
    normal_image_files = all_image_files[all_labels == 3]

    cnv_labels = all_labels[all_labels == 0]
    drusen_labels = all_labels[all_labels == 1]
    dme_labels = all_labels[all_labels == 2]
    normal_labels = all_labels[all_labels == 3]

    test_split = []
    test_label = []
    train_split = []
    train_label = []
    train_fraction = 1 / 6.
    current_train = 6

    train_num = int(len(normal_image_files) * train_fraction)
    train_index = range(train_num * (current_train - 1), train_num * current_train)
    test_index = sorted(set(range(len(normal_image_files))) - set(train_index))
    test_split.extend(normal_image_files[test_index])
    train_split.extend(normal_image_files[train_index])
    test_label.extend(normal_labels[test_index])
    train_label.extend(normal_labels[train_index])

    train_num = int(len(drusen_image_files) * train_fraction)
    train_index = range(train_num * (current_train - 1), train_num * current_train)
    test_index = sorted(set(range(len(drusen_image_files))) - set(train_index))
    test_split.extend(drusen_image_files[test_index])
    train_split.extend(drusen_image_files[train_index])
    test_label.extend(drusen_labels[test_index])
    train_label.extend(drusen_labels[train_index])

    train_num = int(len(cnv_image_files) * train_fraction)
    train_index = range(train_num * (current_train - 1), train_num * current_train)
    test_index = sorted(set(range(len(cnv_image_files))) - set(train_index))
    test_split.extend(cnv_image_files[test_index])
    train_split.extend(cnv_image_files[train_index])
    test_label.extend(cnv_labels[test_index])
    train_label.extend(cnv_labels[train_index])

    train_num = int(len(dme_image_files) * train_fraction)
    train_index = range(train_num * (current_train - 1), train_num * current_train)
    test_index = sorted(set(range(len(dme_image_files))) - set(train_index))
    test_split.extend(dme_image_files[test_index])
    train_split.extend(dme_image_files[train_index])
    test_label.extend(dme_labels[test_index])
    train_label.extend(dme_labels[train_index])

    if save_dir is not None:
        with open(os.path.join(save_dir, 'OCT_train_' + str(current_train) + ".txt"), 'w') as f:
            for i in train_split:
                f.write(i + '\n')

        with open(os.path.join(save_dir, 'OCT_test_' + str(current_train) + ".txt"), 'w') as f:
            for i in test_split:
                f.write(i + '\n')

    return (train_split, train_label), (test_split, test_label)


class OCT_Dataset(Dataset):
    """Image and label dataset."""

    def __init__(self, data, transform=None):
        """
        Args:
            data (tuple): file with label.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):

        path = self.data[0][index]
        label = self.data[1][index]

        img = Image.open(path)
        img = img.convert('RGB')

        image_tensor = self.transform(img)

        return image_tensor, torch.tensor(label).long(), path
