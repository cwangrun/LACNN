import os
from PIL import Image
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from glob import glob
from torchvision.transforms import functional as transF
import numpy as np
import matplotlib.pyplot as plt


class LesionDataset(data.Dataset):
    """ image dataset
    img_root:    image root (root which contain images)
    label_root:  label root (root which contains labels)
    transform:   pre-process for image and label
    """

    def __init__(self, img_dir, img_size, mode='train'):
        self.image_paths = glob(img_dir + '/*.jpeg')
        self.img_size = img_size
        self.mode = mode

    def __getitem__(self, item):

        image_path = self.image_paths[item]
        label_path = image_path.replace('.jpeg', '_mask.png')

        image_PIL = Image.open(image_path).convert('RGB')
        label_PIL = Image.open(label_path).convert('L')

        if self.mode == 'train':
            trans_param = transforms.RandomAffine(degrees=[-15, 15]).get_params(degrees=[-15, 15],
                                                                                translate=[0.2, 0.2],
                                                                                scale_ranges=[0.8, 1.2],
                                                                                shears=[0, 0, 0, 0],
                                                                                img_size=[image_PIL.size[0],
                                                                                          image_PIL.size[1]])
            image_transform = transF.affine(image_PIL, *trans_param, fill=0)
            label_transform = transF.affine(label_PIL, *trans_param, fill=0)
            # print(np.unique(np.array(label_transform)))

            # plt.imshow(np.array(image_transform), 'gray')
            # plt.show()
            # plt.imshow(np.array(label_transform), 'gray')
            # plt.show()

            if np.random.choice(['True', 'False']):
                image_transform = transF.hflip(image_transform)
                label_transform = transF.hflip(label_transform)
        else:
            image_transform = image_PIL
            label_transform = label_PIL

        # normalize
        resize_normalize = transforms.Compose([
            transforms.Resize([self.img_size, self.img_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        t_resize_normalize = transforms.Compose([
            transforms.Resize((self.img_size // 2, self.img_size // 2)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.round(x))  # TODO: it maybe unnecessary
        ])

        image_normalize = resize_normalize(image_transform)
        label_normalize = t_resize_normalize(label_transform)

        # plt.imshow(image_normalize[0].numpy(), 'gray')
        # plt.show()
        # plt.imshow(label_normalize[0].numpy(), 'gray')
        # plt.show()

        return image_normalize, label_normalize, label_path

    def __len__(self):
        return len(self.image_paths)


def get_loader(img_dir, img_size, batch_size, mode):
    dataset = LesionDataset(img_dir, img_size, mode=mode)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return data_loader
