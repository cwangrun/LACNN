import torch
import torch.nn as nn
import torch.nn.functional as F
from LACNN_vgg import LACNN_vgg11_features, LACNN_vgg13_features, LACNN_vgg16_features, LACNN_vgg19_features
import LDN
import matplotlib.pyplot as plt
import numpy as np


base_architecture_to_features = {
                                 'vgg11': LACNN_vgg11_features,
                                 'vgg13': LACNN_vgg13_features,
                                 'vgg16': LACNN_vgg16_features,
                                 'vgg19': LACNN_vgg19_features,
                                }

class LACNN(nn.Module):
    def __init__(self, features, num_classes, LDN_model_path='None'):
        super(LACNN, self).__init__()
        self.num_classes = num_classes
        self.LDN_model_path = LDN_model_path
        self.features = features
        self.fc_layer = nn.Linear(512, self.num_classes)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)

        # pre-trained LDN
        self.LDN = LDN.build_model()
        self.LDN.load_state_dict(torch.load(self.LDN_model_path))
        self.LDN.eval()
        for p in self.LDN.parameters():
            p.requires_grad = False

    def forward(self, x):
        atten = self.LDN(x)
        conv_features = self.features(x, atten)
        conv_features = self._avg_pooling(conv_features).flatten(start_dim=1)
        logits = self.fc_layer(conv_features)
        return logits


def build_model(base_architecture, num_classes=4, LDN_model_path=None):
    features = base_architecture_to_features[base_architecture](pretrained=True)
    return LACNN(features, num_classes, LDN_model_path)
