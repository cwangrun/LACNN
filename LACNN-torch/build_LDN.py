import torch
from collections import OrderedDict
import torch.nn.functional as F
from torch.optim import Adam
from torch.backends import cudnn
from LDN import build_model, weights_init
import numpy as np
from log import create_logger
import os
from PIL import Image
import cv2
from torch.nn.utils import clip_grad_norm_


class Build_LDN(object):
    def __init__(self, train_loader, valid_loader, test_loader, config):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device('cuda')
        self.build_model()

        if config.mode == 'train':
            self.log, self.logclose = create_logger(log_filename=os.path.join(config.save_folder, 'train.log'))
        else:
            self.net.load_state_dict(torch.load(self.config.model_path))
            self.net.eval()
            self.log, self.logclose = create_logger(log_filename=os.path.join(config.test_folder, 'test.log'))

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def build_model(self):
        self.net = build_model().to(self.device)
        self.optimizer = Adam(self.net.parameters(), self.config.lr, )    # weight_decay=1e-4

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def clip(self, y):
        return torch.clamp(y, 0.0, 1.0)

    def eval_mae(self, y_pred, y):
        return torch.abs(y_pred - y).mean()

    def restoresave(self, prob_pred, label_paths):
        for i, path in enumerate(label_paths):
            label_PIL = Image.open(path).convert('L')
            size = label_PIL.size
            prob_restore = F.interpolate(prob_pred[i].unsqueeze(0), size=(size[1], size[0]), mode='bilinear', align_corners=True).cpu().squeeze()
            cv2.imwrite(path.replace('_mask.png', '_LDN.png'), (prob_restore.numpy() * 255).astype(np.uint8))
            # cv2.imwrite(path.replace('.jpeg', '_LDN.png'), (prob_restore.numpy() * 255).astype(np.uint8))

    def validation(self):
        avg_mae = 0.0
        self.net.eval()
        for i, data_batch in enumerate(self.valid_loader):
            with torch.no_grad():
                images, labels, _ = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                prob_pred = self.net(images)
            avg_mae += self.eval_mae(prob_pred, labels).cpu().item()
        self.net.train()
        return avg_mae / len(self.valid_loader)

    def test(self):
        avg_mae, img_num = 0.0, len(self.test_loader)
        self.net.eval()
        for i, data_batch in enumerate(self.test_loader):
            with torch.no_grad():
                images, labels, label_paths = data_batch
                images = images.to(self.device)
                prob_pred = self.net(images)
                self.restoresave(prob_pred, label_paths)
            mae = self.eval_mae(prob_pred.cpu(), labels)
            self.log("[%d] mae: %.4f" % (i, mae))
            avg_mae += mae
        avg_mae = avg_mae / img_num
        self.log('average mae: %.4f' % (avg_mae))

    def train(self):
        iter_num = len(self.train_loader)
        best_mae = 1.0 if self.config.val else None
        self.net.train()
        for epoch in range(self.config.epoch):
            loss_epoch = 0.
            for i, data_batch in enumerate(self.train_loader):
                self.net.zero_grad()
                x, y, _ = data_batch
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.net(x)
                loss = F.binary_cross_entropy(y_pred, y)
                loss.backward()
                # clip_grad_norm_(self.net.parameters(), self.config.clip_gradient)
                self.optimizer.step()
                loss_epoch += loss.item()
                print('epoch: [%d/%d], iter: [%d/%d], loss: [%.4f]' % (epoch, self.config.epoch, i, iter_num, loss.cpu().item()))

            if (epoch + 1) % self.config.epoch_show == 0:
                self.log('epoch: [%d/%d], epoch_loss: [%.4f]' % (epoch, self.config.epoch, loss_epoch / iter_num))

            if self.config.val and (epoch + 1) % self.config.epoch_val == 0:
                mae = self.validation()
                self.log('--- Best MAE: %.4f, Curr MAE: %.4f ---' % (best_mae, mae))
                if best_mae > mae:
                    best_mae = mae
                    torch.save(self.net.state_dict(), '%s/models/best.pth' % self.config.save_folder)
            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d.pth' % (self.config.save_folder, epoch + 1))
        torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_folder)

        self.logclose()
