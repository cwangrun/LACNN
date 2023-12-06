import os
import shutil
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import time
import argparse
from log import create_logger
import build_LACNN
from Dataset_CLS import config_dataset


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print("GPU ID:", os.environ['CUDA_VISIBLE_DEVICES'])


# book keeping namings and code
from settings import img_size, num_classes, experiment_run, base_architecture, LDN_model_path
from settings import train_dir, test_dir, valid_dir, batch_size


torch.multiprocessing.set_sharing_strategy('file_system')


def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


model_dir = 'saved_CLS_models/{}/'.format(datestr()) + '/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))

# load dataset
train_loader, test_loader, valid_loader = config_dataset(img_size, batch_size, train_dir, test_dir, valid_dir, model_dir)


log('train set size: {0}'.format(len(train_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('valid set size: {0}'.format(len(valid_loader.dataset)))


# construct the model
net = build_LACNN.build_model(base_architecture=base_architecture, num_classes=num_classes, LDN_model_path=LDN_model_path)
net = net.cuda()

class_specific = True

# define optimizer
from settings import optimizer_lrs, lr_step_size
optimizer_specs = \
[
 {'params': net.features.parameters(), 'lr': optimizer_lrs['features'], 'weight_decay': 2e-4},
 {'params': net.fc_layer.parameters(), 'lr': optimizer_lrs['fc_layer'], 'weight_decay': 2e-4},
]
optimizer = torch.optim.Adam(optimizer_specs)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.9)   # 0.9


def validation(net, test_loader):
    net.eval()
    targets = []
    preds = []
    for i, data_batch in enumerate(test_loader):
        with torch.no_grad():
            images, labels, label_paths = data_batch
            images = images.to(device)
            pred = net(images)
            preds.append(pred.argmax(axis=1).cpu())
            targets.append(labels)
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    acc = (preds == targets).sum() / len(preds)
    net.train()
    return acc.item()


from settings import num_epochs, epoch_show, epoch_save, epoch_val
from settings import clip_gradient


# train the model
log('start training')
device = torch.device('cuda')
best_acc = 0.0
for epoch in range(num_epochs):
    log('epoch: \t{0}'.format(epoch))
    log('### lr: \t{0}'.format(optimizer.param_groups[0]['lr']))
    loss_epoch = 0.
    for i, data_batch in enumerate(train_loader):
        net.zero_grad()
        x, y, _ = data_batch
        x, y = x.to(device), y.to(device)
        y_pred = net(x)
        loss = F.cross_entropy(y_pred, y)
        loss.backward()
        # clip_grad_norm_(net.parameters(), clip_gradient)
        optimizer.step()
        loss_epoch += loss.item()
        print('epoch: [%d/%d], iter: [%d/%d], loss: [%.4f]' % (epoch, num_epochs, i, len(train_loader), loss.item()))

    lr_scheduler.step()

    if (epoch + 1) % epoch_show == 0:
        log('epoch: [%d/%d], epoch_loss: [%.4f]' % (epoch, num_epochs, loss_epoch / len(train_loader)))

    if (epoch + 1) % epoch_val == 0:
        acc = validation(net, test_loader)
        log('--- Best ACC: %.4f, Curr ACC: %.4f ---' % (best_acc, acc))
        if best_acc < acc:
            best_acc = acc
            torch.save(net.state_dict(), '%s/best.pth' % model_dir)
    if (epoch + 1) % epoch_save == 0:
        torch.save(net.state_dict(), '%s/epoch_%d.pth' % (model_dir, epoch + 1))
torch.save(net.state_dict(), '%s/final.pth' % model_dir)

logclose()

