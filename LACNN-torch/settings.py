base_architecture = 'vgg16'
img_size = 224
num_classes = 4

LDN_model_path = './LDN_epoch_20.pth'

experiment_run = str(img_size)

data_path = '/mnt/c/chong/data/CellData/OCT/'
train_dir = data_path + 'train/'
test_dir = data_path + 'train/'
valid_dir = data_path + 'test/'


batch_size = 36


optimizer_lrs = {'features': 1e-4, 'fc_layer': 1e-4}
lr_step_size = 5


clip_gradient = 10

num_epochs = 30

epoch_show = 1
epoch_save = 1
epoch_val = 1

