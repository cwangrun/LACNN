import argparse
import os
from Dataset_Lesion import get_loader
from build_LDN import Build_LDN
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'   # "0, 1"
print("GPU ID:", os.environ['CUDA_VISIBLE_DEVICES'])


torch.multiprocessing.set_sharing_strategy('file_system')


def main(config):
    train_loader = get_loader(config.train_path, config.resize, config.batch_size, mode='train')
    valid_loader = get_loader(config.valid_path, config.resize, config.batch_size, mode='valid')
    test_loader = get_loader(config.test_path, config.resize, config.batch_size, mode='test')

    LDN = Build_LDN(train_loader, valid_loader, test_loader, config)

    LDN.train()

    # LDN.test()


if __name__ == '__main__':
    data_root = os.path.join(os.path.expanduser('~'), 'data')

    # # -----Lesion dataset-----
    image_path_train = '/mnt/c/chong/data/LesionData'
    image_path_valid = '/mnt/c/chong/data/LesionValid'
    image_path_test = '/mnt/c/chong/data/LesionTest'

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--clip_gradient', type=float, default=1.0)

    # Training settings
    parser.add_argument('--train_path', type=str, default=image_path_train)
    parser.add_argument('--epoch', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val', type=bool, default=True)
    parser.add_argument('--valid_path', type=str, default=image_path_valid)
    parser.add_argument('--num_worker', type=int, default=6)

    parser.add_argument('--save_folder', type=str, default='./saved_LDN_models')
    parser.add_argument('--epoch_val', type=int, default=1)
    parser.add_argument('--epoch_save', type=int, default=1)
    parser.add_argument('--epoch_show', type=int, default=1)

    # Testing settings
    parser.add_argument('--test_path', type=str, default=image_path_test)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--test_folder', type=str, default='./saved_LDN_models/test')

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    config = parser.parse_args()
    if not os.path.exists(config.save_folder): os.mkdir(config.save_folder)
    if not os.path.exists(config.save_folder + '/models'): os.mkdir(config.save_folder + '/models')
    if not os.path.exists(config.test_folder): os.mkdir(config.test_folder)

    main(config)
