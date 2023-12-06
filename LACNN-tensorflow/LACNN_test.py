'''

'''

from __future__ import print_function
from __future__ import division
import tensorflow as tf
import numpy as np
from glob import glob
import os, sys
import random
import time
import cv2
import LACNN
import utils


img_size = np.array([224, 224])
label_size = img_size / 2

if __name__ == '__main__':

    images_dir = "/media/fly/4898FC1598FC02EC/ChongWang/OCT2017"
    attenmap_dir = "/media/fly/4898FC1598FC02EC/ChongWang/OCT2017_attenmap"
    image_lists = utils.create_image_lists(images_dir)
    class_count = len(image_lists.keys())

    path_CNV = sorted(glob(os.path.join(images_dir, 'train', 'CNV') + '/*.jpeg'))
    path_DME = sorted(glob(os.path.join(images_dir, 'train', 'DME') + '/*.jpeg'))
    path_DRUSEN = sorted(glob(os.path.join(images_dir, 'train', 'DRUSEN') + '/*.jpeg'))
    path_NORMAL = sorted(glob(os.path.join(images_dir, 'train', 'NORMAL') + '/*.jpeg'))

    i_subset = 1
    n_subset = 6
    path_train, path_test = utils.split_samples(path_CNV, path_DME, path_DRUSEN, path_NORMAL, i_subset, n_subset)
    save_path = './Model/LACNN/LACNN_' + str(n_subset) + '_' + str(i_subset)

    input_holder = tf.placeholder(tf.float32, [None, img_size[0], img_size[1], 3])
    label_holder = tf.placeholder("float", [None, class_count])

    # Bulid LACNN
    attenmap_holder = tf.placeholder(tf.float32, [None, label_size[0], label_size[1]])
    model = LACNN.Model(vgg16_npy_path=None, mode='train')
    model.build(input_holder, attenmap_holder, keep_prob=1.0)
    print('LACNN Created')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    since = time.time()

    ckpt = tf.train.get_checkpoint_state(os.path.join(save_path, 'model'))
    saver = tf.train.Saver()
    saver.restore(sess, ckpt.model_checkpoint_path)

    # evaluation on test set
    prediction = tf.argmax(model.fc8, 1)
    probability = tf.nn.softmax(model.fc8, -1)


    ################################################ Test for cross validation dataset
    label_dict = {'CNV': [1.0, 0, 0, 0], 'DRUSEN': [0, 1.0, 0, 0], 'DME': [0, 0, 1.0, 0], 'NORMAL': [0, 0, 0, 1.0]}
    probabilities = []
    test_target = []
    for category in path_test:
        test_filenames = path_test[category]
        for i, path in enumerate(test_filenames):
            test_images = cv2.resize(cv2.imread(path), (img_size[0], img_size[1]))[np.newaxis, :]
            test_attenmap = utils.get_batch_of_attenmap_from_name([path])[np.newaxis, :]
            # test_attenmap = np.zeros_like(test_attenmap)
            ground_truth = np.array(label_dict[category])[np.newaxis, :]
            test_target.append(ground_truth)
            prob = sess.run(probability, feed_dict={input_holder: test_images,
                                                    label_holder: ground_truth,
                                                    attenmap_holder: test_attenmap})
            probabilities.append(prob)
    test_target = np.squeeze(np.array(test_target))
    ################################################


    time_elapsed = time.time() - since
    print("Total Model Runtime: {}min, {:0.2f}sec".format(int(time_elapsed // 60), time_elapsed % 60))

    probabilities = np.squeeze(np.array(probabilities))
    predictions = np.argmax(probabilities, axis=1)
    labels = np.argmax(test_target, axis=1)
    test_accuracy = np.sum(np.equal(predictions, labels)) / labels.size
    print("Final Accuracy: {:0.4f}".format(test_accuracy))



    # CNV      [1, 0, 0 ,0]
    # DRUSEN   [0, 1, 0 ,0]
    # DME      [0, 0, 1 ,0]
    # NORMAL   [0, 0, 0 ,1]



    LIST_OF_POS_IDX = [0]
    auc_0, se_0, sp_0, acc_0 = utils.compute_roc(probabilities, labels, LIST_OF_POS_IDX)
    print("POS_IDX:{}, Final Model AUC: {:0.4f}, SE: {:0.4f}, SP: {:0.4f}, ACC: {:0.4f}".format(LIST_OF_POS_IDX, auc_0,
                                                                                                se_0, sp_0, acc_0))

    LIST_OF_POS_IDX = [1]
    auc_1, se_1, sp_1, acc_1 = utils.compute_roc(probabilities, labels, LIST_OF_POS_IDX)
    print("POS_IDX:{}, Final Model AUC: {:0.4f}, SE: {:0.4f}, SP: {:0.4f}, ACC: {:0.4f}".format(LIST_OF_POS_IDX, auc_1,
                                                                                                se_1, sp_1, acc_1))

    LIST_OF_POS_IDX = [2]
    auc_2, se_2, sp_2, acc_2 = utils.compute_roc(probabilities, labels, LIST_OF_POS_IDX)
    print("POS_IDX:{}, Final Model AUC: {:0.4f}, SE: {:0.4f}, SP: {:0.4f}, ACC: {:0.4f}".format(LIST_OF_POS_IDX, auc_2,
                                                                                                se_2, sp_2, acc_2))

    LIST_OF_POS_IDX = [3]
    auc_3, se_3, sp_3, acc_3 = utils.compute_roc(probabilities, labels, LIST_OF_POS_IDX)
    print("POS_IDX:{}, Final Model AUC: {:0.4f}, SE: {:0.4f}, SP: {:0.4f}, ACC: {:0.4f}".format(LIST_OF_POS_IDX, auc_3,
                                                                                                se_3, sp_3, acc_3))
