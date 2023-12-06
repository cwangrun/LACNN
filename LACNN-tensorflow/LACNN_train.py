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

    # Defining other ops using Tensorflow
    ################################################
    loss_reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model.fc8, labels=label_holder))
    loss_total = loss_cls + loss_reg
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model.fc8, 1), tf.argmax(label_holder, 1)), tf.float32))
    prediction = tf.argmax(model.fc8, 1)
    probability = tf.nn.softmax(model.fc8, -1)

    # LACNN Training
    #####################################################
    eval_frequency = 10
    logs_frequency = 50
    save_frequency = 200
    lr = 1e-5
    epochs = 10
    batch_size = 24
    Total_samples = sum([len(temp)for temp in path_train.values()])
    Training_steps = int(epochs * Total_samples / batch_size)
    print('Total Training Step: ', Training_steps)

    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=5)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_total)
    sess.run(tf.global_variables_initializer())

    # Add summaries to a tensor for TensorBoard
    tf.summary.scalar('train_class_loss', loss_cls)
    tf.summary.scalar('train_accuracy', accuracy)
    train_writer = tf.summary.FileWriter(save_path + '/logs', sess.graph)
    tf.logging.set_verbosity(tf.logging.INFO)

    label_dict = {'CNV': 0, 'DRUSEN': 1, 'DME': 2, 'NORMAL': 3}
    (val_target, val_filenames, val_images) = utils.get_val_samples(image_lists, "validation", images_dir, label_dict)
    val_attenmap = utils.get_batch_of_attenmap_from_name(val_filenames)
    # val_attenmap = np.zeros_like(val_attenmap)
    print('Starting training')
    Train_Loss = 0
    Train_Acc = 0
    since = time.time()
    for i in range(Training_steps):
        is_finalstep = (i + 1 == Training_steps)

        (train_target, train_filenames, train_images) = utils.get_batch_of_samples(path_train, batch_size, label_dict)
        train_attenmap = utils.get_batch_of_attenmap_from_name(train_filenames)
        # train_attenmap = np.zeros_like(train_attenmap)
        _, train_acc, loss = sess.run([optimizer, accuracy, loss_total],
                                      feed_dict={input_holder: train_images,
                                                 label_holder: train_target,
                                                 attenmap_holder: train_attenmap})
        Train_Loss += loss
        Train_Acc += train_acc

        # Evaluation on specified frequency
        if i < 0.5 * Training_steps:
            eval_frequency = 100
        else:
            eval_frequency = 10

        if (i % eval_frequency) == 0 or is_finalstep:
            predictions = []
            for j in range(len(val_filenames)):
                pred = sess.run(prediction, feed_dict={input_holder: val_images[j][np.newaxis, :],
                                                       label_holder: val_target[j][np.newaxis, :],
                                                       attenmap_holder: val_attenmap[j][np.newaxis, :]})
                predictions.append(pred)
            predictions = np.squeeze(np.array(predictions))
            val_acc = np.sum(np.equal(predictions, np.argmax(val_target, -1))) / predictions.size
            tf.logging.info("Step: {}, Total loss: {:0.6f}, Train acc: {:0.6f}, Val acc: {:0.6f}".
                            format(i + 1, Train_Loss/(i + 1), Train_Acc/(i + 1), val_acc))

        if (i % logs_frequency) == 0 or is_finalstep:
            summary_op = tf.summary.merge_all()
            summary_str = sess.run(summary_op, feed_dict={input_holder: train_images,
                                                          label_holder: train_target,
                                                          attenmap_holder: train_attenmap})
            train_writer.add_summary(summary_str, i)

        if (i % save_frequency) == 0 or is_finalstep:
            checkpoint_path = os.path.join(save_path, 'model', 'LACNN.ckpt')
            saver.save(sess, checkpoint_path, global_step=i+1)

    time_elapsed = time.time() - since
    print("Total Runtime: {}min, {:0.2f}sec".format(int(time_elapsed // 60), time_elapsed % 60))



