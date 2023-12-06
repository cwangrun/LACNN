import inspect
import os
import numpy as np
import tensorflow as tf
import time

VGG_MEAN = [103.939, 116.779, 123.68]


class Model:
    def __init__(self, vgg16_npy_path=None, mode='train'):
        if vgg16_npy_path is None:
            path = inspect.getfile(Model)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        self.mode = mode
        self._extra_train_ops = []
        print("npy file loaded")

    def build(self, rgb, attention_map, keep_prob):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        alpha = 1
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
           red, green, blue = tf.split(value=rgb, num_or_size_splits=3, axis=3)
           imgs = tf.concat(axis=3, values=[
               red - VGG_MEAN[0],
               green - VGG_MEAN[1],
               blue - VGG_MEAN[2],
           ])

        self.conv1_1 = self.conv_layer(imgs, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.combined_featuremaps_1 = self.conv1_2 + self.get_lesion_featuremaps(self.conv1_2, attention_map) * alpha
        self.pool1 = self.max_pool(self.combined_featuremaps_1, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.combined_featuremaps_2 = self.conv2_2 + self.get_lesion_featuremaps(self.conv2_2, attention_map) * alpha
        self.pool2 = self.max_pool(self.combined_featuremaps_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.combined_featuremaps_3 = self.conv3_3 + self.get_lesion_featuremaps(self.conv3_3, attention_map) * alpha
        self.pool3 = self.max_pool(self.combined_featuremaps_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.combined_featuremaps_4 = self.conv4_3 + self.get_lesion_featuremaps(self.conv4_3, attention_map) * alpha
        self.pool4 = self.max_pool(self.combined_featuremaps_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.combined_featuremaps_5 = self.conv5_3 + self.get_lesion_featuremaps(self.conv5_3, attention_map) * alpha
        self.pool5 = self.max_pool(self.combined_featuremaps_5, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer_new(self.relu7, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def get_lesion_featuremaps(self, bottom, attention_map):
        d = bottom.get_shape()[1].value
        c = bottom.get_shape()[3].value
        attention_maps = tf.stack([attention_map] * c, axis=-1)
        attention_maps = tf.image.resize_bilinear(attention_maps, [d, d])
        return bottom * attention_maps

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def fc_layer_new(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])
            W_regul = lambda x: self.L2(x)
            weights = tf.get_variable('weights', [dim, 4], initializer=tf.contrib.layers.xavier_initializer(),
                                      regularizer=W_regul, trainable=True)
            biases = tf.get_variable('biases', [4], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_conv_filter(self, name):
        W_regul = lambda x: self.L2(x)
        return tf.get_variable(name="filter",
                               shape=self.data_dict[name][0].shape,
                              initializer=tf.contrib.layers.xavier_initializer(),
                              trainable=True,
                              regularizer=W_regul)

    def get_bias(self, name):
        return tf.get_variable(name="biases",
                               shape=self.data_dict[name][1].shape,
                              initializer=tf.contrib.layers.xavier_initializer(),
                              trainable=True)

    def get_fc_weight(self, name):
        W_regul = lambda x: self.L2(x)
        return tf.get_variable(name="filter",
                               shape=self.data_dict[name][0].shape,
                              initializer=tf.contrib.layers.xavier_initializer(),
                              trainable=True,
                              regularizer=W_regul)

    def L2(self, tensor, wd=0.0002):
        return tf.multiply(tf.nn.l2_loss(tensor), wd, name='L2-Loss')

    def conv_layer_scratch(self, bottom, name):
        with tf.variable_scope(name):

            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu
