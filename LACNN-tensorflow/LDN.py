import tensorflow as tf
import vgg16


class Model:
    def __init__(self, input_holder, label_holder):
        self.vgg = vgg16.Vgg16()
        self.input_holder = input_holder
        self.label_holder = label_holder

    def build_model(self):

        #build the VGG16 model
        vgg = self.vgg
        vgg.build(self.input_holder)

        fea_dim = 128
        batch_size = vgg.pool5.get_shape()[0].value

        #Global Feature and Global Score
        self.Fea_Global_1 = tf.nn.relu(self.Conv_2d(vgg.pool5, [3, 3, 512, fea_dim], 0.01,
                                                    padding='VALID', name='Fea_Global_1'))
        self.Fea_Global_2 = tf.nn.relu(self.Conv_2d(self.Fea_Global_1, [3, 3, fea_dim, fea_dim], 0.01,
                                                    padding='VALID', name='Fea_Global_2'))
        self.Fea_Global = self.Conv_2d(self.Fea_Global_2, [3, 3, fea_dim, fea_dim], 0.01,
                                       padding='VALID', name='Fea_Global')

        #Local Score
        self.Fea_P5 = tf.nn.relu(self.Conv_2d(vgg.pool5, [3, 3, 512, fea_dim], 0.01, padding='SAME', name='Fea_P5'))
        self.Fea_P4 = tf.nn.relu(self.Conv_2d(vgg.pool4, [3, 3, 512, fea_dim], 0.01, padding='SAME', name='Fea_P4'))
        self.Fea_P3 = tf.nn.relu(self.Conv_2d(vgg.pool3, [3, 3, 256, fea_dim], 0.01, padding='SAME', name='Fea_P3'))
        self.Fea_P2 = tf.nn.relu(self.Conv_2d(vgg.pool2, [3, 3, 128, fea_dim], 0.01, padding='SAME', name='Fea_P2'))
        self.Fea_P1 = tf.nn.relu(self.Conv_2d(vgg.pool1, [3, 3, 64, fea_dim], 0.01, padding='SAME', name='Fea_P1'))

        self.Fea_P5_LC = self.Contrast_Layer(self.Fea_P5, 3)
        self.Fea_P4_LC = self.Contrast_Layer(self.Fea_P4, 3)
        self.Fea_P3_LC = self.Contrast_Layer(self.Fea_P3, 3)
        self.Fea_P2_LC = self.Contrast_Layer(self.Fea_P2, 3)
        self.Fea_P1_LC = self.Contrast_Layer(self.Fea_P1, 3)

        #Deconv Layer
        self.Fea_P5_Up = tf.nn.relu(self.Deconv_2d(tf.concat([self.Fea_P5, self.Fea_P5_LC], axis=3),
                                                   [batch_size, self.Fea_P4.get_shape()[1].value, self.Fea_P4.get_shape()[2].value, fea_dim],
                                                   5, 2, name='Fea_P5_Deconv'))
        self.Fea_P4_Up = tf.nn.relu(self.Deconv_2d(tf.concat([self.Fea_P4, self.Fea_P4_LC, self.Fea_P5_Up], axis=3),
                                                   [batch_size, self.Fea_P3.get_shape()[1].value, self.Fea_P3.get_shape()[2].value, fea_dim*2],
                                                   5, 2, name='Fea_P4_Deconv'))
        self.Fea_P3_Up = tf.nn.relu(self.Deconv_2d(tf.concat([self.Fea_P3, self.Fea_P3_LC, self.Fea_P4_Up], axis=3),
                                                   [batch_size, self.Fea_P2.get_shape()[1].value, self.Fea_P2.get_shape()[2].value, fea_dim*3],
                                                   5, 2, name='Fea_P3_Deconv'))
        self.Fea_P2_Up = tf.nn.relu(self.Deconv_2d(tf.concat([self.Fea_P2, self.Fea_P2_LC, self.Fea_P3_Up], axis=3),
                                                   [batch_size, self.Fea_P1.get_shape()[1].value, self.Fea_P1.get_shape()[2].value, fea_dim*4],
                                                   5, 2, name='Fea_P2_Deconv'))

        self.Local_Fea = self.Conv_2d(tf.concat([self.Fea_P1, self.Fea_P1_LC, self.Fea_P2_Up], axis=3),
                                      [1, 1, fea_dim*6, fea_dim*5], 0.01, padding='VALID', name='Local_Fea')
        self.Local_Score = self.Conv_2d(self.Local_Fea, [1, 1, fea_dim*5, 2], 0.01, padding='VALID', name='Local_Score')

        self.Global_Score = self.Conv_2d(self.Fea_Global,
                                         [1, 1, fea_dim, 2], 0.01, padding='VALID', name='Global_Score')

        self.Score = self.Local_Score + self.Global_Score
        self.Score = tf.reshape(self.Score, [-1, 2])

        self.Prob = tf.nn.softmax(self.Score)

        #Loss Function
        self.label_holder = tf.reshape(self.label_holder,  [-1, 2])
        self.Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Score,
                                                                                  labels=self.label_holder))

        self.correct_prediction = tf.equal(tf.argmax(self.Score, 1), tf.argmax(self.label_holder, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def Conv_2d(self, input_, shape, stddev, name, padding='SAME'):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=shape,
                                initializer=tf.truncated_normal_initializer(stddev=stddev))

            conv = tf.nn.conv2d(input_, W, [1, 1, 1, 1], padding=padding)

            b = tf.Variable(tf.constant(0.0, shape=[shape[3]]), name='b')
            conv = tf.nn.bias_add(conv, b)
            return conv

    def Deconv_2d(self, input_, output_shape,
                  k_s=3, st_s=2, stddev=0.01, padding='SAME', name="deconv2d"):
        with tf.variable_scope(name):
            W = tf.get_variable('W',
                                shape=[k_s, k_s, output_shape[3], input_.get_shape()[3]],
                                initializer=tf.random_normal_initializer(stddev=stddev))

            deconv = tf.nn.conv2d_transpose(input_, W, output_shape=output_shape,
                                            strides=[1, st_s, st_s, 1], padding=padding)

            b = tf.get_variable('b', [output_shape[3]], initializer=tf.constant_initializer(0.0))
            deconv = tf.nn.bias_add(deconv, b)
        return deconv

    def Contrast_Layer(self, input_, k_s=3):
        h_s = k_s / 2
        return tf.subtract(input_, tf.nn.avg_pool(tf.pad(input_, [[0, 0], [h_s, h_s], [h_s, h_s], [0, 0]], 'SYMMETRIC'),
                                                  ksize=[1, k_s, k_s, 1], strides=[1, 1, 1, 1], padding='VALID'))