import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import time
import vgg16
import LDN
import glob
import utils


img_size = np.array([224, 224])
label_size = img_size / 2


if __name__ == "__main__":

    batch_size = 1
    images_holder = tf.placeholder(tf.float32, [batch_size, img_size[0], img_size[1], 3])
    atten_holder = tf.placeholder(tf.float32, [batch_size, label_size[0], label_size[1], 2])

    # Bulid LDN
    model = LDN.Model(images_holder, atten_holder)
    model.build_model()
    attention_map = tf.reshape(model.Prob, [batch_size, label_size[0], label_size[1], 2])[:, :, :, 0]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state('./Model/LDN')
    saver = tf.train.Saver()
    saver.restore(sess, ckpt.model_checkpoint_path)

    images_dir = "/media/fly/4898FC1598FC02EC/ChongWang/OCT2017"
    attenmap_dir = "/media/fly/4898FC1598FC02EC/ChongWang/OCT2017_attenmap"
    image_lists = utils.create_image_lists(images_dir)

    for category in image_lists:
        for mode in image_lists[category]:
            for f_img in image_lists[category][mode]:

                if mode == 'training':
                    mode_1 = 'train'
                elif mode == 'testing':
                    mode_1 = 'test'
                else:
                    mode_1 = 'val'
                attenmap_path = os.path.join(attenmap_dir, mode_1, category, f_img)
                img = cv2.imread(os.path.join(images_dir, mode_1, category, f_img))
                attenmap_name, ext = os.path.splitext(attenmap_path)

                if img is not None:
                    ori_img = img.copy()
                    img_shape = img.shape
                    img = cv2.resize(img, (img_size[0], img_size[1]))
                    img = img.reshape((1, img_size[0], img_size[1], 3))

                    start_time = time.time()
                    result = sess.run(model.Prob,
                                      feed_dict={model.input_holder: img})
                    print("--- %s seconds ---" % (time.time() - start_time))

                    result = np.reshape(result, (label_size[0], label_size[1], 2))
                    result = result[:, :, 0]
                    result = cv2.resize(np.squeeze(result), (img_shape[1], img_shape[0]))

                    utils.mkdir(os.path.dirname(attenmap_name))
                    save_name = os.path.join(attenmap_name + '_LDN.png')
                    cv2.imwrite(save_name, (result * 255).astype(np.uint8))

    sess.close()




