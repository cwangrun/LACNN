import tensorflow as tf
import LDN
import cv2, os, random
import numpy as np
from utils import data_augmentation

img_size = np.array([224, 224])
label_size = img_size / 2

if __name__ == "__main__":

    dir = './LesionData'
    # data_augmentation(dir)

    input_holder = tf.placeholder(tf.float32, [1, img_size[0], img_size[1], 3])
    label_holder = tf.placeholder(tf.float32, [1, label_size[0], label_size[1], 2])
    model = LDN.Model(input_holder, label_holder)
    model.build_model()

    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=5)
    model_path = './Model/LDN'
    num_epochs = 10
    eval_frequency = 50
    lr = 1e-5
    optimizer = tf.train.AdamOptimizer(lr).minimize(model.Loss)
    sess.run(tf.global_variables_initializer())

    image_dir = dir + "/Image"
    lesion_dir = dir + '/Lesion'
    image_paths = sorted(os.listdir(image_dir))
    lesion_paths = sorted(os.listdir(lesion_dir))
    i = 0
    Loss = 0
    Acc = 0
    for epo in range(num_epochs):
        train_paths = zip(image_paths, lesion_paths)
        random.shuffle(train_paths)
        for path in train_paths:
            img = np.expand_dims(cv2.resize(cv2.imread(os.path.join(image_dir, path[0])), (img_size[0], img_size[1])),0)
            les = cv2.resize(cv2.imread(os.path.join(lesion_dir, path[1]))[:, :, 0]/255, (label_size[0], label_size[1]))
            les = np.expand_dims(np.stack((les, 1 - les), 2), 0)
            _, loss, acc = sess.run([optimizer, model.Loss, model.accuracy], feed_dict={input_holder: img,
                                                                                        label_holder: les})
            i += 1
            Loss += loss
            Acc += acc

            if i % eval_frequency == 0:
                print("Epoch: {}, Step: {}, Loss: {:0.6f}, Acc: {:0.6f}".format(epo, i, Loss / i, Acc / i))

        if (epo + 1) % 2 == 0:
            checkpoint_path = os.path.join(model_path, 'LDN.ckpt')
            saver.save(sess, checkpoint_path, global_step=epo + 1)