from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys
import random
import time
import glob, cv2

from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


img_size = np.array([224, 224])
label_size = img_size / 2

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def data_augmentation(dir):
    image_dir = dir + "/Image"
    lesion_dir = dir + "/Lesion"
    image_paths = os.listdir(image_dir)
    lesion_paths = os.listdir(lesion_dir)
    print("Flipping images")
    for path in zip(image_paths, lesion_paths):
        img = cv2.imread(os.path.join(image_dir, path[0]))
        les = cv2.imread(os.path.join(lesion_dir, path[1]))
        img_name = os.path.join(image_dir, os.path.splitext(path[0])[0] + '_flip.jpeg')
        les_name = os.path.join(lesion_dir, os.path.splitext(path[1])[0] + '_flip.png')
        cv2.imwrite(img_name, cv2.flip(img, 1))
        cv2.imwrite(les_name, cv2.flip(les, 1))

    image_paths = os.listdir(image_dir)
    lesion_paths = os.listdir(lesion_dir)
    print("Rotating images")
    for path in zip(image_paths, lesion_paths):
        img = cv2.imread(os.path.join(image_dir, path[0]))
        les = cv2.imread(os.path.join(lesion_dir, path[1]))
        img_name_15 = os.path.join(image_dir, os.path.splitext(path[0])[0] + '_rotate15.jpeg')
        les_name_15 = os.path.join(lesion_dir, os.path.splitext(path[1])[0] + '_rotate15.png')
        img_name_345 = os.path.join(image_dir, os.path.splitext(path[0])[0] + '_rotate345.jpeg')
        les_name_345 = os.path.join(lesion_dir, os.path.splitext(path[1])[0] + '_rotate345.png')
        cv2.imwrite(img_name_15, rotate(img, 15))
        cv2.imwrite(les_name_15, rotate(les, 15))
        cv2.imwrite(img_name_345, rotate(img, 345))
        cv2.imwrite(les_name_345, rotate(les, 345))


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def create_image_lists(image_dir):
    result = {}
    training_images = []
    testing_images = []
    validation_images = []
    for category in ["train", "test", "val"]:
        category_path = os.path.join(image_dir, category)
        try:
            bins = next(os.walk(category_path))[1]
        except StopIteration:
            sys.exit("ERROR: Missing either train/test/val folders in image_dir")
        for diagnosis in bins:
            bin_path = os.path.join(category_path, diagnosis)
            if category == "train":
                training_images.append(get_image_files(bin_path))
            if category == "test":
                testing_images.append(get_image_files(bin_path))
            if category == "val":
                validation_images.append(get_image_files(bin_path))
    for diagnosis in bins:
        result[diagnosis] = {
            "training": training_images[bins.index(diagnosis)],
            "testing": testing_images[bins.index(diagnosis)],
            "validation": validation_images[bins.index(diagnosis)],
        }
    return result


# Return a sorted list of image files at image_dir
def get_image_files(image_dir):
    fs = glob.glob("{}/*.jpeg".format(image_dir))
    fs = [os.path.basename(filename) for filename in fs]
    return sorted(fs)


# Return a path to an image with the given label at the given index
def get_image_path(image_lists, label_name, index, image_dir, category):
    if label_name not in image_lists:
        tf.logging.fatal("Label does not exist %s.", label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal("Category does not exist %s.", category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal("Label %s has no images in the category %s.", label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    if ("train" in category):
        full_path = os.path.join(image_dir, "train", label_name.upper(), base_name)
    elif ("test" in category):
        full_path = os.path.join(image_dir, "test", label_name.upper(), base_name)
    elif ("val" in category):
        full_path = os.path.join(image_dir, "val", label_name.upper(), base_name)
    return full_path


def get_val_samples(image_lists, category, image_dir, label_dict):
    class_count = len(image_lists.keys())
    ground_truths = []
    filenames = []
    images = []
    for label_name in ['CNV', 'DRUSEN', 'DME', 'NORMAL']:
        for image_index, image_name in enumerate(image_lists[label_name][category]):
            image_name = get_image_path(image_lists, label_name, image_index, image_dir, category)
            ground_truth = np.zeros(class_count, dtype=np.float32)
            ground_truth[label_dict[label_name]] = 1.0
            ground_truths.append(ground_truth)
            filenames.append(image_name)
            images.append(cv2.resize(cv2.imread(image_name), (img_size[0], img_size[1])))
    ground_truths = np.array(ground_truths)
    images = np.array(images)
    return ground_truths, filenames, images


def get_batch_of_samples(path, batch_size, label_dict):
    MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1
    filenames = []
    ground_truths = []
    images = []
    for i in range(batch_size):
        label_name = random.choice(['CNV', 'DRUSEN', 'DME', 'NORMAL'])
        image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1) % len(path[label_name])
        image_path = path[label_name][image_index]
        filenames.append(image_path)
        images.append(cv2.resize(cv2.imread(image_path), (img_size[0], img_size[1])))
        ground_truth = np.zeros(len(path), dtype=np.float32)
        ground_truth[label_dict[label_name]] = 1.0
        ground_truths.append(ground_truth)
    images = np.array(images)
    ground_truths = np.array(ground_truths)
    return ground_truths, filenames, images


def get_batch_of_attenmap_from_name(train_filenames):
    labels = []
    for image_name in train_filenames:
        label_name = image_name.replace('.jpeg', '_LDN.png')
        label_name = label_name.replace('OCT2017', 'OCT2017_attenmap')
        label = cv2.imread(label_name)[:, :, 0].astype(np.float32)
        label = cv2.resize(label, (label_size[0], label_size[1]))
        label = label / 255.
        label = label[np.newaxis, :]
        labels.append(label)
    labels = np.array(labels)
    labels = np.squeeze(labels)
    return labels


# Generates ROC plot and returns AUC using sklearn
def generate_roc(y_test, y_score, pos_label=1):
    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    # plt.figure()
    # plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
    # plt.plot([0, 1], [0, 1], "k--")
    # plt.xlim([0.0, 1.05])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver operating characteristic curve")
    # plt.show()
    return roc_auc


def compute_roc(probabilities, labels, LIST_OF_POS_IDX):
    roc_labels = [1 if label in LIST_OF_POS_IDX else 0 for label in labels]
    pos_probs = probabilities[:, LIST_OF_POS_IDX]
    roc_probs = np.sum(pos_probs, axis=1) if pos_probs.ndim == 2 else pos_probs
    auc = generate_roc(roc_labels, roc_probs, pos_label=1)
    y_preds = list(np.argmax(probabilities, axis=1))
    y_preds = [1 if label in LIST_OF_POS_IDX else 0 for label in y_preds]
    se = recall_score(roc_labels, y_preds, pos_label=1)
    sp = recall_score(roc_labels, y_preds, pos_label=0)
    acc = accuracy_score(roc_labels, y_preds)
    return auc, se, sp, acc


def split_samples(path_CNV, path_DME, path_DRUSEN, path_NORMAL, i, n):
    duration_CNV = int(len(path_CNV) / n)
    path_CNV_train = path_CNV[duration_CNV * (i - 1): duration_CNV * i]

    duration_DME = int(len(path_DME) / n)
    path_DME_train = path_DME[duration_DME * (i - 1): duration_DME * i]

    duration_DRUSEN = int(len(path_DRUSEN) / n)
    path_DRUSEN_train = path_DRUSEN[duration_DRUSEN * (i - 1): duration_DRUSEN * i]

    duration_NORMAL = int(len(path_NORMAL) / n)
    path_NORMAL_train = path_NORMAL[duration_NORMAL * (i - 1): duration_NORMAL * i]

    path_train = {}
    path_train["CNV"] = path_CNV_train
    path_train["DME"] = path_DME_train
    path_train["DRUSEN"] = path_DRUSEN_train
    path_train["NORMAL"] = path_NORMAL_train

    path_test = {}
    path_test["CNV"] = []
    path_test["DME"] = []
    path_test["DRUSEN"] = []
    path_test["NORMAL"] = []
    for path in path_CNV:
        if path not in path_CNV_train:
            path_test["CNV"].append(path)
    for path in path_DME:
        if path not in path_DME_train:
            path_test["DME"].append(path)
    for path in path_DRUSEN:
        if path not in path_DRUSEN_train:
            path_test["DRUSEN"].append(path)
    for path in path_NORMAL:
        if path not in path_NORMAL_train:
            path_test["NORMAL"].append(path)
    return path_train, path_test

