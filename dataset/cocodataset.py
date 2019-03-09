import os
import numpy as np

import torch
from torch.utils.data import Dataset
import cv2
from pycocotools.coco import COCO

from utils.utils import *
import json


def parse_json(data_dir, annotation_file):
    """解析限制品的数据"""
    with open(annotation_file, 'r') as f:
        data = json.load(f)

        images = [os.path.join(data_dir, 'restricted', image["file_name"]) for image in data["images"]]

        # for ind, annotations in enumerate(data["annotations"]):
        #     image_id = annotations["image_id"]
        #     category_id = annotations["category_id"] - 1
        #     labels[image_id][category_id] = 1
        #
        # assert len(images) == len(labels)
        print('the number of images:', len(images))
        print('the number of annotations: ', len(data["annotations"]))
        return images, data["annotations"]


def create_fold(data, save_path):
    dataset = dict()
    images = []
    annotations = []
    for item in data:
        images.append(item[0])
        for a in item[1]:
            annotations.append(a)
    dataset['images'] = images
    dataset['annotations'] = annotations
    with open(save_path, 'w') as f:
        json.dump(dataset, f)


def load_train_val_split(json_data, val_portion=.2):
    num_train = len(json_data['images'])
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(val_portion * num_train))

    train_indices, val_indices = indices[split:], indices[:split]

    print('split the number of {} data for train', len(train_indices))
    print('split the number of {} data for validation', len(val_indices))

    images = json_data['images']
    annotations = json_data['annotations']

    print('total annotations: ', len(annotations))
    print('total images: ', len(images))
    data = []
    max_labels = 0
    for image in images:
        id = image['id']
        anno = []
        for annotation in annotations:
            if id == annotation['image_id']:
                anno.append(annotation)
            if len(annotation) > max_labels:
                max_labels = len(annotation)
        data.append((image, anno))

    print('max labels in single image is: ', max_labels)
    data = np.array(data)
    train_data = data[train_indices]
    val_data = data[val_indices]

    create_fold(train_data, 'train.json')
    create_fold(val_data, 'val.json')

    print('create folds success')


class COCODataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(self, model_type, data_dir='COCO', json_file='train_no_poly.json',
                 name='restricted', img_size=416,
                 augmentation=None, min_size=1, debug=False):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            model_type (str): model name specified in config file
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            min_size (int): bounding boxes smaller than this are ignored
            debug (bool): if True, only one data id is selected from the dataset
        """
        self.data_dir = data_dir
        self.json_file = json_file
        self.model_type = model_type
        # annotation_file = self.data_dir + self.json_file
        self.coco = COCO(self.json_file)
        self.ids = self.coco.getImgIds()
        if debug:
            self.ids = self.ids[1:2]
            print("debug mode...", self.ids)
        # self.class_ids = sorted(self.coco.getCatIds())
        self.name = name
        self.max_labels = 8  # 当前数据集最大为8
        # self.max_labels = 50
        self.img_size = img_size
        self.min_size = min_size
        self.lrflip = augmentation['LRFLIP']
        self.jitter = augmentation['JITTER']
        self.random_placing = augmentation['RANDOM_PLACING']
        self.hue = augmentation['HUE']
        self.saturation = augmentation['SATURATION']
        self.exposure = augmentation['EXPOSURE']
        self.random_distort = augmentation['RANDOM_DISTORT']

        # 新增
        self.imgs, self.annotations = parse_json(data_dir, self.json_file)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up \
        and pre-processed.
        Args:
            index (int): data index
        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data. \
                The shape is :math:`[self.max_labels, 5]`. \
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            id_ (int): same as the input index. Used for evaluation.
        """
        # id_ = self.ids[index]
        img_file = self.imgs[index]
        #
        # anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        # annotations = self.coco.loadAnns(anno_ids)

        lrflip = False
        if np.random.rand() > 0.5 and self.lrflip == True:
            lrflip = True

        # load image and preprocess
        # img_file = os.path.join(self.data_dir, self.name,
        #                         '{:012}'.format(id_) + '.jpg')
        # print(img_file)
        img = cv2.imread(img_file)

        # if self.json_file == 'instances_val5k.json' and img is None:
        #     img_file = os.path.join(self.data_dir, 'train2017',
        #                             '{:012}'.format(id_) + '.jpg')
        #     img = cv2.imread(img_file)
        assert img is not None

        img, info_img = preprocess(img, self.img_size, jitter=self.jitter,
                                   random_placing=self.random_placing)

        if self.random_distort:
            img = random_distort(img, self.hue, self.saturation, self.exposure)

        img = np.transpose(img / 255., (2, 0, 1))

        if lrflip:
            img = np.flip(img, axis=2).copy()

        # load labels
        labels = []
        # for anno in annotations:
        for anno in self.annotations:
            if anno['bbox'][2] > self.min_size and anno['bbox'][3] > self.min_size:
                labels.append([])
                labels[-1].append(float(anno['category_id']))
                # labels[-1].append(self.class_ids.index(anno['category_id']))
                labels[-1].extend(float(x) for x in anno['bbox'])
                # print(labels)

        padded_labels = np.zeros((self.max_labels, 5))
        if len(labels) > 0:
            labels = np.stack(labels)
            if 'YOLO' in self.model_type:
                labels = label2yolobox(labels, info_img, self.img_size, lrflip)
            padded_labels[range(len(labels))[:self.max_labels]
            ] = labels[:self.max_labels]
        padded_labels = torch.from_numpy(padded_labels)

        return img, padded_labels, info_img, '1'

        # return img, padded_labels, info_img, id_


if __name__ == '__main__':
    with open('train_no_poly.json', 'r') as f:
        data = json.load(f)
        # print(data['annotations'])
        # print(len(data['annotations']))
    load_train_val_split(data)
