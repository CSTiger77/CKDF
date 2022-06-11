import csv
import json
import os
import pickle
import random

import torch
import torchvision
from PIL import Image
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader


class imagenet_multi_dataset(Dataset):

    def __init__(self, root, class_list, imagenet_json_data, train=True, target_transform=None, transform=None):
        super(imagenet_multi_dataset, self).__init__()
        self.train = train
        self.root = root
        self.transform = transform
        self.filename_list = []
        self.label_2_index_dict = {}  # label编号到class_list的位置编号
        self.imagename_2_label_dict = {}
        self.imagename_2_path_dict = {}
        self.json_data = imagenet_json_data
        self.target_transform = target_transform
        self.sample_index = 0
        print("imagenet_multi_dataset")
        if train:
            dir_path = "train"
            for i in range(len(class_list)):
                self.label_2_index_dict[self.json_data[class_list[i]]["class_index"]] = i
                self.filename_list += self.json_data[class_list[i]]["train_images"]
                self.imagename_2_label_dict.update(dict(zip(self.json_data[class_list[i]]["train_images"],
                                                            [self.json_data[class_list[i]]["class_index"]] *
                                                            len(self.json_data[class_list[i]]["train_images"]))))
                image_path = [root + "/{}/{}".format(dir_path, imagename) for imagename in
                              self.json_data[class_list[i]]["train_images"]]
                self.imagename_2_path_dict.update(dict(zip(self.json_data[class_list[i]]["train_images"], image_path)))
        else:
            dir_path = "val"
            for i in range(len(class_list)):
                self.label_2_index_dict[self.json_data[class_list[i]]["class_index"]] = i
                self.filename_list += self.json_data[class_list[i]]["val_images"]
                self.imagename_2_label_dict.update(dict(zip(self.json_data[class_list[i]]["val_images"],
                                                            [self.json_data[class_list[i]]["class_index"]] *
                                                            len(self.json_data[class_list[i]]["val_images"]))))
                image_path = [root + "/{}/{}".format(dir_path, imagename) for imagename in
                              self.json_data[class_list[i]]["val_images"]]
                self.imagename_2_path_dict.update(dict(zip(self.json_data[class_list[i]]["val_images"], image_path)))
        assert len(self.filename_list) == len(self.imagename_2_label_dict)
        random.shuffle(self.filename_list)
        print("len(self.filename_list)", len(self.filename_list))
        # print(len(self.filename_list))

    def __getitem__(self, index):
        self.sample_index += 1
        label = self.imagename_2_label_dict[self.filename_list[index]]
        label_index = self.label_2_index_dict[label]
        img = np.array(Image.open(self.imagename_2_path_dict[self.filename_list[index]]).convert(
            "RGB"))  # pil.Image.open(img_path)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            return img, self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.filename_list)


class imagenet_dataset_per_task(Dataset):
    def __init__(self, root, key_list, imagenet_json_data, train=0, transform=None):
        super(imagenet_dataset_per_task, self).__init__()
        self.train = train
        self.root = root
        self.transform = transform
        self.filename_list = []
        self.imagename_2_label_dict = {}
        self.imagename_2_path_dict = {}
        self.json_data = imagenet_json_data
        if train == 0:
            dir_path = "train"
            for i in range(len(key_list)):
                self.filename_list += self.json_data[key_list[i]]["train_images"]
                self.imagename_2_label_dict.update(dict(zip(self.json_data[key_list[i]]["train_images"],
                                                            [self.json_data[key_list[i]]["class_index"]] *
                                                            len(self.json_data[key_list[i]]["train_images"]))))
                image_path = [root + "/{}/{}".format(dir_path, imagename) for imagename in
                              self.json_data[key_list[i]]["train_images"]]
                self.imagename_2_path_dict.update(dict(zip(self.json_data[key_list[i]]["train_images"], image_path)))
        elif train == 1:
            dir_path = "val"
            for i in range(len(key_list)):
                self.filename_list += self.json_data[key_list[i]]["val_images"]
                self.imagename_2_label_dict.update(dict(zip(self.json_data[key_list[i]]["val_images"],
                                                            [self.json_data[key_list[i]]["class_index"]] *
                                                            len(self.json_data[key_list[i]]["val_images"]))))
                image_path = [root + "/{}/{}".format(dir_path, imagename) for imagename in
                              self.json_data[key_list[i]]["val_images"]]
                self.imagename_2_path_dict.update(dict(zip(self.json_data[key_list[i]]["val_images"], image_path)))
        elif train == 2:
            dir_path = "train"
            for i in range(len(key_list)):
                self.filename_list += self.json_data[key_list[i]]["train_val_images"]
                self.imagename_2_label_dict.update(dict(zip(self.json_data[key_list[i]]["train_val_images"],
                                                            [self.json_data[key_list[i]]["class_index"]] *
                                                            len(self.json_data[key_list[i]]["train_val_images"]))))
                image_path = [root + "/{}/{}".format(dir_path, imagename) for imagename in
                              self.json_data[key_list[i]]["train_val_images"]]
                self.imagename_2_path_dict.update(dict(zip(self.json_data[key_list[i]]["train_val_images"], image_path)))
        assert len(self.filename_list) == len(self.imagename_2_label_dict)
        random.shuffle(self.filename_list)
        # print(len(self.filename_list))

    def __getitem__(self, index):
        label = self.imagename_2_label_dict[self.filename_list[index]]
        img = np.array(Image.open(self.imagename_2_path_dict[self.filename_list[index]]).convert(
            "RGB"))  # pil.Image.open(img_path)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.filename_list)
