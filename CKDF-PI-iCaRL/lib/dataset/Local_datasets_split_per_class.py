import copy
import json
import random
from abc import ABC

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from lib.data_transform.data_transform import AVAILABLE_TRANSFORMS


class local_dataset_per_task_class(Dataset):
    def __init__(self, root, key_list, imagenet_json_data, train=0, have_test_dataset=False, image_transform=None,
                 label_transform=None):
        super(local_dataset_per_task_class, self).__init__()
        self.train = train
        self.root = root
        self.transform = image_transform
        self.label_transform = label_transform
        self.filename_list = []
        self.imagename_2_label_dict = {}
        self.imagename_2_path_dict = {}
        self.json_data = imagenet_json_data
        if train == 0:  # for train dataset
            dir_path = "train"
            for i in range(len(key_list)):
                self.filename_list += self.json_data[key_list[i]]["train_images"]
                self.imagename_2_label_dict.update(dict(zip(self.json_data[key_list[i]]["train_images"],
                                                            [self.json_data[key_list[i]]["class_index"]] *
                                                            len(self.json_data[key_list[i]]["train_images"]))))
                image_path = [root + "/{}/{}".format(dir_path, imagename) for imagename in
                              self.json_data[key_list[i]]["train_images"]]
                self.imagename_2_path_dict.update(dict(zip(self.json_data[key_list[i]]["train_images"], image_path)))
        elif train == 2:  # for test dataset
            dir_path = "val"
            for i in range(len(key_list)):
                self.filename_list += self.json_data[key_list[i]]["test_images"]
                self.imagename_2_label_dict.update(dict(zip(self.json_data[key_list[i]]["test_images"],
                                                            [self.json_data[key_list[i]]["class_index"]] *
                                                            len(self.json_data[key_list[i]]["test_images"]))))
                image_path = [root + "/{}/{}".format(dir_path, imagename) for imagename in
                              self.json_data[key_list[i]]["test_images"]]
                self.imagename_2_path_dict.update(dict(zip(self.json_data[key_list[i]]["test_images"], image_path)))
        elif train == 1:  # for validation dataset
            if have_test_dataset:
                dir_path = "train"
            else:
                dir_path = "val"
            for i in range(len(key_list)):
                self.filename_list += self.json_data[key_list[i]]["val_images"]
                self.imagename_2_label_dict.update(dict(zip(self.json_data[key_list[i]]["val_images"],
                                                            [self.json_data[key_list[i]]["class_index"]] *
                                                            len(self.json_data[key_list[i]]["val_images"]))))
                image_path = [root + "/{}/{}".format(dir_path, imagename) for imagename in
                              self.json_data[key_list[i]]["val_images"]]
                self.imagename_2_path_dict.update(
                    dict(zip(self.json_data[key_list[i]]["val_images"], image_path)))
        assert len(self.filename_list) == len(self.imagename_2_label_dict)
        # random.shuffle(self.filename_list)
        # print(len(self.filename_list))

    def __getitem__(self, index):
        label = self.imagename_2_label_dict[self.filename_list[index]]
        img = Image.open(self.imagename_2_path_dict[self.filename_list[index]]).convert(
            "RGB")  # pil.Image.open(img_path)
        # img = np.array(Image.open(self.imagename_2_path_dict[self.filename_list[index]]).convert(
        #     "RGB"))  # pil.Image.open(img_path)
        # img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label

    def __len__(self):
        return len(self.filename_list)


class Local_Datasets_Split_Per_Class:
    """# Local_Datasets is dataset in our disc in the format of .jpeg, .png etc..
       # We give path to read the dataset and split it. The relative storage path is determined (refer to xxx)
       # The format of data_json_file is determined (refer to xx.json)
       # The format of split_selected_file is determined (refer to xx.json)"""

    # def __init__(self, dataset_name, all_classes=None, all_tasks=None, data_json_file=None, data_root=None,
    #              split_selected_data=None,
    #              seed=0):
    def __init__(self, cfg, split_selected_data=None):
        self.dataset_name = cfg.DATASET.dataset_name
        self.all_classes = cfg.DATASET.all_classes
        self.all_tasks = cfg.DATASET.all_tasks
        self._data_json_file = cfg.DATASET.data_json_file
        self._data_root = cfg.DATASET.data_root
        self._split_selected_data = split_selected_data
        self.seed = cfg.DATASET.split_seed
        self.classes_per_task = None
        self.original_imgs_train_datasets = None
        self.original_imgs_train_datasets_per_class = None
        self.val_datasets = None
        self.test_datasets = None
        self.target_transform = None
        self._have_test_dataset = False
        self._class_per_task_list = None
        self._class_name_per_task_list = None
        self._selected_imagenet_datas = {}
        # self.train_dataset_transform = transforms.Compose([
        #     *AVAILABLE_TRANSFORMS[self.dataset_name]['train_transform'],
        # ])
        self.train_dataset_transform = None
        self.val_test_dataset_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS[self.dataset_name]['test_transform'],
        ])
        pass

    def is_legal_initiate(self):
        print(f"Use dataset split: {self.dataset_name}")
        assert self._data_json_file and self._data_root, "self._data_json_file and self._data_root不可为空"
        if self.all_classes:
            assert self.all_tasks, "self.all_tasks不可为空"
            assert self.all_classes % self.all_tasks == 0, "self.all_classes % self.all_tasks != 0"
            if self._split_selected_data:
                assert len(
                    self._split_selected_data) == self.all_tasks, "self._split_selected_data 中的任务数与self.all_tasks不相等"
                classes = 0
                pre_classes = None
                for key, value in self._split_selected_data.items():
                    classes += len(value)
                    if pre_classes is None:
                        pre_classes = len(value)
                    else:
                        assert pre_classes == len(value), "存在类别数不相同的任务."
                assert classes == self.all_classes, "self._split_selected_data 中的类别数与self.all_classes不相等"
        else:
            assert self._split_selected_data, "self.all_classes 与 self._split_selected_data 不可全为空"
            self.all_classes = 0
            self.all_tasks = len(self._split_selected_data)
            pre_classes = None
            for key, value in self._split_selected_data.items():
                self.all_classes += len(value)
                if pre_classes is None:
                    pre_classes = len(value)
                else:
                    assert pre_classes == len(value), "存在类别数不相同的任务."
        self.classes_per_task = int(self.all_classes / self.all_tasks)

    def get_all_classes_tasks(self):
        return self.all_classes, self.all_tasks, self.classes_per_task

    def get_dataset(self):
        self.is_legal_initiate()
        self._class_per_task_list = []
        with open(self._data_json_file, 'r') as fr:
            imagenet_datas = json.load(fr)
            print("len(imagnet_datas):", len(imagenet_datas))
        for key, value in imagenet_datas.items():
            if "test_images" in value.keys():
                self._have_test_dataset = True
                print(f"Dataset uses train-val-test split.")
            else:
                print(f"Dataset uses train-val split.")
            break

        class_index_2_image_name = {}
        for key, value in imagenet_datas.items():
            class_index_2_image_name[value["class_index"]] = key

        data_length = len(imagenet_datas)
        # print(f"class_index_2_image_name: {class_index_2_image_name}")
        if self._split_selected_data:
            self._class_name_per_task_list, self._class_per_task_list, selected_class_list = \
                self.get_determined_class_list(class_index_2_image_name)
        else:
            self._class_name_per_task_list, self._class_per_task_list, selected_class_list \
                = self.get_random_class_list(class_index_2_image_name, data_length)
        print(f"selected_class_list: {selected_class_list}")
        '''original class index mapping to class_index in current exp.'''
        original_classIndex_2_exp_classIndex = {}
        for exp_class_index in range(len(selected_class_list)):
            if self.seed == 0 and self._split_selected_data is None:
                assert selected_class_list[exp_class_index] == exp_class_index
            else:
                original_classIndex_2_exp_classIndex[selected_class_list[exp_class_index]] = exp_class_index
        if self.seed == 0 and self._split_selected_data is None:
            self.target_transform = None
            print(f"self.target_transform = None")
        else:
            self.target_transform = transforms.Lambda(lambda y, p=original_classIndex_2_exp_classIndex: int(p[y]))

        # if self.all_classes < len(imagenet_datas):
        #     for key, value in imagenet_datas.items():
        #         if value["class_index"] in selected_class_list:
        #             self._selected_imagenet_datas[key] = value
        for key, value in imagenet_datas.items():
            if value["class_index"] in selected_class_list:
                self._selected_imagenet_datas[key] = value
        assert len(
            self._selected_imagenet_datas) == self.all_classes, "Err: len(self._selected_imagenet_datas) != self.all_classes"
        del imagenet_datas
        print(f"self._class_name_per_task_list: {self._class_name_per_task_list}")
        self.original_imgs_train_datasets, self.original_imgs_train_datasets_per_class, self.val_datasets, \
        self.test_datasets = self.get_multitask_imagenet_experiment()
        if self.test_datasets is None:
            self.test_datasets = self.val_datasets
            self.val_datasets = None

    def get_determined_class_list(self, class_index_2_image_name):
        print("Use determined_class_list.")
        class_per_task_list = []
        selected_class_list = []
        class_name_per_task_list = []
        for task in range(self.all_tasks):
            class_per_task_list.append(self._split_selected_data[task])
            class_name_per_task_list.append([class_index_2_image_name[original_class_index] for original_class_index in
                                             self._split_selected_data[task]])
            selected_class_list.extend(self._split_selected_data[task])
        return class_name_per_task_list, class_per_task_list, selected_class_list
        pass

    def get_random_class_list(self, class_index_2_image_name, data_length):
        class_per_task_list = []
        class_name_per_task_list = []
        selected_list = np.array([class_index for class_index in range(data_length)])
        if self.seed != 0:
            print("Use random selected_class_list.")
            np.random.seed(self.seed)
            np.random.shuffle(selected_list)
        print("Use naturally ordered selected_class_list.")
        selected_list = list(selected_list)
        for task in range(self.all_tasks):
            class_per_task_list.append(
                selected_list[self.classes_per_task * task: self.classes_per_task * (task + 1)])
            class_name_per_task_list.append([class_index_2_image_name[original_class_index] for original_class_index in
                                             selected_list[
                                             self.classes_per_task * task: self.classes_per_task * (task + 1)]])

        return class_name_per_task_list, class_per_task_list, selected_list
        pass

    def get_multitask_imagenet_experiment(self):
        '''Load, organize and return train- and test-dataset for requested experiment.

        [exception]:    <bool>; if True, for visualization no permutation is applied to first task (permMNIST) or digits
                                are not shuffled before being distributed over the tasks (splitMNIST)'''
        train_datasets = []
        train_datasets_per_class = []
        test_datasets = None
        val_datasets = []
        dataset_name = self.dataset_name.lower()
        if "imagenet" in dataset_name:
            if self._have_test_dataset:
                test_datasets = []
            for task_id in range(self.all_tasks):
                ImageNet_train, ImageNet_train_per_class = self.get_dataset_per_task(mode="train",
                                                                                     key_list=
                                                                                     self._class_name_per_task_list[
                                                                                         task_id])
                ImageNet_val = self.get_dataset_per_task(mode="val",
                                                         key_list=self._class_name_per_task_list[task_id])
                train_datasets.append(ImageNet_train)
                train_datasets_per_class.append(ImageNet_train_per_class)
                val_datasets.append(ImageNet_val)
                if self._have_test_dataset:
                    ImageNet_test = self.get_dataset_per_task(mode="test",
                                                              key_list=self._class_name_per_task_list[task_id])
                    test_datasets.append(ImageNet_test)
            pass
        else:
            raise RuntimeError('Given undefined experiment: {}'.format(self.dataset_name))

        return train_datasets, train_datasets_per_class, val_datasets, test_datasets

    def get_dataset_per_task(self, mode='train', key_list=None):
        '''Create [train|valid|test]-dataset.'''
        dataset_list_per_task = []
        if mode == 'test':
            dataset = local_dataset_per_task_class(self._data_root, key_list, self._selected_imagenet_datas,
                                                   train=2,
                                                   image_transform=self.val_test_dataset_transform,
                                                   label_transform=self.target_transform)
            return dataset

        elif mode == "train":
            for class_name in key_list:
                dataset = local_dataset_per_task_class(self._data_root, [class_name], self._selected_imagenet_datas,
                                                       train=0,
                                                       image_transform=self.train_dataset_transform,
                                                       label_transform=self.target_transform)
                dataset_list_per_task.append(dataset)
            dataset = local_dataset_per_task_class(self._data_root, key_list, self._selected_imagenet_datas,
                                                   train=0,
                                                   image_transform=self.train_dataset_transform,
                                                   label_transform=self.target_transform)
            return dataset, dataset_list_per_task

        elif mode == "val":
            dataset = local_dataset_per_task_class(self._data_root, key_list, self._selected_imagenet_datas,
                                                   train=1,
                                                   have_test_dataset=self._have_test_dataset,
                                                   image_transform=self.val_test_dataset_transform,
                                                   label_transform=self.target_transform)
            return dataset
        else:
            raise RuntimeError('Given undefined mode: {}'.format(mode))

    def write_split_selected_data(self, file_name):
        # self._split_selected_data
        fw = open(file_name, 'w')
        self._split_selected_data = {}
        for task_id in range(len(self._class_per_task_list)):
            self._split_selected_data[task_id] = self._class_per_task_list[task_id]
        json.dump(self._split_selected_data, fw, indent=4)
        fw.close()

    def get_split_selected_data(self):
        self._split_selected_data = {}
        for task_id in range(len(self._class_per_task_list)):
            self._split_selected_data[task_id] = self._class_per_task_list[task_id]
        return self._split_selected_data

    def get_selected_data(self):
        return self._class_per_task_list, self._class_name_per_task_list

    def update_split_selected_data(self, split_selected_data):
        self._split_selected_data = split_selected_data
