import json

import numpy as np
from torch.utils.data import ConcatDataset
from torchvision.transforms import transforms

from lib.data_transform.data_transform import AVAILABLE_TRANSFORMS, DATASET_CONFIGS, AVAILABLE_DATASETS
from lib.dataset import SubDataset, split_dataset, TransformedDataset


class Torchvision_Datasets_Split:
    """# The format of data_json_file is determined (refer to xx.json)
       # The format of split_selected_file is determined (refer to xx.json)"""

    def __init__(self, cfg, split_selected_data=None):
        self.dataset_name = cfg.DATASET.dataset_name
        self.dataset_root = cfg.DATASET.data_root
        self.all_classes = cfg.DATASET.all_classes
        self.all_tasks = cfg.DATASET.all_tasks
        self.input_size = cfg.INPUT_SIZE
        self.color_space = cfg.COLOR_SPACE
        self._split_selected_data = split_selected_data
        self.seed = cfg.DATASET.split_seed
        self._val_length = cfg.DATASET.val_length
        self._use_svhn_extra = cfg.DATASET.use_svhn_extra
        self.classes_per_task = None
        self.original_imgs_train_datasets = None
        self.val_datasets = None
        self.test_datasets = None
        self.target_transform = None
        self._class_per_task_list = None
        # self.train_dataset_transform = transforms.Compose([
        #     *AVAILABLE_TRANSFORMS[self.dataset_name]['train_transform'],
        # ])
        self.train_dataset_transform = None
        self.val_test_dataset_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS[self.dataset_name]['test_transform'],
        ])

    def is_legal_initiate(self):
        print(f"Use dataset_split: {self.dataset_name}")
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

    def get_dataset(self):
        self.is_legal_initiate()
        self._class_per_task_list = []

        if self._split_selected_data:
            self._class_per_task_list, ordered_class_list = self.get_determined_class_list()
        else:
            self._class_per_task_list, ordered_class_list = self.get_random_class_list()
        print(f"ordered_class_list: {ordered_class_list}")

        '''original class index mapping to class_index in current exp.'''
        original_classIndex_2_exp_classIndex = {}
        for exp_class_index in range(len(ordered_class_list)):
            if self.seed == 0 and self._split_selected_data is None:
                assert ordered_class_list[exp_class_index] == exp_class_index
            else:
                original_classIndex_2_exp_classIndex[ordered_class_list[exp_class_index]] = exp_class_index
        if self.seed == 0 and self._split_selected_data is None:
            self.target_transform = None
            print(f"self.target_transform = None")
        else:
            self.target_transform = transforms.Lambda(lambda y, p=original_classIndex_2_exp_classIndex: int(p[y]))
            print(f"self.target_transform:{self.target_transform}")

        self.original_imgs_train_datasets, self.val_datasets, self.test_datasets = \
            self.get_multitask_experiment_splited_dataset()

    def get_determined_class_list(self):
        print("Use determined_class_list.")
        class_per_task_list = []
        full_class_list = []
        for task in range(self.all_tasks):
            class_per_task_list.append(self._split_selected_data[task])
            full_class_list.extend(self._split_selected_data[task])
        return class_per_task_list, full_class_list
        pass

    def get_random_class_list(self):
        class_per_task_list = []
        order_list = np.array([class_index for class_index in range(self.all_classes)])
        if self.seed != 0:
            print("Use random selected_class_list.")
            np.random.seed(self.seed)
            np.random.shuffle(order_list)
        else:
            print("Use naturally ordered selected_class_list.")
        order_list = list(order_list)
        for task in range(self.all_tasks):
            class_per_task_list.append(
                order_list[self.classes_per_task * task: self.classes_per_task * (task + 1)])

        return class_per_task_list, order_list
        pass

    def get_multitask_experiment_splited_dataset(self):
        '''Load, organize and return train- and test-dataset for requested experiment.

        [exception]:    <bool>; if True, for visualization no permutation is applied to first task (permMNIST) or digits
                                are not shuffled before being distributed over the tasks (splitMNIST)'''
        print("get_multitask_experiment datasets")
        dataset_name = self.dataset_name.lower()
        if 'mnist' in dataset_name:
            # check for number of tasks
            if self.all_classes != 10:
                raise ValueError("Experiment 'splitMNIST' cannot have more than 10 tasks!")
            # prepare train and test datasets with all classes
            mnist_val = None
            dataset = 'mnist'
            if self._val_length <= 0:
                mnist_train = load_dataset(dataset, mode="train", dir=self.dataset_root,
                                           image_transform=self.train_dataset_transform)
            else:
                mnist_original_dataset = load_dataset(dataset, mode="train", dir=self.dataset_root,
                                                      image_transform=None)
                mnist_train_dataset, mnist_val_dataset = split_dataset(mnist_original_dataset, self._val_length,
                                                                       class_list=[i for i in range(10)],
                                                                       seed=self.seed)
                mnist_train = mnist_train_dataset
                if self.train_dataset_transform:
                    mnist_train = TransformedDataset(mnist_train_dataset, transform=self.train_dataset_transform)
                mnist_val = TransformedDataset(mnist_val_dataset, transform=self.val_test_dataset_transform)

            mnist_test = load_dataset(dataset, mode="test", dir=self.dataset_root,
                                      image_transform=self.val_test_dataset_transform)

            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            val_datasets = None
            if self._val_length <= 0:
                for labels in self._class_per_task_list:
                    train_datasets.append(TransformedDataset(SubDataset(mnist_train, labels),
                                                             target_transform=self.target_transform))
                    test_datasets.append(TransformedDataset(SubDataset(mnist_test, labels),
                                                            target_transform=self.target_transform))
            else:
                val_datasets = []
                assert mnist_val is not None, "Err: mnist_val is NULL."
                for labels in self._class_per_task_list:
                    train_datasets.append(TransformedDataset(SubDataset(mnist_train, labels),
                                                             target_transform=self.target_transform))
                    test_datasets.append(TransformedDataset(SubDataset(mnist_test, labels),
                                                            target_transform=self.target_transform))
                    val_datasets.append(TransformedDataset(SubDataset(mnist_val, labels),
                                                           target_transform=self.target_transform))


        elif "cifar10" in dataset_name and "cifar100" not in dataset_name:
            if self.all_classes != 10:
                raise ValueError("Experiment 'split_CIFAR10' cannot have more than 10 classes!")
            # prepare train and test datasets with all classes
            CIFAR10_val = None
            dataset = 'CIFAR10'
            if self._val_length <= 0:
                CIFAR10_train = load_dataset(dataset, mode="train", dir=self.dataset_root,
                                             image_transform=self.train_dataset_transform)
            else:
                CIFAR10_original_dataset = load_dataset(dataset, mode="train", dir=self.dataset_root,
                                                        image_transform=None)
                CIFAR10_train_dataset, CIFAR10_val_dataset = split_dataset(CIFAR10_original_dataset, self._val_length,
                                                                           class_list=[i for i in range(10)],
                                                                           seed=self.seed)
                CIFAR10_train = CIFAR10_train_dataset
                if self.train_dataset_transform:
                    CIFAR10_train = TransformedDataset(CIFAR10_train_dataset, transform=self.train_dataset_transform)
                CIFAR10_val = TransformedDataset(CIFAR10_val_dataset, transform=self.val_test_dataset_transform)

            CIFAR10_test = load_dataset(dataset, mode="test", dir=self.dataset_root,
                                        image_transform=self.val_test_dataset_transform)
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            val_datasets = None
            if self._val_length <= 0:
                for labels in self._class_per_task_list:
                    train_datasets.append(TransformedDataset(SubDataset(CIFAR10_train, labels),
                                                             target_transform=self.target_transform))
                    test_datasets.append(TransformedDataset(SubDataset(CIFAR10_test, labels),
                                                            target_transform=self.target_transform))
            else:
                val_datasets = []
                assert CIFAR10_val is not None, "Err: CIFAR100_val is NULL."
                for labels in self._class_per_task_list:
                    train_datasets.append(TransformedDataset(SubDataset(CIFAR10_train, labels),
                                                             target_transform=self.target_transform))
                    test_datasets.append(TransformedDataset(SubDataset(CIFAR10_test, labels),
                                                            target_transform=self.target_transform))
                    val_datasets.append(TransformedDataset(SubDataset(CIFAR10_val, labels),
                                                           target_transform=self.target_transform))

        elif "cifar100" in dataset_name:
            if self.all_classes != 100:
                raise ValueError("Experiment 'split_CIFAR100' cannot have more than 100 classes!")
            # prepare train and test datasets with all classes
            CIFAR100_val = None
            dataset = 'CIFAR100'
            # print("96", ":", self.target_transform(96))
            if self._val_length <= 0:
                CIFAR100_train = load_dataset(dataset, mode="train", dir=self.dataset_root,
                                              image_transform=self.train_dataset_transform)
            else:
                CIFAR100_original_dataset = load_dataset(dataset, mode="train", dir=self.dataset_root,
                                                         image_transform=None)
                CIFAR100_train_dataset, CIFAR10_val_dataset = split_dataset(CIFAR100_original_dataset, self._val_length,
                                                                            class_list=[i for i in range(100)],
                                                                            seed=self.seed)
                CIFAR100_train = CIFAR100_train_dataset
                if self.train_dataset_transform:
                    CIFAR100_train = TransformedDataset(CIFAR100_train_dataset, transform=self.train_dataset_transform)
                CIFAR100_val = TransformedDataset(CIFAR10_val_dataset, transform=self.val_test_dataset_transform)

            CIFAR100_test = load_dataset(dataset, mode="test", dir=self.dataset_root,
                                         image_transform=self.val_test_dataset_transform)

            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            val_datasets = None
            if self._val_length <= 0:
                for labels in self._class_per_task_list:
                    train_datasets.append(TransformedDataset(SubDataset(CIFAR100_train, labels),
                                                             target_transform=self.target_transform))
                    test_datasets.append(TransformedDataset(SubDataset(CIFAR100_test, labels),
                                                            target_transform=self.target_transform))
            else:
                val_datasets = []
                assert CIFAR100_val is not None, "Err: CIFAR100_val is NULL."
                for labels in self._class_per_task_list:
                    train_datasets.append(TransformedDataset(SubDataset(CIFAR100_train, labels),
                                                             target_transform=self.target_transform))
                    test_datasets.append(TransformedDataset(SubDataset(CIFAR100_test, labels),
                                                            target_transform=self.target_transform))
                    val_datasets.append(TransformedDataset(SubDataset(CIFAR100_val, labels),
                                                           target_transform=self.target_transform))


        elif "svhn" in dataset_name:
            if self.all_classes != 10:
                raise ValueError("Experiment 'split_svhn' cannot have more than 10 classes!")
            dataset = 'SVHN'
            SVHN_train = load_dataset(dataset, mode="train", dir=self.dataset_root,
                                      image_transform=None)
            SVHN_extra = load_dataset(dataset, mode="train", dir=self.dataset_root,
                                      image_transform=None)

            if self._use_svhn_extra:
                SVHN_train = ConcatDataset([SVHN_train, SVHN_extra])

            SVHN_test = load_dataset(dataset, mode="test", dir=self.dataset_root,
                                     image_transform=self.val_test_dataset_transform)
            SVHN_val = None
            if self._val_length <= 0:
                SVHN_train = TransformedDataset(SVHN_train, transform=self.train_dataset_transform)
            else:

                SVHN_train, SVHN_val = split_dataset(SVHN_train, self._val_length,
                                                     class_list=[i for i in range(10)],
                                                     seed=self.seed)
                if self.train_dataset_transform:
                    SVHN_train = TransformedDataset(SVHN_train, transform=self.train_dataset_transform)
                SVHN_val = TransformedDataset(SVHN_val, transform=self.val_test_dataset_transform)

            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            val_datasets = None
            if self._val_length > 0:
                val_datasets = []
                assert SVHN_val is not None, "Err: SVHN_val is NULL."
                for labels in self._class_per_task_list:
                    train_datasets.append(TransformedDataset(SubDataset(SVHN_train, labels),
                                                             target_transform=self.target_transform))
                    test_datasets.append(TransformedDataset(SubDataset(SVHN_test, labels),
                                                            target_transform=self.target_transform))
                    val_datasets.append(TransformedDataset(SubDataset(SVHN_val, labels),
                                                           target_transform=self.target_transform))
            else:
                for labels in self._class_per_task_list:
                    train_datasets.append(TransformedDataset(SubDataset(SVHN_train, labels),
                                                             target_transform=self.target_transform))
                    test_datasets.append(TransformedDataset(SubDataset(SVHN_test, labels),
                                                            target_transform=self.target_transform))
        else:
            raise RuntimeError('Given undefined experiment: {}'.format(self.dataset_name))

        # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
        return train_datasets, val_datasets, test_datasets

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
        return self._class_per_task_list

    def update_split_selected_data(self, split_selected_data):
        self._split_selected_data = split_selected_data


def load_dataset(dataset_name, mode='train', download=True, dir='./datasets', image_transform=None,
                 target_transform=None):
    '''Create [train|valid|test]-dataset.'''
    data_name = 'mnist' if dataset_name == 'mnist28' else dataset_name
    dataset_class = AVAILABLE_DATASETS[data_name]
    dataset = None
    # load data-set
    if 'SVHN' not in dataset_name:
        if mode == 'test':
            dataset = dataset_class(dir, train=False,
                                    download=download, transform=image_transform)
        elif mode == 'train':
            dataset = dataset_class(dir, train=True,
                                    download=download, transform=image_transform)
    else:
        if mode == 'test':
            dataset = dataset_class(dir, split='test',
                                    download=download, transform=image_transform)
        elif mode == 'train':
            dataset = dataset_class(dir, split='train',
                                    download=download, transform=image_transform)
        elif mode == 'extra':
            dataset = dataset_class(dir, split='extra',
                                    download=download, transform=image_transform)

    return dataset


if __name__ == "__main__":
    # dataset = Torchvision_Datasets_Split()
    pass
