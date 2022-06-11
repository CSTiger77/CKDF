import copy
import json

import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset
import torch

# ----------------------------------------------------------------------------------------------------------#
from Caltech_100.caltech_dataset import caltech_dataset_per_task
from ImageNet.imgenet_dataset import imagenet_dataset_per_task
from tiny_imagenet.tiny_imgenet_dataset import tiny_imagenet_dataset_per_task


def _permutate_image_pixels(image, permutation):
    '''Permutate the pixels of an image according to [permutation].

    [image]         3D-tensor containing the image
    [permutation]   <ndarray> of pixel-indeces in their new order'''

    if permutation is None:
        return image
    else:
        c, h, w = image.size()
        image = image.view(c, -1)
        image = image[:, permutation]  # --> same permutation for each channel
        image = image.view(c, h, w)
        return image


def get_dataset(name, type='train', download=True, capacity=None, dir='./datasets',
                verbose=False, target_transform=None, train_data_transform=None, val_data_transform=None):
    '''Create [train|valid|test]-dataset.'''
    if "ImageNet" not in name:
        data_name = 'mnist' if name == 'mnist28' else name
        dataset_class = AVAILABLE_DATASETS[data_name]

        # specify image-transformations to be applied
        if train_data_transform is not None:
            train_dataset_transform = train_data_transform
        else:
            train_dataset_transform = transforms.Compose([
                *AVAILABLE_TRANSFORMS[name]['train_transform'],
            ])
        test_dataset_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS[name]['test_transform'],
        ])

        # load data-set
        if type == 'test':
            dataset = dataset_class(dir, train=False,
                                    download=download, transform=test_dataset_transform,
                                    target_transform=target_transform)
        elif type == 'train':
            dataset = dataset_class(dir, train=True,
                                    download=download, transform=train_dataset_transform,
                                    target_transform=target_transform)
        elif type == 'original_train':
            dataset = dataset_class(dir, train=True,
                                    download=download, transform=test_dataset_transform,
                                    target_transform=target_transform)

        # print information about dataset on the screen
        if verbose:
            print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

        # if dataset is (possibly) not large enough, create copies until it is.
        if capacity is not None and len(dataset) < capacity:
            dataset_copy = copy.deepcopy(dataset)
            dataset = ConcatDataset([dataset_copy for _ in range(int(np.ceil(capacity / len(dataset))))])
        if verbose:
            print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    return dataset


def get_dataset_per_task(name, type='train', dir='./datasets', imagenet_datas=None,
                         key_list=None, verbose=False):
    '''Create [train|valid|test]-dataset.'''
    if "ImageNet" in name:

        # specify image-transformations to be applied
        train_dataset_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS["imagenet"]['train_transform'],
        ])
        test_dataset_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS["imagenet"]['test_transform'],
        ])

        # load data-set
        if type == 'test':
            dataset = imagenet_dataset_per_task(dir, key_list, imagenet_datas, train=1,
                                                transform=test_dataset_transform)

        elif type == "train":
            dataset = imagenet_dataset_per_task(dir, key_list, imagenet_datas, train=0,
                                                transform=train_dataset_transform)
        elif type == "train_val":
            dataset = imagenet_dataset_per_task(dir, key_list, imagenet_datas, train=2,
                                                transform=train_dataset_transform)

        # print information about dataset on the screen
        if verbose:
            print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    return dataset


def get_caltech100_dataset_per_task(name, type='train', dir='./datasets', caltech100_datas=None,
                                    key_list=None, verbose=False):
    '''Create [train|valid|test]-dataset.'''
    if "Caltech" in name:

        # specify image-transformations to be applied
        train_dataset_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS["caltech"]['train_transform'],
        ])
        test_dataset_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS["caltech"]['test_transform'],
        ])

        # load data-set
        if type == 'test':
            dataset = caltech_dataset_per_task(dir, key_list, caltech100_datas, train=False,
                                               transform=test_dataset_transform)

        else:
            dataset = caltech_dataset_per_task(dir, key_list, caltech100_datas, train=True,
                                               transform=train_dataset_transform)

        # print information about dataset on the screen
        if verbose:
            print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    return dataset
    pass


def get_tiny_imagenet_dataset_per_task(name, type='train', dir='./datasets', tiny_imagenet_datas=None,
                                       key_list=None, verbose=False):
    '''Create [train|valid|test]-dataset.'''
    if "tiny_imagenet" in name:

        # specify image-transformations to be applied
        train_dataset_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS["tiny_imagenet"]['train_transform'],
        ])
        test_dataset_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS["tiny_imagenet"]['test_transform'],
        ])

        # load data-set
        if type == 'test':
            dataset = tiny_imagenet_dataset_per_task(dir, key_list, tiny_imagenet_datas, train=False,
                                                     transform=test_dataset_transform)

        else:
            dataset = tiny_imagenet_dataset_per_task(dir, key_list, tiny_imagenet_datas, train=True,
                                                     transform=train_dataset_transform)

        # print information about dataset on the screen
        if verbose:
            print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    return dataset


# ----------------------------------------------------------------------------------------------------------#


class SubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "targets"):
                if self.dataset.target_transform is None:
                    label = self.dataset.targets[index]
                else:
                    label = self.dataset.target_transform(self.dataset.targets[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indeces.append(index)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample


class ExemplarDataset(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).

    The images at the i-th entry of [exemplar_sets] belong to class [i], unless a [target_transform] is specified'''

    def __init__(self, exemplar_sets, img_transform=None, inv_transform=None, target_transform=None):
        super().__init__()
        self.exemplar_sets = exemplar_sets
        self.img_transform = img_transform
        self.inv_transform = inv_transform
        self.target_transform = target_transform

    def __len__(self):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            total += len(self.exemplar_sets[class_id])
        return total

    def __getitem__(self, index):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            exemplars_in_this_class = len(self.exemplar_sets[class_id])
            if index < (total + exemplars_in_this_class):
                class_id_to_return = class_id if self.target_transform is None else self.target_transform(class_id)
                exemplar_id = index - total
                break
            else:
                total += exemplars_in_this_class
        if self.img_transform:
            image = torch.from_numpy(self.exemplar_sets[class_id][exemplar_id])
            image = self.inv_transform(image)
            image = image.numpy()
            image = image.transpose(1, 2, 0)
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image = self.img_transform(image)
            return (image, class_id_to_return)
        else:
            image = torch.from_numpy(self.exemplar_sets[class_id][exemplar_id])
            return (image, class_id_to_return)


class FeaturesDataset(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).

    The images at the i-th entry of [exemplar_sets] belong to class [i], unless a [target_transform] is specified'''

    def __init__(self, feature_sets, label_sets, target_transform=None):
        super().__init__()
        self.feature_sets = feature_sets
        self.target_transform = target_transform
        self.label_sets = label_sets
        state = np.random.get_state()
        np.random.shuffle(self.label_sets)
        np.random.set_state(state)
        np.random.shuffle(self.feature_sets)

    def __len__(self):
        return len(self.feature_sets)

    def __getitem__(self, index):
        class_id = self.label_sets[index]
        if self.target_transform:
            class_id = self.target_transform(self.label_sets[index])
        return (torch.from_numpy(self.feature_sets[index]), int(class_id))


class Features_target_hats_Dataset(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).

    The images at the i-th entry of [exemplar_sets] belong to class [i], unless a [target_transform] is specified'''

    def __init__(self, feature_sets, target_hat_sets):
        super().__init__()
        self.feature_sets = feature_sets
        self.target_hat_sets = target_hat_sets
        state = np.random.get_state()
        np.random.shuffle(self.target_hat_sets)
        np.random.set_state(state)
        np.random.shuffle(self.feature_sets)

    def __len__(self):
        return len(self.feature_sets)

    def __getitem__(self, index):
        return self.feature_sets[index], self.target_hat_sets[index]


class TransformedDataset(Dataset):
    '''Modify existing dataset with transform; for creating multiple MNIST-permutations w/o loading data every time.'''

    def __init__(self, original_dataset, transform=None, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        (input, target) = self.dataset[index]
        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        return (input, target)


# ----------------------------------------------------------------------------------------------------------#


# specify available data-sets.

CIFAR_100_means = np.array([125.3, 123.0, 113.9]) / 255.0
CIFAR_100_stds = np.array([63.0, 62.1, 66.7]) / 255.0

inv_CIFAR_100_stds = 1 / CIFAR_100_stds
inv_CIFAR_100_means = -CIFAR_100_means * inv_CIFAR_100_stds

CIFAR_100_normalize = transforms.Normalize(mean=CIFAR_100_means,
                                           std=CIFAR_100_stds)

inv_CIFAR_100_normalize = transforms.Normalize(mean=inv_CIFAR_100_means,
                                               std=inv_CIFAR_100_stds)

imagenet_means = np.array([0.485, 0.456, 0.406])
imagenet_stds = np.array([0.229, 0.224, 0.225])
inv_imagenet_stds = 1 / imagenet_stds
inv_imagenet_means = -imagenet_means * inv_imagenet_stds

imagenet_normalize = transforms.Normalize(mean=imagenet_means,
                                          std=imagenet_stds)

inv_imagenet_normalize = transforms.Normalize(mean=inv_imagenet_means,
                                              std=inv_imagenet_stds)

AVAILABLE_DATASETS = {
    'mnist': datasets.MNIST,
    'CIFAR100': datasets.CIFAR100,
    'CIFAR10': datasets.CIFAR10,
}

# specify available transforms.
AVAILABLE_TRANSFORMS = {
    'mnist': {
        "train_transform": [
            transforms.Pad(2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ],
        "test_transform": [
            transforms.Pad(2),
            transforms.ToTensor(),
        ],
    },
    'mnist28': {
        "train_transform": [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ],
        "test_transform": [
            transforms.ToTensor(),
        ],
    },
    'CIFAR100': {
        "train_transform": [
            transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            CIFAR_100_normalize,  # R,G,B每层的归一化用到的均值和方差
        ],
        "test_transform": [
            transforms.ToTensor(),
            CIFAR_100_normalize,
        ],
    },
    'CIFAR100_examplar': {
        "train_transform": [
            transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            CIFAR_100_normalize,  # R,G,B每层的归一化用到的均值和方差
        ],
        "BiC_train_transform": [
            transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            CIFAR_100_normalize,  # R,G,B每层的归一化用到的均值和方差
        ],
        "test_transform": [
            transforms.ToTensor(),
            CIFAR_100_normalize,
        ],
    },
    'CIFAR10': {
        "train_transform": [
            transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            CIFAR_100_normalize,  # R,G,B每层的归一化用到的均值和方差
        ],
        "test_transform": [
            transforms.ToTensor(),
            CIFAR_100_normalize,
        ],
    },
    'imagenet': {
        "train_transform": [
            transforms.Resize(256),  # 重置图像分辨率
            transforms.CenterCrop(224),  # 中心裁剪
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            imagenet_normalize,  # 归一化
        ],
        "BiC_train_transform": [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            imagenet_normalize,  # 归一化
        ],
        "test_transform": [
            transforms.Resize(256),  # 重置图像分辨率
            transforms.CenterCrop(224),  # 中心裁剪
            transforms.ToTensor(),
            imagenet_normalize,  # 归一化
        ],
    },

    'imagenet_examplar': {
        "train_transform": [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            imagenet_normalize,  # 归一化
        ],
        "test_transform": [
            transforms.Resize(256),  # 重置图像分辨率
            transforms.CenterCrop(224),  # 中心裁剪
            transforms.ToTensor(),
            imagenet_normalize,  # 归一化
        ],
    },
    'caltech': {
        "train_transform": [
            transforms.Resize(256),  # 重置图像分辨率
            transforms.CenterCrop(224),  # 中心裁剪
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            imagenet_normalize,  # 归一化
        ],
        "test_transform": [
            transforms.Resize(256),  # 重置图像分辨率
            transforms.CenterCrop(224),  # 中心裁剪
            transforms.ToTensor(),
            imagenet_normalize,  # 归一化
        ],
    },
    'imagenet_32': {
        "train_transform": [
            transforms.Resize((32, 32)),  # 重置图像分辨率
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            CIFAR_100_normalize,  # use CIFAR100 mean and std 归一化
        ],
        "test_transform": [
            transforms.Resize((32, 32)),  # 重置图像分辨率
            transforms.ToTensor(),
            CIFAR_100_normalize,  # use CIFAR100 mean and std 归一化
        ],
    },
    'tiny_imagenet': {
        "train_transform": [
            transforms.Resize((64, 64)),  # 重置图像分辨率
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            imagenet_normalize,  # use imagenet mean and std 归一化
        ],
        "test_transform": [
            transforms.Resize((64, 64)),  # 重置图像分辨率
            transforms.ToTensor(),
            imagenet_normalize,  # use imagenet mean and std 归一化
        ],
    }
}

# specify configurations of available data-sets.
DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'mnist28': {'size': 28, 'channels': 1, 'classes': 10},
    'CIFAR100': {'size': 32, 'channels': 3, 'classes': 100},
    'CIFAR10': {'size': 32, 'channels': 3, 'classes': 10},
    'imagenet': {'size': 224, 'channels': 3, 'classes': 1000},
    'imagenet100': {'size': 224, 'channels': 3, 'classes': 100},
    'imagenet_32': {'size': 32, 'channels': 3, 'classes': 1000},
    'caltech100': {'size': 224, 'channels': 3, 'classes': 100},
    'tiny_imagenet': {'size': 64, 'channels': 3, 'classes': 100},
}


# ----------------------------------------------------------------------------------------------------------#


def get_multitask_experiment(name, tasks, data_dir="./datasets", only_config=False, verbose=False,
                             train_data_transform=None, exception=False):
    '''Load, organize and return train- and test-dataset for requested experiment.

    [exception]:    <bool>; if True, for visualization no permutation is applied to first task (permMNIST) or digits
                            are not shuffled before being distributed over the tasks (splitMNIST)'''
    print("get_multitask_experiment")
    if name == 'splitMNIST':
        # check for number of tasks
        if tasks > 10:
            raise ValueError("Experiment 'splitMNIST' cannot have more than 10 tasks!")
        # configurations
        config = DATASET_CONFIGS['mnist28']
        classes_per_task = int(np.floor(10 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.array(list(range(10))) if exception else np.random.permutation(list(range(10)))
            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
            config["permutation"] = permutation.tolist()
            # prepare train and test datasets with all classes
            mnist_train = get_dataset('mnist28', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose)
            mnist_test = get_dataset('mnist28', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
                # [[0,1], [2,3], [4,5], [6,7], [8,9]]
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                train_datasets.append(SubDataset(mnist_train, labels))
                test_datasets.append(SubDataset(mnist_test, labels))
    elif name == 'CIFAR100':
        # check for number of tasks
        # if tasks > 20:
        #     raise ValueError("Experiment 'CIFAR100' cannot have more than 20 tasks!")
        # configurations
        config = DATASET_CONFIGS['CIFAR100']
        classes_per_task = int(np.floor(100 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.array(list(range(100))) if exception else np.random.permutation(list(range(100)))
            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
            config["permutation"] = permutation.tolist()
            # prepare train and test datasets with all classes
            CIFAR_train = get_dataset('CIFAR100', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose, train_data_transform=train_data_transform)
            CIFAR_original_train = get_dataset('CIFAR100', type="original_train", dir=data_dir,
                                               target_transform=target_transform, verbose=verbose,
                                               train_data_transform=train_data_transform)
            CIFAR_test = get_dataset('CIFAR100', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose, train_data_transform=train_data_transform)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
                # [[0,1], [2,3], [4,5], [6,7], [8,9]]
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            original_train_datasets = []
            for labels in labels_per_task:
                train_datasets.append(SubDataset(CIFAR_train, labels))
                test_datasets.append(SubDataset(CIFAR_test, labels))
                original_train_datasets.append(SubDataset(CIFAR_original_train, labels))
    elif name == 'CIFAR10':
        # check for number of tasks
        # if tasks > 20:
        #     raise ValueError("Experiment 'CIFAR100' cannot have more than 20 tasks!")
        # configurations
        config = DATASET_CONFIGS['CIFAR10']
        classes_per_task = int(np.floor(10 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.array(list(range(10))) if exception else np.random.permutation(list(range(100)))
            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
            config["permutation"] = permutation.tolist()
            # prepare train and test datasets with all classes
            CIFAR_train = get_dataset('CIFAR10', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose, train_data_transform=train_data_transform)
            CIFAR_original_train = get_dataset('CIFAR10', type="original_train", dir=data_dir,
                                               target_transform=target_transform, verbose=verbose,
                                               train_data_transform=train_data_transform)
            CIFAR_test = get_dataset('CIFAR10', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose, train_data_transform=train_data_transform)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
                # [[0,1], [2,3], [4,5], [6,7], [8,9]]
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            original_train_datasets = []
            for labels in labels_per_task:
                train_datasets.append(SubDataset(CIFAR_train, labels))
                test_datasets.append(SubDataset(CIFAR_test, labels))
                original_train_datasets.append(SubDataset(CIFAR_original_train, labels))
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    # If needed, update number of (total) classes in the config-dictionary
    config['classes'] = classes_per_task * tasks

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    return config if only_config else ((original_train_datasets, train_datasets, test_datasets), config, classes_per_task)


def get_multitask_imagenet_experiment(name, tasks, data_dir="./datasets", get_val=False, only_config=False,
                                      verbose=False, imagenet_json_path=None):
    '''Load, organize and return train- and test-dataset for requested experiment.

    [exception]:    <bool>; if True, for visualization no permutation is applied to first task (permMNIST) or digits
                            are not shuffled before being distributed over the tasks (splitMNIST)'''
    if "ImageNet" in name:
        config = DATASET_CONFIGS['imagenet100'] if "100" in name else DATASET_CONFIGS['imagenet']
        train_datasets = []
        test_datasets = []
        train_val_datasets = []
        with open(imagenet_json_path, 'r') as fr:
            imagenet_datas = json.load(fr)
            print("len(imagnet_datas):", len(imagenet_datas))
        assert len(imagenet_datas) % tasks == 0
        classes_per_task = int(np.floor(len(imagenet_datas) / tasks))
        class_list = []
        for task_id in range(tasks):
            classid_per_task = []
            for class_id in range(task_id * classes_per_task, (task_id + 1) * classes_per_task):
                for key, value in imagenet_datas.items():
                    if value["class_index"] == class_id:
                        classid_per_task.append(key)
            class_list.append(classid_per_task)
        for task_id in range(tasks):
            ImageNet_train = get_dataset_per_task(name, type="train", dir=data_dir, imagenet_datas=imagenet_datas,
                                                  key_list=class_list[task_id], verbose=verbose)
            ImageNet_test = get_dataset_per_task(name, type="test", dir=data_dir, imagenet_datas=imagenet_datas,
                                                 key_list=class_list[task_id], verbose=verbose)
            train_datasets.append(ImageNet_train)
            test_datasets.append(ImageNet_test)
            if get_val:
                ImageNet_trainVal = get_dataset_per_task(name, type="train_val", dir=data_dir,
                                                         imagenet_datas=imagenet_datas, key_list=class_list[task_id],
                                                         verbose=verbose)
                train_val_datasets.append(ImageNet_trainVal)
        pass
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    # If needed, update number of (total) classes in the config-dictionary
    config['classes'] = classes_per_task * tasks

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    if get_val:
        return config if only_config else (
            (class_list, train_datasets, test_datasets, train_val_datasets), config, classes_per_task)
    else:
        return config if only_config else ((train_datasets, test_datasets), config, classes_per_task)


def fetch_imagenet_data(name, data_dir="./datasets", class_list=None, imagenet_json_data=None, verbose=False):
    '''Load, organize and return train- and test-dataset for requested experiment.

    [exception]:    <bool>; if True, for visualization no permutation is applied to first task (permMNIST) or digits
                            are not shuffled before being distributed over the tasks (splitMNIST)'''
    ImageNet_trainVal = None
    if "ImageNet" in name:
        config = DATASET_CONFIGS['imagenet100'] if "100" in name else DATASET_CONFIGS['imagenet']
        ImageNet_trainVal = get_dataset_per_task(name, type="train_val", dir=data_dir,
                                                 imagenet_datas=imagenet_json_data,
                                                 key_list=class_list,
                                                 verbose=verbose)
        pass
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    # If needed, update number of (total) classes in the config-dictionary
    config['classes'] = len(class_list)
    return ImageNet_trainVal, config


def get_multitask_caltech100_experiment(name, tasks, data_dir="./datasets", only_config=False, verbose=False,
                                        caltech100_json_path=None):
    '''Load, organize and return train- and test-dataset for requested experiment.

    [exception]:    <bool>; if True, for visualization no permutation is applied to first task (permMNIST) or digits
                            are not shuffled before being distributed over the tasks (splitMNIST)'''
    if "Caltech100" in name:
        config = DATASET_CONFIGS['caltech100']
        train_datasets = []
        test_datasets = []
        with open(caltech100_json_path, 'r') as fr:
            caltech100_datas = json.load(fr)
            print("len(caltech100_datas):", len(caltech100_datas))
        assert len(caltech100_datas) % tasks == 0
        classes_per_task = int(np.floor(len(caltech100_datas) / tasks))
        class_list = []
        for task_id in range(tasks):
            classid_per_task = []
            for class_id in range(task_id * classes_per_task, (task_id + 1) * classes_per_task):
                for key, value in caltech100_datas.items():
                    if value["id"] == class_id:
                        classid_per_task.append(key)
            class_list.append(classid_per_task)
        for task_id in range(tasks):
            caltech100_train = get_caltech100_dataset_per_task(name, type="train", dir=data_dir,
                                                               caltech100_datas=caltech100_datas,
                                                               key_list=class_list[task_id], verbose=verbose)
            caltech100_test = get_caltech100_dataset_per_task(name, type="test", dir=data_dir,
                                                              caltech100_datas=caltech100_datas,
                                                              key_list=class_list[task_id], verbose=verbose)
            train_datasets.append(caltech100_train)
            test_datasets.append(caltech100_test)
        pass
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    # If needed, update number of (total) classes in the config-dictionary
    config['classes'] = classes_per_task * tasks

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    return config if only_config else ((train_datasets, test_datasets), config, classes_per_task)


def get_multitask_tiny_imagenet_experiment(name, tasks, data_dir="./datasets", only_config=False, verbose=False,
                                           tiny_imagenet_json_path=None):
    '''Load, organize and return train- and test-dataset for requested experiment.

    [exception]:    <bool>; if True, for visualization no permutation is applied to first task (permMNIST) or digits
                            are not shuffled before being distributed over the tasks (splitMNIST)'''
    if "tiny_imagenet" in name:
        config = DATASET_CONFIGS['tiny_imagenet']
        train_datasets = []
        test_datasets = []
        with open(tiny_imagenet_json_path, 'r') as fr:
            imagenet_datas = json.load(fr)
        assert len(imagenet_datas) % tasks == 0
        classes_per_task = int(np.floor(len(imagenet_datas) / tasks))
        class_list = []
        for task_id in range(tasks):
            classid_per_task = []
            for class_id in range(task_id * classes_per_task, (task_id + 1) * classes_per_task):
                for key, value in imagenet_datas.items():
                    if value["class_index"] == class_id:
                        classid_per_task.append(key)
            class_list.append(classid_per_task)
        for task_id in range(tasks):
            tiny_imagenet_train = get_tiny_imagenet_dataset_per_task(name, type="train", dir=data_dir,
                                                                     tiny_imagenet_datas=imagenet_datas,
                                                                     key_list=class_list[task_id], verbose=verbose)
            tiny_imagenet_test = get_tiny_imagenet_dataset_per_task(name, type="test", dir=data_dir,
                                                                    tiny_imagenet_datas=imagenet_datas,
                                                                    key_list=class_list[task_id], verbose=verbose)
            train_datasets.append(tiny_imagenet_train)
            test_datasets.append(tiny_imagenet_test)
        pass
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    # If needed, update number of (total) classes in the config-dictionary
    config['classes'] = classes_per_task * tasks

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    return config if only_config else ((train_datasets, test_datasets), config, classes_per_task)
