import numpy as np
from torchvision import datasets
from torchvision.transforms import transforms

# specify configurations of available data-sets.
DATASET_CONFIGS = {
    'mnist32': {'size': 32, 'channels': 1, 'classes': 10},
    'mnist': {'size': 28, 'channels': 1, 'classes': 10},
    'CIFAR100': {'size': 32, 'channels': 3, 'classes': 100},
    'CIFAR10': {'size': 32, 'channels': 3, 'classes': 10},
    'imagenet': {'size': 224, 'channels': 3, 'classes': 1000},
    'imagenet100': {'size': 224, 'channels': 3, 'classes': 100},
    'imagenet_32': {'size': 32, 'channels': 3, 'classes': 1000},
    'caltech100': {'size': 224, 'channels': 3, 'classes': 100},
    'tiny_imagenet': {'size': 64, 'channels': 3, 'classes': 100},
}

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

tiny_imagenet_means = np.array([0.4802, 0.4480, 0.3975])
tiny_imagenet_stds = np.array([0.2770, 0.2691, 0.2821])
inv_tiny_imagenet_stds = 1 / tiny_imagenet_stds
inv_tiny_imagenet_means = -tiny_imagenet_means * tiny_imagenet_stds

tiny_imagenet_normalize = transforms.Normalize(mean=tiny_imagenet_means,
                                               std=tiny_imagenet_stds)

inv_tiny_imagenet_normalize = transforms.Normalize(mean=inv_tiny_imagenet_means,
                                                   std=inv_tiny_imagenet_stds)

AVAILABLE_DATASETS = {
    'mnist': datasets.MNIST,
    'CIFAR100': datasets.CIFAR100,
    'CIFAR10': datasets.CIFAR10,
}

# specify available transforms.
AVAILABLE_TRANSFORMS = {
    'mnist': {
        "Contra_train_transform": [
            transforms.RandomResizedCrop(size=28, scale=(0.7, 1.)),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
                                   p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
        ],
        "train_transform": [
            # transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=28, scale=(0.7, 1.)),
            transforms.ToTensor(),
        ],
        "test_transform": [
            transforms.ToTensor(),
        ],
    },
    'CIFAR100': {
        "Contra_train_transform": [
            # transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
                                   p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            CIFAR_100_normalize,  # R,G,B每层的归一化用到的均值和方差
        ],
        "train_transform": [
            transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            CIFAR_100_normalize,  # R,G,B每层的归一化用到的均值和方差
        ],
        "test_transform": [
            transforms.ToTensor(),
            CIFAR_100_normalize,
        ],
    },
    'CIFAR10': {
        "Contra_train_transform": [
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
                                   p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            CIFAR_100_normalize,  # R,G,B每层的归一化用到的均值和方差
        ],
        "train_transform": [
            transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            CIFAR_100_normalize,  # R,G,B每层的归一化用到的均值和方差
        ],
        "test_transform": [
            transforms.ToTensor(),
            CIFAR_100_normalize,
        ],
    },
    'imagenet': {
        "Contra_train_transform": [
            transforms.Resize(256),  # 重置图像分辨率
            transforms.CenterCrop(224),  # 中心裁剪
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
                                   p=0.8),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=[7, 7])], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            imagenet_normalize,  # 归一化
        ],

        "train_transform": [
            transforms.Resize(256),  # 重置图像分辨率
            transforms.CenterCrop(224),  # 中心裁剪
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
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
        "Contra_train_transform": [
            transforms.RandomResizedCrop(size=64, scale=(0.1, 1.)),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
                                   p=0.8),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=[7, 7])], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            # imagenet_normalize,  # use imagenet mean and std 归一化
            tiny_imagenet_normalize,
        ],
        "train_transform": [
            # transforms.Resize((64, 64)),  # 重置图像分辨率
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # imagenet_normalize,  # use imagenet mean and std 归一化
            tiny_imagenet_normalize,
        ],
        "test_transform": [
            # transforms.Resize((64, 64)),  # 重置图像分辨率
            transforms.ToTensor(),
            # imagenet_normalize,  # use imagenet mean and std 归一化
            tiny_imagenet_normalize,
        ],


        # "transform_for_original_exemplar": [
        #     transforms.Resize(256),  # 重置图像分辨率
        #     transforms.CenterCrop(224),  # 中心裁剪
        # ],
        #
        # "transform_original_exemplar_for_train": [
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     imagenet_normalize,  # 归一化
        # ],
        # "transform_original_exemplar_for_val": [
        #     transforms.ToTensor(),
        #     imagenet_normalize,  # 归一化
        # ],
    }
}

# specify configurations of available data-sets.
# DATASET_CONFIGS = {
#     'mnist32': {'size': 32, 'channels': 1, 'classes': 10},
#     'mnist': {'size': 28, 'channels': 1, 'classes': 10},
#     'CIFAR100': {'size': 32, 'channels': 3, 'classes': 100},
#     'CIFAR10': {'size': 32, 'channels': 3, 'classes': 10},
#     'imagenet': {'size': 224, 'channels': 3, 'classes': 1000},
#     'imagenet100': {'size': 224, 'channels': 3, 'classes': 100},
#     'imagenet_32': {'size': 32, 'channels': 3, 'classes': 1000},
#     'caltech100': {'size': 224, 'channels': 3, 'classes': 100},
#     'tiny_imagenet': {'size': 64, 'channels': 3, 'classes': 200},
# }
