import os
import sys

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append("/share/home/kcli/CL_research/iCaRL_ILtFA")
from public.data import get_dataset, get_multitask_experiment, AVAILABLE_TRANSFORMS


# name, tasks, data_dir="./datasets", only_config=False, verbose=False,
#                              exception=False, imagenet_json_path=None, imagenet_exception=False,
#                              train_data_transform=None, val_data_transform=None
def get_multiTask_dataset(dataset_name, dataset_path, imagenet_json_path, train_dataset_transform):
    print("get imagenet1000 dataset.")
    (train_datasets, test_datasets), config, classes_per_task = get_multitask_experiment(
        name=dataset_name, tasks=10, data_dir=dataset_path, exception=True,
        imagenet_json_path=imagenet_json_path, train_data_transform=train_dataset_transform)
    return train_datasets, test_datasets, config, classes_per_task


CIFAR_100_means = np.array([125.3, 123.0, 113.9]) / 255.0
CIFAR_100_stds = np.array([63.0, 62.1, 66.7]) / 255.0


def transNorm(recon, mean=CIFAR_100_means,
              std=CIFAR_100_stds):
    print(type(recon))
    print(recon.shape)
    recon = recon.cpu()
    recon_tansNorm = []
    for img in recon:
        img_temp = []
        for id, channel in enumerate(img):
            channel_temp = channel * std[id]
            channel_temp += mean[id]
            img_temp.append(channel_temp.numpy())
        recon_tansNorm.append(img_temp)
    recon_tansNorm = torch.Tensor(np.array(recon_tansNorm))
    return recon_tansNorm.to("cuda")


def visualize(self, epoch, *args):
    self.net.eval()
    for imgs, _ in self.dl_test():
        break
    imgs = imgs.to(self.device)
    with torch.no_grad():
        recon, _ = self.net(imgs)
    recon = self.transNorm(recon, mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    whole = torch.cat((imgs, recon), dim=0)
    grid = torchvision.utils.make_grid(whole, nrow=16, padding=2, pad_value=255)
    filename = 'cifar100_reconstruction_epoch_{}.png'.format(epoch)
    if len(args) != 0:
        filename = 'cifar100_reconstruction_epoch_{}_{}.png'.format(epoch, '_'.join(args))
    torchvision.utils.save_image(grid, fp=os.path.join("./", filename))


def main():
    dataset_path = "/n02dat01/public_resource/dataset/ImageNet/"
    dataset_name = "ImageNet100"
    imagenet_json_path = "/share/home/kcli/Chore/datapreprocess/imagenet100.json"
    # extra_train_datasets = get_dataset("CIFAR10", 'train', dir=dataset_path)
    train_dataset_transform = transforms.Compose([
        *AVAILABLE_TRANSFORMS["imagenet_32"]['train_transform']
    ])
    imagenet100_train_datasets = get_dataset(dataset_name, type='train', dir=dataset_path,
                                             imagenet_json_path=imagenet_json_path, verbose=False,
                                             train_data_transform=train_dataset_transform)
    # train_datasets, val_datasets, data_config, classes_per_task = \
    #     get_multiTask_dataset(dataset_name, dataset_path=dataset_path, imagenet_json_path=imagenet_json_path,
    #                           train_dataset_transform=train_dataset_transform)
    print(len(imagenet100_train_datasets))
    train_loader = DataLoader(dataset=imagenet100_train_datasets, batch_size=16, num_workers=1, shuffle=True)
    for imgs, labels in train_loader:
        break
    print(imgs.shape)
    print(labels.shape)


if __name__ == "__main__":
    main()
