import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset


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
            sample[1] = target
        return sample


class ExemplarDataset(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).

    The images at the i-th entry of [exemplar_sets] belong to class [i], unless a [target_transform] is specified'''

    '''In ExemplarDataset, generally don't need target_transform. if exemplar is original images, need img_transform and 
    inv_transform else don't need'''

    def __init__(self, exemplar_sets, is_original_img, img_transform=None, target_transform=None, Two_transform=False):
        super().__init__()
        self.exemplar_sets = exemplar_sets
        self.is_original_img = is_original_img
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.Two_transform = Two_transform
        self.is_legal()

    def is_legal(self):
        if self.is_original_img:
            assert self.img_transform, "Err: self.is_original_img is True, needing self.img_transform"
        else:
            self.img_transform = None
        assert self.target_transform is None, "In ExemplarDataset, generally don't need target_transform"

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
            image = self.exemplar_sets[class_id][exemplar_id]
            image = Image.fromarray(image)
            if self.Two_transform:
                image_1 = self.img_transform(image)
                image_2 = self.img_transform(image)
                return ([image_1, image_2], class_id_to_return)
            image = self.img_transform(image)
            return (image, class_id_to_return)
        else:
            image = torch.from_numpy(self.exemplar_sets[class_id][exemplar_id])
            if self.Two_transform:
                return ([image, image], class_id_to_return)
            return (image, class_id_to_return)


class Domain_ExemplarDataset(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).

    The images at the i-th entry of [exemplar_sets] belong to class [i], unless a [target_transform] is specified'''

    '''In ExemplarDataset, generally don't need target_transform. if exemplar is original images, need img_transform and 
    inv_transform else don't need'''

    def __init__(self, exemplar_sets, is_original_img, img_transform=None, target_transform=None, Two_transform=False):
        super().__init__()
        self.exemplar_sets = exemplar_sets
        self.is_original_img = is_original_img
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.Two_transform = Two_transform
        self.is_legal()

    def is_legal(self):
        if self.is_original_img:
            assert self.img_transform, "Err: self.is_original_img is True, needing self.img_transform"
        else:
            self.img_transform = None
        #assert self.target_transform is None, "In ExemplarDataset, generally don't need target_transform"

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
            image = self.exemplar_sets[class_id][exemplar_id]
            image = Image.fromarray(image)
            if self.Two_transform:
                image_1 = self.img_transform(image)
                image_2 = self.img_transform(image)
                return ([image_1, image_2], class_id_to_return)
            image = self.img_transform(image)
            return (image, class_id_to_return)
        else:
            image = torch.from_numpy(self.exemplar_sets[class_id][exemplar_id])
            if self.Two_transform:
                return ([image, image], class_id_to_return)
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

        return input, target


# class TwoCropTransform:
#     """Create two crops of the same image"""
#     def __init__(self, transform):
#         self.transform = transform
#
#     def __call__(self, x):
#         return [self.transform(x), self.transform(x)]

class Transformed_2_Dataset(Dataset):
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
        inputs = [input, input]
        if self.transform:
            inputs = [self.transform(input), self.transform(input)]
        if self.target_transform:
            target = self.target_transform(target)

        return inputs, target


'''Specially designed for exemplars management'''


class ExemplarDataset_for_exemplars(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).

    The images at the i-th entry of [exemplar_sets] belong to class [i], unless a [target_transform] is specified'''

    '''In ExemplarDataset, generally don't need target_transform. if exemplar is original images, need img_transform'''

    def __init__(self, exemplar_sets, is_original_img, img_transform=None, target_transform=None, is_domain=False):
        super().__init__()
        self.exemplar_sets = exemplar_sets
        self.is_original_img = is_original_img
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.is_legal()
        self.is_domain = is_domain

    def is_legal(self):
        if self.is_original_img:
            assert self.img_transform, "Err: self.is_original_img is True, needing self.img_transform"
        '''if self.is_domain:
            assert self.target_transform is None, "In ExemplarDataset, generally don't need target_transform"'''

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
            image = self.exemplar_sets[class_id][exemplar_id]
            image_tensor = Image.fromarray(image)
            transformed_image = self.img_transform(image_tensor)
            return (transformed_image, class_id_to_return, image)
        else:
            image = torch.from_numpy(self.exemplar_sets[class_id][exemplar_id])
            return (image, class_id_to_return)


class TransformedDataset_for_exemplars(Dataset):
    '''Modify existing dataset with transform; for creating multiple MNIST-permutations w/o loading data every time.'''

    def __init__(self, original_dataset, store_original_imgs=True, transform=None, target_transform=None):
        super().__init__()
        self.store_original_imgs = store_original_imgs
        self.dataset = original_dataset
        self.transform = transform
        self.target_transform = target_transform
        # print(f"TransformedDataset_for_exemplars: {self.transform}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        (input, target) = self.dataset[index]
        transformed_imgs = input
        if self.transform:
            transformed_imgs = self.transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        if self.store_original_imgs:
            return transformed_imgs, target, np.array(input)
        else:
            return transformed_imgs, target, target


'''As for BiC, train_dataset must be further split to train_dataset and 
a small validate dataset for correcting bias'''

'''Spliting dataset does not involve transform including images_transform or label_transform'''


def split_dataset(original_dataset, split_number_for_trainVal, class_list, seed=0):
    new_train_dataset = None
    generated_val_dataset = None
    for exp_class_index in class_list:
        dataset_for_exp_class_index = SubDataset(original_dataset, [exp_class_index], target_transform=None)
        assert len(dataset_for_exp_class_index) - split_number_for_trainVal > 0, \
            "len(dataset_for_exp_class_index) must be larger than split_number_for_trainVal in split_dataset func."
        train_dataset_for_exp_class_index, val_dataset_for_exp_class_index = torch.utils.data.random_split(
            dataset_for_exp_class_index, [len(dataset_for_exp_class_index) - split_number_for_trainVal,
                                          split_number_for_trainVal],
            generator=torch.Generator().manual_seed(
                seed))
        if new_train_dataset is None:
            new_train_dataset = train_dataset_for_exp_class_index
            generated_val_dataset = val_dataset_for_exp_class_index
        else:
            new_train_dataset = ConcatDataset([new_train_dataset, train_dataset_for_exp_class_index])
            generated_val_dataset = ConcatDataset([generated_val_dataset, val_dataset_for_exp_class_index])

    return new_train_dataset, generated_val_dataset
