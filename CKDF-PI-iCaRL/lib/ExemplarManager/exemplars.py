import abc
import copy
import os
import random
import time

import numpy as np
from sklearn.cluster import KMeans
from torch.nn import functional as Func
import joblib
import torch
from sklearn import svm
from sklearn.model_selection import train_test_split
from torch import nn
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader

from lib.ExemplarManager.utils import get_mean_nearest_elem, get_decreasing_value_index, get_mean_nearest_elem_index, \
    get_mean_dis_order_cluster_index, get_min_dis_dif_class
from lib.dataset import TransformedDataset_for_exemplars, ExemplarDataset, ExemplarDataset_for_exemplars, SubDataset, \
    Domain_ExemplarDataset


class ExemplarManager:
    """Abstract  module for a classifier that can store and use exemplars.

    Adds a exemplar-methods to subclasses, and requires them to provide a 'feature-extractor' method."""

    def __init__(self, memory_budget, mng_approach, store_original_imgs, norm_exemplars,
                 centroid_order, img_transform_for_val=None, img_transform_for_train=None,
                 reduce_farest_exemplar=False, device='cuda'):
        super().__init__()

        # list with exemplar-sets
        self.exemplar_sets = []  # --> each exemplar_set is an <np.array> of N images with shape (N, Ch, H, W)

        # settings
        self.memory_budget = memory_budget

        '''# if self.store_original_imgs is True, Store original imgs(Image.numpy()) 
           # else store transformed imgs tensors.numpy().'''
        self.store_original_imgs = store_original_imgs
        self._img_transform_for_val = img_transform_for_val
        self._img_transform_for_train = img_transform_for_train
        self.mng_approach = mng_approach
        self._device = device

        # common parameters for herding and kmeans_selector
        self._norm_exemplars = norm_exemplars

        # parameters for herding
        self.exemplar_means = []
        self.compute_means = True
        # parameters for kmeans_selector
        self._reduce_farest_exemplar = reduce_farest_exemplar
        self._centroid_order = centroid_order
        self.centroid_features = []
        self.recompute_centroid_feature = True
        self.islegal()
        # parameters for classifying with exemplars

    def resume_manager(self, breakpoint_data, dis_cls=None):
        if dis_cls is None:
            self.exemplar_sets = breakpoint_data["exemplar_sets"]
        elif 'dis' in dis_cls:
            self.exemplar_sets = breakpoint_data["dis_exemplar_sets"]
        elif 'cls' in dis_cls:
            self.exemplar_sets = breakpoint_data["cls_exemplar_sets"]
        self.store_original_imgs = breakpoint_data["store_original_imgs"]
        self.islegal()
        pass

    def islegal(self):
        if self.store_original_imgs:
            assert self._img_transform_for_train, "self.store_original_imgs is true, " \
                                                  "self._img_transform_for_train can not be None."

    ####----MANAGING EXEMPLAR SETS----####

    def reduce_exemplar_sets(self, m_list, model=None, val_test_transform=None, batch_size=None, num_workers=None):
        if "kmeans" in self.mng_approach and self._reduce_farest_exemplar:
            self._reduce_farest_exemplars_func(model, val_test_transform, m_list, batch_size=batch_size,
                                               num_workers=num_workers)
        else:
            for y, P_y in enumerate(self.exemplar_sets):
                self.exemplar_sets[y] = P_y[:m_list[y]]

    def _reduce_farest_exemplars_func(self, model, val_test_transform, to_stored_number_list, batch_size, num_workers):
        # todo
        mode = model.training
        model.eval()
        for y, exemplar_per_class in enumerate(self.exemplar_sets):
            transformed_exemplar_dataset = ExemplarDataset([exemplar_per_class],
                                                           is_original_img=self.store_original_imgs,
                                                           img_transform=val_test_transform)
            dataloader = DataLoader(transformed_exemplar_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=False,
                                    drop_last=False)
            first_entry = True
            for (image_batch, _) in dataloader:
                image_batch = image_batch.to(self._device)
                feature_batch = self._get_extracted_features(model, image_batch)
                if first_entry:
                    features = feature_batch
                    first_entry = False
                else:
                    features = torch.cat([features, feature_batch], dim=0)
            if self._norm_exemplars:
                features = F.normalize(features, p=2, dim=1)
            to_stored_ordered_index_array = get_mean_nearest_elem_index(features, self._norm_exemplars,
                                                                        to_stored_number_list[y])
            to_stored_ordered_index_array = to_stored_ordered_index_array.sort()
            self.exemplar_sets[y] = exemplar_per_class[to_stored_ordered_index_array]
        model.train(mode=mode)
        pass

    '''use model.feature_extrator() to extract features to construct exemplar_sets for classes of current task.'''

    def construct_exemplar_set(self, dataset, model, to_store_number, batch_size, num_workers, **kwargs):
        '''Construct set of [n] exemplars from [dataset] using 'herding'.

        Note that [dataset] should be from specific class; selected sets are added to [self.exemplar_sets] in order.'''

        # set model to eval()-mode
        mode = model.training
        model.eval()
        n_max = len(dataset)
        print(f"original dataset length: {n_max}")
        to_store_number = min(to_store_number, n_max)
        exemplar_set = None
        if "herding" in self.mng_approach:
            exemplar_set = self.herding(dataset, model, batch_size, num_workers, to_store_number, **kwargs)
        elif "random" in self.mng_approach:
            exemplar_set = []
            indeces_selected = np.random.choice(n_max, size=to_store_number, replace=False)
            for k in indeces_selected:
                if self.store_original_imgs:
                    exemplar_set.append(dataset[k][-1])
                else:
                    exemplar_set.append(dataset[k][0].numpy())
        elif "kmeans" in self.mng_approach:
            exemplar_set = self.kmeans_selector(dataset, model, batch_size, num_workers, to_store_number, **kwargs)

        else:
            raise RuntimeError('self.mng_approach is illegal: {}'.format(self.mng_approach))

        # add this [exemplar_set] as a [n]x[ich]x[isz]x[isz] to the list of [exemplar_sets]
        self.exemplar_sets.append(np.array(exemplar_set))

        # set mode of model back
        model.train(mode=mode)


    @staticmethod
    def _get_extracted_features(model, image_batch):
        feature_batch = model(x=image_batch, is_nograd=True, herding_feature=True)
        return feature_batch

    def herding(self, dataset, model, batch_size, num_workers, to_store_number, **kwargs):
        # compute features for each example in [dataset]
        exemplar_set = []
        print(f"Use herding to construct exemplar sets")
        first_entry = True
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False,
                                drop_last=False, persistent_workers=True)
        for (image_batch, _, _) in dataloader:
            image_batch = image_batch.to(self._device)
            feature_batch = self._get_extracted_features(model, image_batch).cpu()
            if first_entry:
                features = feature_batch
                first_entry = False
            else:
                features = torch.cat([features, feature_batch], dim=0)
        if self._norm_exemplars:
            features = F.normalize(features, p=2, dim=1)

        # calculate mean of all features
        class_mean = torch.mean(features, dim=0, keepdim=True)
        if self._norm_exemplars:
            class_mean = F.normalize(class_mean, p=2, dim=1)

        # one by one, select exemplar that makes mean of all exemplars as close to [class_mean] as possible
        exemplar_set = self._herding_func(dataset, class_mean, features, to_store_number)
        del dataloader
        return exemplar_set

    def _herding_func(self, dataset, class_mean, features, to_store_number):
        exemplar_set = []
        exemplar_features = torch.zeros_like(features[:to_store_number])
        list_of_selected = []
        for k in range(to_store_number):
            if k > 0:
                exemplar_sum = torch.sum(exemplar_features[:k], dim=0).unsqueeze(0)
                features_means = (features + exemplar_sum) / (k + 1)
                features_dists = features_means - class_mean
            else:
                features_dists = features - class_mean
            index_selected = np.argmin(torch.norm(features_dists, p=2, dim=1).cpu())
            if index_selected in list_of_selected:
                raise ValueError("Exemplars should not be repeated!!!!")
            list_of_selected.append(index_selected)
            if self.store_original_imgs:
                exemplar_set.append(dataset[index_selected][-1])
            else:
                exemplar_set.append(dataset[index_selected][0].numpy())
            exemplar_features[k] = copy.deepcopy(features[index_selected])

            # make sure this example won't be selected again
            features[index_selected] = features[index_selected] + 100000

        return exemplar_set

    def kmeans_selector(self, dataset, model, batch_size, num_workers, to_store_number, **kwargs):  # todo, Done
        exemplar_set = []
        print(f"Use kmeans_selector to construct exemplar sets")
        first_entry = True
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False,
                                drop_last=False)
        for (transformed_image_batch, _, _) in dataloader:
            transformed_image_batch = transformed_image_batch.to(self._device)
            feature_batch = self._get_extracted_features(model, transformed_image_batch)
            if first_entry:
                features = feature_batch
                first_entry = False
            else:
                features = torch.cat([features, feature_batch], dim=0)
        if self._norm_exemplars:
            features = F.normalize(features, p=2, dim=1)

            # calculate mean of all features
        class_mean = torch.mean(features, dim=0, keepdim=True)
        if self._norm_exemplars:
            class_mean = F.normalize(class_mean, p=2, dim=1)
        kmeans = KMeans(n_clusters=to_store_number, random_state=0).fit(features.cpu())
        # centroids = kmeans.cluster_centers_
        labels = kmeans.labels_

        assert to_store_number == max(labels) + 1, "KMeans Err: to_store_number != max(labels) + 1"
        cluster_label_2_feature_index = [[] for i in range(to_store_number)]
        cluster_label_2_features = [None for i in range(to_store_number)]
        cluster_label_2_feature_numbers = [0 for i in range(to_store_number)]
        for label_id in range(to_store_number):
            cluster_label_2_feature_index[label_id] = np.where(labels == label_id)[0]
            cluster_label_2_feature_numbers[label_id] = len(cluster_label_2_feature_index[label_id])
            cluster_label_2_features[label_id] = features[cluster_label_2_feature_index[label_id]]
        # for feature_index in range(len(labels)):
        #     cluster_label_2_feature_index[labels[feature_index]].append(feature_index)
        #     if cluster_label_2_features[labels[feature_index]] is None:
        #         cluster_label_2_features[labels[feature_index]] = features[feature_index]
        #     else:
        #         cluster_label_2_features[labels[feature_index]] = torch.vstack([cluster_label_2_features[labels[
        #             feature_index]], features[feature_index]])
        #     cluster_label_2_feature_numbers[labels[feature_index]] += 1

        if "herding" in self._centroid_order:
            print("use herding to sort centroids")
            exemplar_set = self._centroid_hearding_order(dataset, cluster_label_2_feature_index,
                                                         cluster_label_2_features,
                                                         class_mean, features)
        elif "distance" in self._centroid_order:
            print("use distance to sort centroids")
            exemplar_set = self._centroid_mean_farest_order(dataset, cluster_label_2_feature_numbers,
                                                            cluster_label_2_feature_index, cluster_label_2_features,
                                                            class_mean, features)
        elif self._centroid_order is None:
            ''' # cluster_label_2_feature_centroid_index is a list, index is the kmeans_label_id, value is centroid 
                # index in the dataset'''
            cluster_label_2_feature_centroid_index = self._get_all_centroid_features(cluster_label_2_feature_index,
                                                                                     cluster_label_2_features)
            if self.store_original_imgs:
                exemplar_set = [dataset[centroid_idex][-1] for centroid_idex in
                                cluster_label_2_feature_centroid_index]
            else:
                exemplar_set = [dataset[centroid_idex][0].numpy() for centroid_idex in
                                cluster_label_2_feature_centroid_index]
        else:
            raise RuntimeError('self._centroid_order is illegal: {}'.format(self._centroid_order))
        return exemplar_set
        pass

    def _get_all_centroid_features(self, cluster_label_2_feature_index, cluster_label_2_features):
        cluster_label_2_feature_centroid_index = []
        for cluster_index in range(len(cluster_label_2_feature_index)):
            mean_nearest_feature_index = get_mean_nearest_elem(cluster_label_2_feature_index[
                                                                   cluster_index],
                                                               cluster_label_2_features[
                                                                   cluster_index],
                                                               norm_feature_mean=self._norm_exemplars)
            cluster_label_2_feature_centroid_index.append(mean_nearest_feature_index)
        return cluster_label_2_feature_centroid_index

    def _centroid_hearding_order(self, dataset, cluster_label_2_feature_index,
                                 cluster_label_2_features, class_mean, features):
        exemplar_set = None
        # Decreasing_feature_numbers_cluster_index = get_decreasing_value_index(cluster_label_2_feature_numbers,
        #                                                                       to_store_number)
        mean_nearest_feature_index_list = []
        for cluster_index in range(len(cluster_label_2_feature_index)):
            mean_nearest_feature_index = get_mean_nearest_elem(cluster_label_2_feature_index[
                                                                   cluster_index],
                                                               cluster_label_2_features[
                                                                   cluster_index],
                                                               norm_feature_mean=self._norm_exemplars)
            mean_nearest_feature_index_list.append(mean_nearest_feature_index)
        mean_nearest_features = features[mean_nearest_feature_index_list]
        exemplar_set = self._centroid_herding_order_func(dataset, class_mean, mean_nearest_features,
                                                         mean_nearest_feature_index_list,
                                                         to_store_number=len(mean_nearest_feature_index_list))
        return exemplar_set

    def _centroid_herding_order_func(self, dataset, class_mean, mean_nearest_features, mean_nearest_feature_index_list,
                                     to_store_number):
        exemplar_set = []
        exemplar_features = torch.zeros_like(mean_nearest_features[:to_store_number])
        list_of_selected = []
        for k in range(to_store_number):
            if k > 0:
                exemplar_sum = torch.sum(exemplar_features[:k], dim=0).unsqueeze(0)
                features_means = (mean_nearest_features + exemplar_sum) / (k + 1)
                features_dists = features_means - class_mean
            else:
                features_dists = mean_nearest_features - class_mean
            index_selected = np.argmin(torch.norm(features_dists.cpu(), p=2, dim=1))
            if index_selected in list_of_selected:
                raise ValueError("Exemplars should not be repeated!!!!")
            list_of_selected.append(index_selected)
            if self.store_original_imgs:
                exemplar_set.append(dataset[mean_nearest_feature_index_list[index_selected]][-1])
            else:
                exemplar_set.append(dataset[mean_nearest_feature_index_list[index_selected]][0].numpy())
            exemplar_features[k] = copy.deepcopy(mean_nearest_features[index_selected])

            # make sure this example won't be selected again
            mean_nearest_features[index_selected] = mean_nearest_features[index_selected] + 100000

        return exemplar_set

    def _centroid_mean_farest_order(self, dataset, cluster_label_2_feature_numbers,
                                    cluster_label_2_feature_index, cluster_label_2_features,
                                    class_mean, features):
        exemplar_set = []
        cluster_label_2_feature_centroid = []
        cluster_label_2_feature_centroid_index = []
        for cluster_index in range(len(cluster_label_2_feature_index)):
            mean_nearest_feature_index = get_mean_nearest_elem(cluster_label_2_feature_index[
                                                                   cluster_index],
                                                               cluster_label_2_features[
                                                                   cluster_index],
                                                               norm_feature_mean=self._norm_exemplars)
            cluster_label_2_feature_centroid.append(features[mean_nearest_feature_index])
            cluster_label_2_feature_centroid_index.append(mean_nearest_feature_index)
        cluster_label_2_feature_centroid = torch.stack(cluster_label_2_feature_centroid)
        reordered_cluster_index = get_mean_dis_order_cluster_index(class_mean, cluster_label_2_feature_numbers,
                                                                   cluster_label_2_feature_centroid)
        for cluster_index in reordered_cluster_index:
            if self.store_original_imgs:
                exemplar_set.append(dataset[cluster_label_2_feature_centroid_index[cluster_index]][-1])
            else:
                exemplar_set.append(dataset[cluster_label_2_feature_centroid_index[cluster_index]][0].numpy())

        return exemplar_set

    ####----CLASSIFICATION----####

    def classify_with_exemplars(self, x, model, classifying_approach="NCM", allowed_classes=None, **kwargs):

        # Set model to eval()-mode
        mode = model.training
        model.eval()
        assert len(self.exemplar_sets) > 0 and len(self.exemplar_sets[0]) > 0
        pred = None
        if "NCM" in classifying_approach:
            pred = self.NCM_classify(x, model, allowed_classes, **kwargs)
        elif "centroid" in classifying_approach:
            pred = self.multi_centroid_classify(x, model, allowed_classes)
        else:
            raise RuntimeError('classifying_approach is illegal: {}'.format(classifying_approach))
        model.train(mode=mode)
        return pred

    def NCM_classify(self, x, model, allowed_classes=None, **kwargs):
        """Classify images by nearest-means-of-exemplars (after transform to feature representation)

                INPUT:      x = <tensor> of size (bsz,ich,isz,isz) with input image batch
                            allowed_classes = None or <list> containing all "active classes" between which should be chosen

                OUTPUT:     preds = <tensor> of size (bsz,)"""
        # Do the exemplar-means need to be recomputed?
        batch_size = x.size(0)
        if self.compute_means:
            exemplar_means = []  # --> list of 1D-tensors (of size [feature_size]), list is of length [n_classes]
            exemplar_dataset = ExemplarDataset_for_exemplars(self.exemplar_sets, self.store_original_imgs,
                                                             img_transform=self._img_transform_for_val)
            # for class_id in new_classes:
            #     # create new dataset containing only all examples of this class
            #     print("construct_exemplar_set class_id:", class_id)
            #     class_dataset = SubDataset(original_dataset=train_dataset_for_EM,
            #                                sub_labels=[class_id])
            #     # based on this dataset, construct new exemplar-set for this class
            #     self.exemplar_manager.construct_exemplar_set(class_dataset, self.ddc_model.model,
            #                                                  exemplars_per_class, self.cfg.TRAIN.BATCH_SIZE,
            #                                                  self.cfg.TRAIN.NUM_WORKERS)
            for class_id in range(len(self.exemplar_sets)):
                class_dataset = SubDataset(original_dataset=exemplar_dataset, sub_labels=[class_id])
                exemplars = []
                # Collect all exemplars in P_y into a <tensor> and extract their features
                if self.store_original_imgs:
                    for (transformed_imgs, _, _) in class_dataset:
                        exemplars.append(transformed_imgs)
                else:
                    for (stored_imgs, _) in class_dataset:
                        exemplars.append(stored_imgs)
                exemplars = torch.stack(exemplars).to(self._device)
                features = self._get_extracted_features(model, exemplars)
                if self._norm_exemplars:
                    features = F.normalize(features, p=2, dim=1)
                # Calculate their mean and add to list
                mu_y = features.mean(dim=0, keepdim=True)
                if self._norm_exemplars:
                    mu_y = F.normalize(mu_y, p=2, dim=1)
                exemplar_means.append(mu_y.squeeze())  # -> squeeze removes all dimensions of size 1
            # Update model's attributes
            self.exemplar_means = exemplar_means
            self.compute_means = False

        # Reorganize the [exemplar_means]-<tensor>
        exemplar_means = self.exemplar_means if allowed_classes is None else [
            self.exemplar_means[i] for i in allowed_classes
        ]
        means = torch.stack(exemplar_means)  # (n_classes, feature_size)
        means = torch.stack([means] * batch_size)  # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2)  # (batch_size, feature_size, n_classes)

        # Extract features for input data (and reorganize)
        feature = self._get_extracted_features(model, x)  # (batch_size, feature_size)
        if self._norm_exemplars:
            feature = F.normalize(feature, p=2, dim=1)
        feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
        feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)

        # For each data-point in [x], find which exemplar-mean is closest to its extracted features
        dists = (feature - means).pow(2).sum(dim=1).squeeze()  # (batch_size, n_classes)
        _, preds = dists.min(1)
        if allowed_classes:
            preds = torch.Tensor([allowed_classes[i] for i in preds]).to(self._device)

        return preds

    def multi_centroid_classify(self, x, model, allowed_classes):
        """Classify images by multi-centroids-classify (after transform to feature representation)

        INPUT:      x = <tensor> of size (bsz,ich,isz,isz) with input image batch
                    allowed_classes = None or <list> containing all "active classes" between which should be chosen

        OUTPUT:     preds = <tensor> of size (bsz,)"""
        # Do the exemplar-means need to be recomputed?
        batch_size = x.size(0)
        all_classes = len(self.exemplar_sets)
        centroid_features = []  # --> list of 1D-tensors (of size [feature_size]), list is of length [n_classes]
        centroid_numbers = []
        if self.recompute_centroid_feature:
            exemplar_means = []  # --> list of 1D-tensors (of size [feature_size]), list is of length [n_classes]
            exemplar_dataset = ExemplarDataset_for_exemplars(self.exemplar_sets, self.store_original_imgs,
                                                             img_transform=self._img_transform_for_val)
            for class_id in range(len(self.exemplar_sets)):
                class_dataset = SubDataset(original_dataset=exemplar_dataset, sub_labels=[class_id])
                centroid_numbers.append(len(self.exemplar_sets[class_id]))
                exemplars = []
                # Collect all exemplars in P_y into a <tensor> and extract their features
                if self.store_original_imgs:
                    for (transformed_imgs, _, _) in class_dataset:
                        exemplars.append(transformed_imgs)
                else:
                    for (stored_imgs, _) in class_dataset:
                        exemplars.append(stored_imgs)
                exemplars = torch.stack(exemplars).to(self._device)
                features = self._get_extracted_features(model, exemplars)
                if self._norm_exemplars:
                    features = F.normalize(features, p=2, dim=1)
                centroid_features.append(features)
            self.centroid_features = centroid_features
            self.recompute_centroid_feature = False
        allowed_centroid_features = self.centroid_features if allowed_classes is None else [
            self.centroid_features[i] for i in allowed_classes
        ]
        allowed_classes_number = len(allowed_centroid_features)
        allowed_centroid_features = torch.stack(allowed_centroid_features)  # (n_classes, n_centroids, feature_size)
        allowed_centroid_features = torch.stack(
            [allowed_centroid_features] * batch_size)  # (batch_size, n_classes, n_centroids, feature_size)
        feature = self._get_extracted_features(model, x)
        if self._norm_exemplars:
            feature = F.normalize(feature, p=2, dim=1)

        # Reorganize the [exemplar_means]-<tensor>
        feature = torch.stack([feature] * allowed_centroid_features.size(1))  # (n_classes, batch_size, feature_size)
        feature = feature.transpose(0, 1)  # (batch_size, n_classes, feature_size)
        feature = torch.stack(
            [feature] * allowed_centroid_features.size(2))  # (n_centroids, batch_size, n_classes, feature_size)
        feature = feature.transpose(0, 1)  # (batch_size, n_centroids, n_classes, feature_size)
        feature = feature.transpose(1, 2)  # (batch_size, n_classes, n_centroids, feature_size)
        dists = (feature - allowed_centroid_features)  # (batch_size, n_classes, n_centroids, feature_size)
        dists = dists.pow(2).sum(dim=3)  # (batch_size, n_classes, n_centroids)
        # class_dist_min_values = []                                           # (batch_size, n_classes)
        # class_dist_max_values = []
        # class_dist_min_preds = []                                            # (batch_size, n_classes)
        # class_dist_max_preds = []
        preds = []
        for sample_id in range(batch_size):
            min_values, min_preds = dists[sample_id].min(1)  # (1, n_classes)
            max_values, max_preds = dists[sample_id].max(1)  # (1, n_classes)
            # class_dist_min_values.append(min_values)
            # class_dist_min_preds.append(min_preds)
            # class_dist_max_preds.append(max_preds)
            # class_dist_max_values.append(max_values)
            wanted_index = self._multi_centroid_classify_func(min_values, max_values, allowed_classes_number)
            preds.append(wanted_index)
        # max_values
        # for index in range(allowed_classes_number):

        if allowed_classes:
            preds = [allowed_classes[i] for i in preds]

        return torch.IntTensor(preds).to(self._device)

    @staticmethod
    def _multi_centroid_classify_func(min_values, max_values, allowed_classes_number):
        wanted_index = -1
        max_dis_dif = float("-inf")
        if allowed_classes_number == 1:
            return 0
        for index in range(allowed_classes_number):
            max_dis_current_class = max_values[index]
            min_dis_dif_class = get_min_dis_dif_class(min_values, index, allowed_classes_number)
            dis_dif = min_dis_dif_class - max_dis_current_class
            if max_dis_dif < dis_dif:
                max_dis_dif = dis_dif
                wanted_index = index

        str = f"max_dis_dif: {max_dis_dif} || wanted_index: {wanted_index} || " \
              f"allowed_classes_number: {allowed_classes_number}"
        # print(str)
        assert 0 <= wanted_index < allowed_classes_number, str
        return wanted_index

    def get_ExemplarDataset(self, for_train=True, Two_transform=False):
        if Two_transform:
            assert for_train
            exemplar_dataset = ExemplarDataset(self.exemplar_sets, is_original_img=self.store_original_imgs,
                                               img_transform=self._img_transform_for_train, Two_transform=True)
        else:
            if for_train:
                exemplar_dataset = ExemplarDataset(self.exemplar_sets, is_original_img=self.store_original_imgs,
                                                   img_transform=self._img_transform_for_train)
            else:
                exemplar_dataset = ExemplarDataset(self.exemplar_sets, is_original_img=self.store_original_imgs,
                                                   img_transform=self._img_transform_for_val)
        return exemplar_dataset
        pass
