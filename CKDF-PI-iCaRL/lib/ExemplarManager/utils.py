import numpy as np
import torch
from torch.nn import functional as F


def get_mean_nearest_elem(feature_index_list, features_tensors, norm_feature_mean=False):
    feature_mean = torch.mean(features_tensors, dim=0, keepdim=True)
    if norm_feature_mean:
        feature_mean = F.normalize(feature_mean, p=2, dim=1)
    features_dists = features_tensors - feature_mean
    index_selected = np.argmin(torch.norm(features_dists.cpu(), p=2, dim=1))
    return feature_index_list[index_selected]


def get_mean_nearest_elem_index(features, norm_exemplars, to_stored_number):
    class_mean = torch.mean(features, dim=0, keepdim=True)
    if norm_exemplars:
        class_mean = F.normalize(class_mean, p=2, dim=1)
    features_dists = features - class_mean
    index_selected_array = k_smallest_index_argsort(torch.norm(features_dists.cpu(), p=2, dim=1).numpy(), to_stored_number)
    return index_selected_array


def k_smallest_index_argsort(a, k=None):
    if k:
        idx = np.argsort(a)[:k]
    else:
        idx = np.argsort(a)
    return idx


def k_largest_index_argsort(a, k=None):
    if k:
        idx = np.argsort(a)[:-k - 1:-1]
    else:
        idx = np.argsort(a)[::-1]
    return idx


def get_decreasing_value_index(cluster_label_2_feature_numbers, k):
    cluster_label_2_feature_numbers = np.array(cluster_label_2_feature_numbers)
    return k_largest_index_argsort(cluster_label_2_feature_numbers, k)


'''The cluster centroids are sorted according to feature numbers which they includes  and the distances to the 
class_mean, the priority is the feature numbers. '''

'''cluster_label_2_feature_centroid is tensor with shape (n_cluster, feature_size)
   cluster_label_2_feature_numbers is list with shape (n_cluster, type=int)'''


def get_mean_dis_order_cluster_index(class_mean, cluster_label_2_feature_numbers, cluster_label_2_feature_centroid):
    cluster_label_2_feature_numbers_temp = np.array(cluster_label_2_feature_numbers)
    feature_numbers = np.unique(cluster_label_2_feature_numbers_temp)
    ordered_feature_numbers = sorted(feature_numbers, reverse=True)

    # feature_number_dict = {}
    # for cluster_index in range(len(cluster_label_2_feature_numbers)):
    #     if cluster_label_2_feature_numbers[cluster_index] not in feature_number_dict.keys():
    #         feature_number_dict[cluster_label_2_feature_numbers[cluster_index]] = [cluster_index]
    #     else:
    #         feature_number_dict[cluster_label_2_feature_numbers[cluster_index]].append(cluster_index)
    # ordered_feature_numbers = sorted(feature_number_dict.keys(), reverse=True)
    reordered_cluster_index_list = []
    for feature_number in ordered_feature_numbers:
        cluster_index_list = np.where(cluster_label_2_feature_numbers == feature_number)[0]
        feature_number_2_centroids = cluster_label_2_feature_centroid[cluster_index_list]
        # feature_number_2_centroids = None
        # for cluster_index in cluster_index_list:
        #     feature_number_2_centroids = cluster_label_2_feature_centroid[]
        # if feature_number_2_centroids is None:
        #     feature_number_2_centroids = cluster_label_2_feature_centroid[cluster_index]
        # else:
        #     feature_number_2_centroids = torch.vstack((feature_number_2_centroids,
        #                                                cluster_label_2_feature_centroid[cluster_index]))
        features_dists = feature_number_2_centroids - class_mean
        ordered_cluster_index = k_smallest_index_argsort(torch.norm(features_dists, p=2, dim=1).cpu().numpy(), k=None)
        cluster_index_list = np.array(cluster_index_list)[ordered_cluster_index]
        reordered_cluster_index_list.extend(list(cluster_index_list))

    return reordered_cluster_index_list


def get_min_dis_dif_class(min_values, index, allowed_classes_number):
    min_dis_dif_class = None
    for i in range(allowed_classes_number):
        if i == index:
            continue
        else:
            if min_dis_dif_class is None or min_dis_dif_class > min_values[i]:
                min_dis_dif_class = min_values[i]
    return min_dis_dif_class
