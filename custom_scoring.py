import numpy as np


def dodrans(class_count_list, true_labels):
    return class_count_list ** (3 / 4)


def entropy(class_count_list, true_labels):
    return -np.multiply(class_count_list, np.log(class_count_list / len(true_labels)))
