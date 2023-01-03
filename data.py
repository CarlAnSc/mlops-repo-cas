import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torchvision import transforms


def mnist():
    # exchange with the corrupted mnist dataset
    train_img = [[],[],[],[],[]]
    train_labels = [[], [], [], [], []]
    for i in [0,1,2,3,4]:
        dir = f'corruptmnist/train_{i}.npz'
        train_img[i] = np.load(dir, mmap_mode="r")['images']
        train_labels[i] = np.load(dir, mmap_mode="r")['labels']

    test_img = np.load('corruptmnist/test.npz', mmap_mode="r")['images']
    test_labels = np.load('corruptmnist/test.npz', mmap_mode="r")['labels']

    train = [np.concatenate(train_img), np.concatenate(train_labels)]
    test =  [np.array(test_img), np.array(test_labels)]
    return train, test

#mnist()

print('erg oinreglk nwrlgk nslrkng lknrg lsknsrlkng lkrsng ')
# exit() ved "invalid syntax"