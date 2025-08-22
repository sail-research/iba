import csv
import os
import argparse
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import logging
import torchvision
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

from itertools import product
import math
import copy
import time
import logging
import pickle
import random

pattern_dict = {
    'cifar10': {
        '0_poison_pattern': [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]],
        '1_poison_pattern': [[0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14]],
        '2_poison_pattern': [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5]],
        '3_poison_pattern': [[4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14]],
    },
    'mnist': {
        '0_poison_pattern': [[0, 0], [0, 1], [0, 2], [0, 3]],
        '1_poison_pattern': [[0, 6], [0, 7], [0, 8], [0, 9]],
        '2_poison_pattern': [[3, 0], [3, 1], [3, 2], [3, 3]],
        '3_poison_pattern': [[3, 6], [3, 7], [3, 8], [3, 9]],
    },
    'tiny-imagenet': {
        '0_poison_pattern': [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2], [0, 3], [1, 3], [0, 4], [1, 4], [0, 5], [1, 5], [0, 6], [1, 6], [0, 7], [1, 7], [0, 8], [1, 8], [0, 9], [1, 9]],
        '1_poison_pattern': [[0, 12], [1, 12], [0, 13], [1, 13], [0, 14], [1, 14], [0, 15], [1, 15], [0, 16], [1, 16], [0, 17], [1, 17], [0, 18], [1, 18], [0, 19], [1, 19], [0, 20], [1, 20], [0, 21], [1, 21]],
        '2_poison_pattern': [[4, 0], [5, 0], [4, 1], [5, 1], [4, 2], [5, 2], [4, 3], [5, 3], [4, 4], [5, 4], [4, 5], [5, 5], [4, 6], [5, 6], [4, 7], [5, 7], [4, 8], [5, 8], [4, 9], [5, 9]],
        '3_poison_pattern': [[4, 12], [5, 12], [4, 13], [5, 13], [4, 14], [5, 14], [4, 15], [5, 15], [4, 16], [5, 16], [4, 17], [5, 17], [4, 18], [5, 18], [4, 19], [5, 19], [4, 20], [5, 20], [4, 21], [5, 21]],
    }
}
def add_pixel_pattern(image, dataset, device):
    pattern_s = pattern_dict[dataset]
    poison_patterns = []
    for i in range(0,4):
        poison_patterns = poison_patterns + pattern_s[str(i) + '_poison_pattern']
    if dataset in ['cifar10', 'timagenet']:
        for i in range(0,len(poison_patterns)):
            pos = poison_patterns[i]
            image[0][pos[0]][pos[1]] = 1
            image[1][pos[0]][pos[1]] = 1
            image[2][pos[0]][pos[1]] = 1
    elif dataset == 'mnist':
        for i in range(0, len(poison_patterns)):
            pos = poison_patterns[i]
            image[0][pos[0]][pos[1]] = 1
    return image.to(device)

def get_poison_batch(bptt, dataset, device, target_label=1, poisoning_per_batch=8, evaluation=False, target_transform=None):
        images, targets = bptt
        poison_count= 0
        new_images=images
        new_targets=targets
        original_imgs, bd_imgs = [], []
        for index in range(0, len(images)):
            if evaluation: # poison all data when testing
                # if adversarial_index == -1:
                original_imgs.append(images[index]) 
                new_targets[index] = target_label
                new_images[index] = add_pixel_pattern(images[index], dataset, device)
                poison_count+=1
                # if adversarial_index == -1:
                bd_imgs.append(new_images[index])


            else: # poison part of data when training
                if index < poisoning_per_batch:
                    original_imgs.append(images[index])
                    new_targets[index] = target_transform(targets[index])
                    new_images[index] = add_pixel_pattern(images[index], dataset, device)
                    poison_count += 1
                    bd_imgs.append(new_images[index])
                else:
                    new_images[index] = images[index]
                    new_targets[index]= targets[index]
        new_images = new_images.to(device)
        new_targets = new_targets.to(device).long()
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_images,new_targets,poison_count, original_imgs, bd_imgs

def get_dba_poison(inputs, opt):
    pattern_s = pattern_dict[opt.dataset]
    poison_patterns = []
    new_images = inputs
    original_imgs, bd_imgs = [], []
    for index in range(0, len(original_imgs)):
        ori_image = original_imgs[index]
        image = copy.deepcopy(ori_image)
        for i in range(0,4):
            poison_patterns = poison_patterns + pattern_s[str(i) + '_poison_pattern']
        if opt.dataset in ['cifar10', 'timagenet']:
            for i in range(0,len(poison_patterns)):
                pos = poison_patterns[i]
                image[0][pos[0]][pos[1]] = 1
                image[1][pos[0]][pos[1]] = 1
                image[2][pos[0]][pos[1]] = 1
        elif opt.dataset == 'mnist':
            for i in range(0, len(poison_patterns)):
                pos = poison_patterns[i]
                image[0][pos[0]][pos[1]] = 1
        new_images[index] = image
        new_images = new_images.to(opt.device)
    return new_images

from datasets import MNIST_truncated, EMNIST_truncated, CIFAR10_truncated, TinyImageNet_truncated, ImageFolderTruncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #output = F.log_softmax(x, dim=1)
        return x
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def record_net_data_stats(y_train, net_dataidx_map):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts

def load_mnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_emnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    emnist_train_ds = EMNIST_truncated(datadir, train=True, download=True, transform=transform)
    emnist_test_ds = EMNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = emnist_train_ds.data, emnist_train_ds.target
    X_test, y_test = emnist_test_ds.data, emnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)

def load_tinyimagenet_data(datadir):
            
    transform = transforms.Compose([transforms.ToTensor()])

    tinyimagenet_train_ds = TinyImageNet_truncated(datadir, train=True, download=True, transform=transform)
    tinyimagenet_test_ds = TinyImageNet_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = tinyimagenet_train_ds.data, tinyimagenet_train_ds.target
    X_test, y_test = tinyimagenet_test_ds.data, tinyimagenet_test_ds.target
    
    return (X_train, y_train, X_test, y_test)


# def partition_data(dataset, datadir, partition, n_nets, alpha, args):
#     print("Start partition_data")
#     if dataset == 'mnist':
#         X_train, y_train, X_test, y_test = load_mnist_data(datadir)
#         n_train = X_train.shape[0]
#     elif dataset == 'emnist':
#         X_train, y_train, X_test, y_test = load_emnist_data(datadir)
#         n_train = X_train.shape[0]
        
#     elif dataset == 'tiny-imagenet':
#         # print(f"Check dataset: {dataset}")
        
#         # X_train, y_train, X_test, y_test = load_tinyimagenet_data(datadir)
#         # n_train = X_train.shape[0]
#         _train_dir = './data/tiny-imagenet-200/train'
        
#         # cinic_mean = [0.47889522, 0.47227842, 0.43047404]
#         # cinic_std = [0.24205776, 0.23828046, 0.25874835]
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])

#         trainset = datasets.ImageFolder(_train_dir, 
#                                         transform=transforms.Compose([
#                                                  transforms.RandomResizedCrop(224),
#                                                  transforms.RandomHorizontalFlip(),
#                                                  transforms.ToTensor(),
#                                                  normalize,
#                                                  ]))
        
#         # tiny_mean=[0.485, 0.456, 0.406],
#         # tiny_std=[0.229, 0.224, 0.225],
#         # _data_transforms = {
#         #     'train': transforms.Compose([
#         #         transforms.Resize(224),
#         #         transforms.RandomHorizontalFlip(),
#         #         transforms.ToTensor(),
#         #     ]),
#         #     'val': transforms.Compose([
#         #         transforms.Resize(224),
#         #         transforms.ToTensor(),
#         #     ]),
#         # }
#         # trainset = ImageFolderTruncated(_train_dir, transform=_data_transforms['train'])
        
#         # y_train = trainset.get_train_labels
#         # n_train = y_train.shape[0]
#         trainset = datasets.ImageFolder(_train_dir, 
#                                         transform=transforms.Compose([
#                                                  transforms.RandomResizedCrop(224),
#                                                  transforms.RandomHorizontalFlip(),
#                                                  transforms.ToTensor(),
#                                                  normalize,
#                                                  ]))
        
#         y_train = np.array([img[-1] for img in trainset.imgs])
#         n_train = y_train.shape[0]
        
        
        
        
#         # import IPython
#         # IPython.embed()
        
#         # exit(0)
        
#     elif dataset.lower() == 'cifar10':
#         X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
#         # if args.poison_type == "howto":
#         #     sampled_indices_train = [874, 49163, 34287, 21422, 48003, 47001, 48030, 22984, 37533, 41336, 3678, 37365,
#         #                                 19165, 34385, 41861, 39824, 561, 49588, 4528, 3378, 38658, 38735, 19500,  9744, 47026, 1605, 389]
#         #     sampled_indices_test = [32941, 36005, 40138]
#         #     cifar10_whole_range = np.arange(X_train.shape[0])
#         #     remaining_indices = [i for i in cifar10_whole_range if i not in sampled_indices_train+sampled_indices_test]
#         #     X_train = X_train[sampled_indices_train, :, :, :]
#         #     logger.info("@@@ Poisoning type: {} Num of Remaining Data Points (excluding poisoned data points): {}".format(
#         #                                 args.poison_type, 
#         #                                 X_train.shape[0]))
        
#         # # 0-49999 normal cifar10, 50000 - 50735 wow airline
#         # if args.poison_type == 'southwest+wow':
#         #     with open('./saved_datasets/wow_images_new_whole.pkl', 'rb') as train_f:
#         #         saved_wow_dataset_whole = pickle.load(train_f)
#         #     X_train = np.append(X_train, saved_wow_dataset_whole, axis=0)
#         n_train = X_train.shape[0]

#     elif dataset == 'cinic10':
#         _train_dir = './data/cinic10/cinic-10-trainlarge/train'
#         cinic_mean = [0.47889522, 0.47227842, 0.43047404]
#         cinic_std = [0.24205776, 0.23828046, 0.25874835]
#         trainset = ImageFolderTruncated(_train_dir, transform=transforms.Compose([transforms.ToTensor(),
#                                                             transforms.Lambda(lambda x: F.pad(Variable(x.unsqueeze(0), 
#                                                                                             requires_grad=False),
#                                                                                             (4,4,4,4),mode='reflect').data.squeeze()),
#                                                             transforms.ToPILImage(),
#                                                             transforms.RandomCrop(32),
#                                                             transforms.RandomHorizontalFlip(),
#                                                             transforms.ToTensor(),
#                                                             transforms.Normalize(mean=cinic_mean,std=cinic_std),
#                                                             ]))
#         y_train = trainset.get_train_labels
#         n_train = y_train.shape[0]
        
#     elif dataset == "shakespeare":
#         net_dataidx_map = {}
#         with open(datadir[0]) as json_file:
#             train_data = json.load(json_file)

#         with open(datadir[1]) as json_file:
#             test_data = json.load(json_file)

#         for j in range(n_nets):
#             client_user_name = train_data["users"][j]

#             client_train_data = train_data["user_data"][client_user_name]['x']
#             num_samples_train = len(client_train_data)
#             net_dataidx_map[j] = [i for i in range(num_samples_train)] # TODO: this is a dirty hack. needs modification
#         return None, net_dataidx_map, None

#     if partition == "homo":
#         idxs = np.random.permutation(n_train)
#         batch_idxs = np.array_split(idxs, n_nets)
#         net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

#     elif partition == "hetero-dir":
#         print("Start partition hetero-dir")
#         min_size = 0
#         # K = 200
#         from collections import Counter
#         counter_class = Counter(y_train)
#         # print(f"Counter class for dataset: {dataset}: {counter_class}")
#         # K = len(counter_class)
        
#         if dataset == "tiny-imagenet":
#             # K = 1000
#             K = 200
#         elif dataset == "cifar10":
#             K = 10
#         N = y_train.shape[0]
#         net_dataidx_map = {}
        
#         # print(N)
#         # import IPython
#         # IPython.embed()
        
#         while (min_size < 10) or (dataset == 'mnist' and min_size < 100):
#             idx_batch = [[] for _ in range(n_nets)]
#             # for each class in the dataset
            
#             for k in range(K):
#                 idx_k = np.where(y_train == k)[0]
#                 np.random.shuffle(idx_k)
#                 proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
#                 ## Balance
#                 proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
#                 proportions = proportions/proportions.sum()
#                 proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
#                 idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
#                 min_size = min([len(idx_j) for idx_j in idx_batch])

#         for j in range(n_nets):
#             np.random.shuffle(idx_batch[j])
#             net_dataidx_map[j] = idx_batch[j]
        
#         for j in range(n_nets):
#             # print("Number of data points in client {}: {}".format(j, ))
#             y_class_j = y_train[net_dataidx_map[j]]
#             # print("Number of data points in each class in client {} total {} {}".format(j, len(net_dataidx_map[j]), Counter(y_class_j)))
        
#         # print("Number of data points for all clients: {}".format(sum([len(net_dataidx_map[j]) for j in range(n_nets)])))
#         # print("---"*30)
        
#         # print("Debug IPython")
#         # import IPython
#         # IPython.embed()
        
#         if dataset == 'cifar10':
#             if args.poison_type == 'howto' or args.poison_type == 'greencar-neo':
#                 green_car_indices = [874, 49163, 34287, 21422, 48003, 47001, 48030, 22984, 37533, 41336, 3678, 37365, 19165, 34385, 41861, 39824, 561, 49588, 4528, 3378, 38658, 38735, 19500,  9744, 47026, 1605, 389] + [32941, 36005, 40138]
#                 #sanity_check_counter = 0
#                 for k, v in net_dataidx_map.items():
#                     remaining_indices = [i for i in v if i not in green_car_indices]
#                     #sanity_check_counter += len(remaining_indices)
#                     net_dataidx_map[k] = remaining_indices

#             #logger.info("Remaining total number of data points : {}".format(sanity_check_counter))
#             # sanity check:
#             #aggregated_val = []
#             #for val in net_dataidx_map.values():
#             #    aggregated_val+= val
#             #black_box_indices = [i for i in range(50000) if i not in aggregated_val]
#             #logger.info("$$$$$$$$$$$$$$ recovered black box indices: {}".format(black_box_indices))
#             #exit()
#     traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

#     return net_dataidx_map


# def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
#     if dataset in ('mnist', 'emnist', 'cifar10'):
#         if dataset == 'mnist':
#             dl_obj = MNIST_truncated

#             transform_train = transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))])

#             transform_test = transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))])
#         if dataset == 'emnist':
#             dl_obj = EMNIST_truncated

#             transform_train = transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))])

#             transform_test = transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))])

#         elif dataset == 'cifar10':
#             dl_obj = CIFAR10_truncated

#             normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
#                                 std=[x/255.0 for x in [63.0, 62.1, 66.7]])
#             transform_train = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Lambda(lambda x: F.pad(
#                                     Variable(x.unsqueeze(0), requires_grad=False),
#                                     (4,4,4,4),mode='reflect').data.squeeze()),
#                 transforms.ToPILImage(),
#                 transforms.RandomCrop(32),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 normalize,
#                 ])
#             # data prep for test set
#             transform_test = transforms.Compose([transforms.ToTensor(),normalize])
            
#         # elif dataset == 'tiny-imagenet':

#         #     _data_transforms = {
#         #         'train': transforms.Compose([
#         #             transforms.Resize(224),
#         #             transforms.RandomHorizontalFlip(),
#         #             transforms.ToTensor(),
#         #         ]),
#         #         'val': transforms.Compose([
#         #             transforms.Resize(224),
#         #             transforms.ToTensor(),
#         #         ]),
#         #     }
        
#         #     dl_obj = ImageFolderTruncated

        
        
#         #     transform_train = _data_transforms['train']

#         #     transform_test = _data_transforms['val']
#     elif dataset == "tiny-imagenet":
#         dl_obj = ImageFolderTruncated

#         _train_dir = "data/tiny-imagenet-200/train" #imagenet data location
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225]) 
#         transform_train = transform=transforms.Compose([
#                          transforms.RandomResizedCrop(224),
#                          transforms.RandomHorizontalFlip(),
#                          transforms.ToTensor(),
#                          normalize
#                      ])
#     transform_test = transforms.Compose([transforms.ToTensor(),normalize])
    
#     if dataset == 'tiny-imagenet':
#         # _train_dir = './data/tiny-imagenet-200/train'
#         # _val_dir = './data/tiny-imagenet-200/val'
#         # train_ds = dl_obj(root=_train_dir, dataidxs=dataidxs, transform=transform_train)
        
#         # test_ds = dl_obj(root=_val_dir, transform=transform_test)
#         train_ds = dl_obj(_train_dir, dataidxs=dataidxs, transform=transform_train)
#         #test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
#         train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
#         test_dl = None
#     else:
#         train_ds = dl_obj(datadir, train=True, dataidxs=dataidxs, transform=transform_train, download=True)
#         test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
#     # train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
#     # test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

#     return train_dl, test_dl

def partition_data(dataset, datadir, partition, n_nets, alpha, args):
    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
        n_train = X_train.shape[0]
    elif dataset == 'emnist':
        X_train, y_train, X_test, y_test = load_emnist_data(datadir)
        n_train = X_train.shape[0]
    elif dataset.lower() == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
        # TODO(hwang): this is actually a bad solution, please verify if we really want this
        # if args.poison_type == "howto":
        #     sampled_indices_train = [874, 49163, 34287, 21422, 48003, 47001, 48030, 22984, 37533, 41336, 3678, 37365,
        #                                 19165, 34385, 41861, 39824, 561, 49588, 4528, 3378, 38658, 38735, 19500,  9744, 47026, 1605, 389]
        #     sampled_indices_test = [32941, 36005, 40138]
        #     cifar10_whole_range = np.arange(X_train.shape[0])
        #     remaining_indices = [i for i in cifar10_whole_range if i not in sampled_indices_train+sampled_indices_test]
        #     X_train = X_train[sampled_indices_train, :, :, :]
        #     logger.info("@@@ Poisoning type: {} Num of Remaining Data Points (excluding poisoned data points): {}".format(
        #                                 args.poison_type, 
        #                                 X_train.shape[0]))
        
        # # 0-49999 normal cifar10, 50000 - 50735 wow airline
        # if args.poison_type == 'southwest+wow':
        #     with open('./saved_datasets/wow_images_new_whole.pkl', 'rb') as train_f:
        #         saved_wow_dataset_whole = pickle.load(train_f)
        #     X_train = np.append(X_train, saved_wow_dataset_whole, axis=0)
        n_train = X_train.shape[0]

    elif dataset == 'cinic10':
        _train_dir = './data/cinic10/cinic-10-trainlarge/train'
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        trainset = ImageFolderTruncated(_train_dir, transform=transforms.Compose([transforms.ToTensor(),
                                                            transforms.Lambda(lambda x: F.pad(Variable(x.unsqueeze(0), 
                                                                                            requires_grad=False),
                                                                                            (4,4,4,4),mode='reflect').data.squeeze()),
                                                            transforms.ToPILImage(),
                                                            transforms.RandomCrop(32),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=cinic_mean,std=cinic_std),
                                                            ]))
        y_train = trainset.get_train_labels
        n_train = y_train.shape[0]
    elif dataset.lower() == "tiny-imagenet":
        # _train_dir = "~/data/train" #imagenet data location
        _train_dir = "data/tiny-imagenet-200/train" #imagenet data location
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        trainset = datasets.ImageFolder(_train_dir, 
                                        transform=transforms.Compose([
                                                 transforms.RandomResizedCrop(224),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 normalize,
                                                 ]))

        # K = 1000
        # class_counter = {}
        # for i in range(K):
        #     class_counter[i] = 0
        # for idx, img in enumerate(trainset.imgs):
        #     class_counter[img[-1]] += 1
        # for idx, (k, v) in enumerate(class_counter.items()):
        #     logger.info("$$$$$ Class: {} Num Count: {}".format(k, v))
        # logger.info('#########################################################')
        # logger.info("Key with min val: {}".format(min(class_counter, key = lambda k: class_counter[k])))
        # min_class_index = min(class_counter, key = lambda k: class_counter[k])
        # for idx, (k, v) in enumerate(trainset.class_to_idx.items()):
        #     if v == min_class_index:
        #         min_class_name = k
        #         break
        # logger.info("class_to_idx: {}, {}".format(min_class_name, trainset.class_to_idx[min_class_name]))
        #exit()

        #y_train = trainset.get_train_labels
        y_train = np.array([img[-1] for img in trainset.imgs])
        n_train = y_train.shape[0]

    elif dataset == "shakespeare":
        net_dataidx_map = {}
        with open(datadir[0]) as json_file:
            train_data = json.load(json_file)

        with open(datadir[1]) as json_file:
            test_data = json.load(json_file)

        for j in range(n_nets):
            client_user_name = train_data["users"][j]

            client_train_data = train_data["user_data"][client_user_name]['x']
            num_samples_train = len(client_train_data)
            net_dataidx_map[j] = [i for i in range(num_samples_train)] # TODO: this is a dirty hack. needs modification
        return None, net_dataidx_map, None

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero-dir":

        #TODO: 
        min_size = 0
        if dataset == "tiny-imagenet":
            # K = 1000
            K = 200
        elif dataset in ["cifar10", "mnist"]:
            K = 10
        N = y_train.shape[0]
        net_dataidx_map = {}

        while (min_size < 10) or (dataset == 'mnist' and min_size < 100):
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        if dataset == 'cifar10':
            if args.poison_type == 'howto' or args.poison_type == 'greencar-neo':
                green_car_indices = [874, 49163, 34287, 21422, 48003, 47001, 48030, 22984, 37533, 41336, 3678, 37365, 19165, 34385, 41861, 39824, 561, 49588, 4528, 3378, 38658, 38735, 19500,  9744, 47026, 1605, 389] + [32941, 36005, 40138]
                #sanity_check_counter = 0
                for k, v in net_dataidx_map.items():
                    remaining_indices = [i for i in v if i not in green_car_indices]
                    #sanity_check_counter += len(remaining_indices)
                    net_dataidx_map[k] = remaining_indices

            #logger.info("Remaining total number of data points : {}".format(sanity_check_counter))
            # sanity check:
            #aggregated_val = []
            #for val in net_dataidx_map.values():
            #    aggregated_val+= val
            #black_box_indices = [i for i in range(50000) if i not in aggregated_val]
            #logger.info("$$$$$$$$$$$$$$ recovered black box indices: {}".format(black_box_indices))
            #exit()
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return net_dataidx_map

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    if dataset in ('mnist', 'emnist', 'cifar10'):
        if dataset == 'mnist':
            dl_obj = MNIST_truncated

            transform_train = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

            transform_test = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])
        if dataset == 'emnist':
            dl_obj = EMNIST_truncated

            transform_train = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

            transform_test = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

        elif dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                                    Variable(x.unsqueeze(0), requires_grad=False),
                                    (4,4,4,4),mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
            # data prep for test set
            transform_test = transforms.Compose([transforms.ToTensor(),normalize])

    elif dataset == "tiny-imagenet":
        dl_obj = ImageFolderTruncated

        _train_dir = "data/tiny-imagenet-200/train" #imagenet data location
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]) 
        transform_train = transform=transforms.Compose([
                         transforms.RandomResizedCrop(224),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         normalize
                     ])
        transform_test = transforms.Compose([transforms.ToTensor(),normalize])

    if dataset == "tiny-imagenet":
        train_ds = dl_obj(_train_dir, dataidxs=dataidxs, transform=transform_train)
        #test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
        #test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)
    else:
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        #test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
        #test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)        
        
    return train_dl, None

# def get_dataloader_normal_case(dataset, datadir, train_bs, test_bs, 
#                                 dataidxs=None, 
#                                 user_id=0, 
#                                 num_total_users=200,
#                                 poison_type="southwest",
#                                 ardis_dataset=None,
#                                 attack_case='normal-case'):
#     if dataset in ('mnist', 'emnist', 'cifar10'):
#         if dataset == 'mnist':
#             dl_obj = MNIST_truncated

#             transform_train = transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))])

#             transform_test = transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))])
#         if dataset == 'emnist':
#             dl_obj = EMNIST_NormalCase_truncated

#             transform_train = transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))])

#             transform_test = transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))])
#         elif dataset == 'cifar10':
#             dl_obj = CIFAR10NormalCase_truncated

#             normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
#                                 std=[x/255.0 for x in [63.0, 62.1, 66.7]])
#             transform_train = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Lambda(lambda x: F.pad(
#                                     Variable(x.unsqueeze(0), requires_grad=False),
#                                     (4,4,4,4),mode='reflect').data.squeeze()),
#                 transforms.ToPILImage(),
#                 transforms.RandomCrop(32),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 normalize,
#                 ])
#             # data prep for test set
#             transform_test = transforms.Compose([transforms.ToTensor(),normalize])

#         # this only supports cifar10 right now, please be super careful when calling it using other datasets
#         # def __init__(self, root, 
#         #                 dataidxs=None, 
#         #                 train=True, 
#         #                 transform=None, 
#         #                 target_transform=None, 
#         #                 download=False,
#         #                 user_id=0,
#         #                 num_total_users=200,
#         #                 poison_type="southwest"):        
#         train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True,
#                                     user_id=user_id, num_total_users=num_total_users, poison_type=poison_type,
#                                     ardis_dataset_train=ardis_dataset, attack_case=attack_case)
        
#         test_ds = None #dl_obj(datadir, train=False, transform=transform_test, download=True)

#         train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
#         test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

#     return train_dl, test_dl

def _get_dataloader_kwargs(args):
    """Get common DataLoader kwargs based on CUDA availability."""
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    return {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}


def _load_mnist_dataset(args, num_sampled_data_points):
    """Load and prepare MNIST/EMNIST dataset."""
    # Prepare transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))  # Uncomment if normalization needed
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Sample data points
    total_samples = len(train_dataset)
    sampled_indices = np.random.choice(total_samples, num_sampled_data_points, replace=False)
    
    # Subset the training data
    train_dataset.data = train_dataset.data[sampled_indices, :, :]
    train_dataset.targets = np.array(train_dataset.targets)[sampled_indices]
    
    # Create data loaders
    kwargs = _get_dataloader_kwargs(args)
    vanilla_test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    clean_train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    
    return {
        'poisoned_train_loader': None,
        'vanilla_test_loader': vanilla_test_loader,
        'targetted_task_test_loader': None,
        'clean_train_loader': clean_train_loader,
        'num_dps_poisoned_dataset': train_dataset.data.shape[0]
    }


def _load_tiny_imagenet_dataset(args, num_sampled_data_points):
    """Load and prepare Tiny ImageNet dataset."""
    # Normalization parameters
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Transforms
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load validation dataset for testing
    vanilla_test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder("data/tiny-imagenet-200/val", transform_test),
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Load and sample training dataset
    train_dataset = datasets.ImageFolder("data/tiny-imagenet-200/train", transform_train)
    total_datapoints = len(train_dataset.imgs)
    sampled_indices = np.random.choice(total_datapoints, num_sampled_data_points, replace=False)
    
    # Clean up memory
    del train_dataset
    
    # Create truncated dataset
    train_dataset = ImageFolderTruncated(
        root="data/tiny-imagenet-200/train",
        dataidxs=sampled_indices,
        transform=transform_train
    )
    
    logger.info(f"Num dps in poisoned dataset: {len(train_dataset.imgs)}")
    
    # Create data loader
    clean_train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    
    return {
        'poisoned_train_loader': None,
        'vanilla_test_loader': vanilla_test_loader,
        'targetted_task_test_loader': None,
        'clean_train_loader': clean_train_loader,
        'num_dps_poisoned_dataset': num_sampled_data_points
    }


def _load_cifar10_dataset(args, num_sampled_data_points):
    """Load and prepare CIFAR-10 dataset."""
    # Transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load training dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    # Sample and create poisoned dataset
    poisoned_trainset = copy.deepcopy(trainset)
    sampled_indices = np.random.choice(
        poisoned_trainset.data.shape[0], num_sampled_data_points, replace=False
    )
    poisoned_trainset.data = poisoned_trainset.data[sampled_indices, :, :, :]
    poisoned_trainset.targets = np.array(poisoned_trainset.targets)[sampled_indices]
    
    logger.info(f"Num clean data points in the mixed dataset: {num_sampled_data_points}")
    
    # Keep a copy of clean data
    clean_trainset = copy.deepcopy(poisoned_trainset)
    
    # Log dataset information
    logger.info(f"Poisoned trainset data shape: {poisoned_trainset.data.shape}")
    logger.info(f"Poisoned trainset targets shape: {poisoned_trainset.targets.shape}")
    logger.info(f"Sum of poisoned targets: {sum(poisoned_trainset.targets)}")
    
    # Create data loaders
    kwargs = _get_dataloader_kwargs(args)
    poisoned_train_loader = torch.utils.data.DataLoader(
        poisoned_trainset, batch_size=args.batch_size, shuffle=True, **kwargs
    )
    clean_train_loader = torch.utils.data.DataLoader(
        clean_trainset, batch_size=args.batch_size, shuffle=True, **kwargs
    )
    
    # Load test dataset
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    poisoned_testset = copy.deepcopy(testset)
    
    vanilla_test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )
    targetted_task_test_loader = torch.utils.data.DataLoader(
        poisoned_testset, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )
    
    return {
        'poisoned_train_loader': poisoned_train_loader,
        'vanilla_test_loader': vanilla_test_loader,
        'targetted_task_test_loader': targetted_task_test_loader,
        'clean_train_loader': clean_train_loader,
        'num_dps_poisoned_dataset': poisoned_trainset.data.shape[0]
    }


def load_poisoned_dataset(args):
    """Load poisoned dataset based on the specified dataset type.
    
    Args:
        args: Arguments object containing dataset configuration
        
    Returns:
        tuple: (poisoned_train_loader, vanilla_test_loader, targetted_task_test_loader, 
                num_dps_poisoned_dataset, clean_train_loader)
    """
    num_sampled_data_points = args.num_dps_attacker
    
    # Handle fraction parameter for MNIST/EMNIST
    if args.dataset in ("mnist", "emnist"):
        if args.fraction < 1:
            fraction = args.fraction
        else:
            fraction = int(args.fraction)
        
        result = _load_mnist_dataset(args, num_sampled_data_points)
    
    elif args.dataset == 'tiny-imagenet':
        result = _load_tiny_imagenet_dataset(args, num_sampled_data_points)
    
    elif args.dataset == "cifar10":
        result = _load_cifar10_dataset(args, num_sampled_data_points)
    
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Return as tuple to maintain compatibility
    return (
        result['poisoned_train_loader'],
        result['vanilla_test_loader'],
        result['targetted_task_test_loader'],
        result['num_dps_poisoned_dataset'],
        result['clean_train_loader']
    )

def seed_experiment(seed=0):
    # seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #TODO: Do we need deterministic in cudnn ? Double check
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Seeded everything")

def get_grad_mask(model, optimizer, clean_dataloader, ratio=0.5, device="cpu", save_f=False, historical_grad_mask=None, cnt_masks=0):
    """
    Generate a gradient mask based on the given dataset
    This function is employed for Neurotoxin method
    https://proceedings.mlr.press/v162/zhang22w.html
    """        

    model.train()
    model.zero_grad()
    loss_fn = nn.CrossEntropyLoss()
    # Let's assume we have a model trained on clean data and we conduct aggregation for all layer
    for batch_idx, batch in enumerate(clean_dataloader):
        bs = len(batch)
        data, targets = batch
        clean_images, clean_targets = copy.deepcopy(data).to(device), copy.deepcopy(targets).to(device)
        optimizer.zero_grad()
        output = model(clean_images)
        loss_clean = loss_fn(output, clean_targets)
        loss_clean.backward(retain_graph=True)
    mask_grad_list = []
    grad_list = []
    grad_abs_sum_list = []
    k_layer = 0
    for _, parms in model.named_parameters():
        if parms.requires_grad:
            grad_list.append(parms.grad.abs().view(-1))
            grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())
            k_layer += 1

    grad_list = torch.cat(grad_list).to(device)
    
    _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
    mask_flat_all_layer = torch.zeros(len(grad_list)).to(device)
    mask_flat_all_layer[indices] = 1.0

    if historical_grad_mask:
        cummulative_mask = ((cnt_masks-1)/cnt_masks)*historical_grad_mask+(1/cnt_masks)*mask_flat_all_layer
        _, indices = torch.topk(-1*cummulative_mask, int(len(cummulative_mask)*ratio))
        mask_flat_all_layer = torch.zeros(len(grad_list)).to(device)
        mask_flat_all_layer[indices] = 1.0
    else:
        cummulative_mask = copy.deepcopy(grad_list)
    # if save_f:
    #     with open("neurotoxin_logging.csv", "a+") as lf:
    #         writer = csv.writer(lf)
    #         writer.writerow(indices)
    count = 0
    percentage_mask_list = []
    k_layer = 0
    grad_abs_percentage_list = []
    for _, parms in model.named_parameters():
        if parms.requires_grad:
            gradients_length = len(parms.grad.abs().view(-1))

            mask_flat = mask_flat_all_layer[count:count + gradients_length ].to(device)
            mask_grad_list.append(mask_flat.reshape(parms.grad.size()).to(device))

            count += gradients_length

            percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0

            percentage_mask_list.append(percentage_mask1)

            grad_abs_percentage_list.append(grad_abs_sum_list[k_layer]/np.sum(grad_abs_sum_list))

            k_layer += 1
    return mask_grad_list, cummulative_mask

def compute_hessian(model, loss):
    """
    Computes the Hessian matrix of the loss function with respect to the model parameters.

    Args:
        model: PyTorch model object.
        loss: PyTorch tensor object representing the loss value.

    Returns:
        H: Hessian matrix (torch.Tensor object).
    """

    # Initialize the Hessian matrix.
    parameters = list(model.parameters())
    n_params = sum(p.numel() for p in parameters)
    H = torch.zeros(n_params, n_params)

    # Compute the gradient of the loss with respect to the parameters.
    grad_params = torch.autograd.grad(loss, parameters, create_graph=True)

    # Compute the Hessian matrix.
    k = 0
    for i in range(n_params):
        grad2 = torch.autograd.grad(grad_params[i], parameters, create_graph=True)
        for j in range(n_params):
            H[i, j] = grad2[j].view(-1)[k]
        k += 1

    return H

def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.parameters()])

def get_grad_mask_F3BA(model_vec, ori_model_vec, optimizer=None, clean_dataloader=None, ratio=0.01, device="cpu"):
    important_scores = (model_vec - ori_model_vec).mul(ori_model_vec)
    _, indices = torch.topk(-1*important_scores, int(len(important_scores)*ratio))
    mask_flat_all_layer = torch.zeros(len(important_scores))
    mask_flat_all_layer[indices] = 1.0
    return mask_flat_all_layer

def apply_grad_mask(model, mask_grad_list):
    mask_grad_list_copy = iter(mask_grad_list)
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            parms.grad = parms.grad * next(mask_grad_list_copy)

def apply_PGD(model, helper, global_model_copy):
    weight_difference, difference_flat = helper.get_weight_difference(global_model_copy, model.named_parameters())
    clipped_weight_difference, l2_norm = helper.clip_grad(helper.params['s_norm'], weight_difference, difference_flat)
    weight_difference, difference_flat = helper.get_weight_difference(global_model_copy, clipped_weight_difference)
    copy_params(model, weight_difference)

def copy_params(model, target_params_variables):
    for name, layer in model.named_parameters():
        layer.data = copy.deepcopy(target_params_variables[name])