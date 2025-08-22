import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.datasets import MNIST, EMNIST, CIFAR10
from torchvision.datasets import DatasetFolder, ImageFolder
from torchvision import transforms

from PIL import Image

import os
import os.path
import sys
import logging
import pickle
import copy


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class MNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        mnist_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            data = mnist_dataobj.train_data
            target = mnist_dataobj.train_labels
        else:
            data = mnist_dataobj.test_data
            target = mnist_dataobj.test_labels

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class EMNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        emnist_dataobj = EMNIST(self.root, split="digits", train=self.train, 
                                transform=self.transform, 
                                target_transform=self.target_transform, 
                                download=self.download)

        if self.train:
            data = emnist_dataobj.train_data
            target = emnist_dataobj.train_labels
        else:
            data = emnist_dataobj.test_data
            target = emnist_dataobj.test_labels

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


# def get_ardis_dataset():
#     # load the data from csv's
#     ardis_images=np.loadtxt('./data/ARDIS/ARDIS_train_2828.csv', dtype='float')
#     ardis_labels=np.loadtxt('./data/ARDIS/ARDIS_train_labels.csv', dtype='float')


#     #### reshape to be [samples][width][height]
#     ardis_images = ardis_images.reshape(ardis_images.shape[0], 28, 28).astype('float32')

#     # labels are one-hot encoded
#     indices_seven = np.where(ardis_labels[:,7] == 1)[0]
#     images_seven = ardis_images[indices_seven,:]
#     images_seven = torch.tensor(images_seven).type(torch.uint8)

#     labels_seven = torch.tensor([7 for y in ardis_labels])

#     ardis_dataset = EMNIST('./data', split="digits", train=True, download=True,
#                        transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))
#                        ]))

#     ardis_dataset.data = images_seven
#     ardis_dataset.targets = labels_seven

#     return ardis_dataset


# def get_southwest_dataset(attack_case='normal-case'):
#     if attack_case == 'normal-case':
#         with open('./saved_datasets/southwest_images_honest_full_normal.pkl', 'rb') as train_f:
#             saved_southwest_dataset_train = pickle.load(train_f)
#     elif attack_case == 'almost-edge-case':
#         with open('./saved_datasets/southwest_images_honest_almost_edge_case.pkl', 'rb') as train_f:
#             saved_southwest_dataset_train = pickle.load(train_f)
#     else:
#         saved_southwest_dataset_train = None
#     return saved_southwest_dataset_train


class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            #print("train member of the class: {}".format(self.train))
            #data = cifar_dataobj.train_data
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class TinyImageNet_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        # tinyimagenet_dataobj = TinyImageNet(self.root, self.train, self.transform, self.target_transform, self.download)
        tinyimagenet_dataobj = ImageFolder(self.root) 
        
        # if self.dataidxs is not None:
        #     images = [images[i] for i in self.dataidxs]

        # data = []
        # target = []

        # for img_path, target_class in images:
        #     with open(img_path, 'rb') as f:
        #         img = Image.open(f).convert('RGB')
        #         if self.transform is not None:
        #             img = self.transform(img)
        #         data.append(img)
        #         target.append(target_class)

        # return data, target
    
        if self.train:
            data = tinyimagenet_dataobj.data
            target = np.array(tinyimagenet_dataobj.targets)
        else:
            data = tinyimagenet_dataobj.data
            target = np.array(tinyimagenet_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
    
class ImageFolderTruncated(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, dataidxs=None, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(ImageFolderTruncated, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.dataidxs = dataidxs

        ### we need to fetch training labels out here:
        
        
        self.__build_truncated_dataset__()
    
        self._train_labels = np.array([tup[-1] for tup in self.imgs])

        
    def __build_truncated_dataset__(self):
        if self.dataidxs is not None:
            #self.imgs = self.imgs[self.dataidxs]
            # try:
            self.imgs = [self.imgs[idx] for idx in self.dataidxs]
            # except Exception as e:
            #     import IPython
            #     IPython.embed()
                
            #     print(e)
            #     exit(0)
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            
            sample = self.transform(sample)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    @property
    def get_train_labels(self):
        return self._train_labels
