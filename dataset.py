''' Datasets
    This file contains definitions for our CIFAR, ImageFolder, and HDF5 datasets
'''
import numpy as np
# from synthesis_plate import gen_syn_plate

import os
import os.path
import sys
from PIL import Image
import numpy as np
from tqdm import tqdm, trange

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
import cv2
from torch.autograd import Variable
import torch
from preprocessing import pre_process


max_seq_len = 10
vocab = "<!\"()*,-.0126789:?ABCDEGHKLMNOPQRSTVXabcdeghiklmnopqrstuvxyÁÍÔÝàáâãèéêìíòóôõùúýăĐđĩũƠơưạảẤấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ“…>"
# start symbol <
# end symbol >
char2token = {"PAD": 0}
token2char = {0: "PAD"}
for i, c in enumerate(vocab):
    char2token[c] = i+1
    token2char[i+1] = c


def load_image(path):
    im = cv2.imread(path)
    #im_bw = cv2.resize(im, (256,64))
    return im


def read_label(anno_path):
    f = open(anno_path, "rt")
    label = f.read().strip().replace('\n', '')
    f.close()
    return label


def load_by_folder_path(imgs_path, annos_path, type):
    n_image = 0
    X = []
    y = []
    # count = 0
    for r, d, f in os.walk(imgs_path):
        # print(f)
        for file in f:
            if type in file:
                #print(str(n_image) + ". Load image " + file)
                image_path = os.path.join(r, file)
                # print(image_path)
                anno_path = os.path.join(
                    annos_path, file.replace(type, '.txt'))
                label = read_label(anno_path)
                label = label.replace('@', '')
                # count +=1
                # print(count)    
                if '$' in label:
                    label = ''
                label = label.upper()

                X.append(image_path)
                y.append(label)
                n_image += 1
        
    return X, y


class Dataset(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
        root/dogball/xxx.png
        root/dogball/xxy.png
        root/dogball/xxz.png
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
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, data_path, dataset_name, training):
        
        annos_path = os.path.join(data_path, '{}/annos'.format(dataset_name))
        imgs_path = os.path.join(data_path, '{}/imgs'.format(dataset_name))
#         print("okay")
        self.imgs, self.labels = load_by_folder_path(imgs_path, annos_path, '.png')
#         print("check . . . . ")
        #x_val, y_val = data_load_path(val_path, '.jpg')
        # print("check . . . . ")
        print("Number images: ", len(self.imgs))
        self.training = training
        self.transform = transforms.Compose([transforms.ToTensor()]) 
    def __getitem__(self, index):
        
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, label = self.imgs[index], self.labels[index]
        img = cv2.imread(path)
        img = pre_process(img)
        original_img = np.copy(img)
        
        # if self.training:
        #     img, label = gen_syn_plate(img, label)

        img = img/255.0
        img = np.moveaxis(img, 2, 0)

        img = torch.from_numpy(img).float()
        label_token = np.zeros(max_seq_len, dtype=int)
        #print(path, label, len(label))
        for i, c in enumerate('<'+label):
            try:
                label_token[i] = char2token[c]
            except:
#                 print(path)
                break
        label_token = torch.from_numpy(label_token)
        label_token_y = np.zeros(max_seq_len, dtype=int)
        for i, c in enumerate(label + '>'):
            try:
                label_token_y[i] = char2token[c]
            except:
#                 print(path)
                break
        label_token_y = torch.from_numpy(label_token_y)
        # original_img = self.transform(self.imgs[index])
        # label_token = self.transform(self.labels[index])
        return img,  label_token_y, label_token, label

        
        

    def __len__(self):
        return len(self.imgs)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
        subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, imgs, trg_y, trg, pad=0):
        self.imgs = Variable(imgs, requires_grad=False)
        self.src_mask = Variable(torch.from_numpy(
            np.ones([imgs.size(0), 1, 377], dtype=np.bool)).cuda())
        if trg is not None:
            self.trg = Variable(trg.cuda(), requires_grad=False)
            self.trg_y = Variable(trg_y.cuda(), requires_grad=False)
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return Variable(tgt_mask.cuda(), requires_grad=False)
