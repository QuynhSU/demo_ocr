import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable
import os
import cv2
import numpy as np
from torchvision import datasets, models, transforms
from tqdm import tqdm
from preprocessing import pre_process
label_len = 30
vocab =  "<,.+:-?$ 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ>"
# start symbol <
# end symbol >
char2token = {"PAD":0}
token2char = {0:"PAD"}
for i, c in enumerate(vocab):
    char2token[c] = i+1
    token2char[i+1] = c


def illegal(label):
    if len(label) > label_len-1:
        return True
    for l in label:
        if l not in vocab[1:-1]:
            # print('check',l)
            return True
    return False


class ListDataset(Dataset):
    def __init__(self, fname, training=False):
        self.lines = []
        # if not isinstance(fname, list):
        #     fname = [fname]
        # for f in fname:
        lines = open(fname).readlines()
        # for i in lines :
        #     print(i.strip('\n').split(", ")[0])
            # if not illegal(i.strip('\n').split(", ")[1].replace(" ", "").replace("\"", "")):
            #     print(i)
        self.lines += [i for i in lines if not illegal(i.strip('\n').split(", ")[1].replace(" ", "").replace("\"", "")) ]
        self.training = training
        # print(self.lines)
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, index):
        '''
        line: image path\tlabel
        '''
        line = self.lines[index]
        # print(line)
        img_path, label_y_str = line.strip('\n').split(', ')
        label_y_str= label_y_str.replace(" ", "").replace("\"", "")
        if self.training:
            img_path = os.path.join("IC15/train", img_path.replace("ï»¿",'').replace("\ufeff", ""))
        else:
            img_path = os.path.join("IC15/test", img_path.replace("ï»¿",'').replace("\ufeff", ""))
        img = cv2.imread(img_path)
        # print(img_path)
        # print(type(img))
        # if img is None:
        #     return None
        img = pre_process(img)
        img = img / 255.
        # Channels-first
        # img= np.resize(img, (16, 48, 3))
        
        # print(img.shape)
        img = np.transpose(img, (2, 0, 1))
        # As pytorch tensor
        img = torch.from_numpy(img).float()
        label = np.zeros(label_len, dtype=int)
        for i, c in enumerate('<'+label_y_str):
            label[i] = char2token[c]
        label = torch.from_numpy(label)

        label_y = np.zeros(label_len, dtype=int)
        for i, c in enumerate(label_y_str+'>'):
            label_y[i] = char2token[c]
        label_y = torch.from_numpy(label_y) 
        
        return img, label_y, label, label_y_str

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
        self.imgs = Variable(imgs.cuda().float(), requires_grad=False)
        self.src_mask = Variable(torch.from_numpy(np.ones([imgs.size(0), 1, 377], dtype=np.bool)).cuda().float())
        if trg is not None:
            self.trg = Variable(trg.cuda().long(), requires_grad=False)
            self.trg_y = Variable(trg_y.cuda().long(), requires_grad=False)
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

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, name):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.name = name
    def forward(self, x):
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name is self.name:
                b = x.size(0)
                c = x.size(1)
                return x.view(b, c, -1).permute(0, 2, 1)
        return None

# if __name__=='__main__':
#     listdataset = ListDataset('your-lines')
#     dataloader = torch.utils.data.DataLoader(listdataset, batch_size=2, shuffle=False, num_workers=0)
#     for epoch in range(1):
#         for batch_i, (imgs, labels_y, labels) in enumerate(dataloader):
#             continue


















