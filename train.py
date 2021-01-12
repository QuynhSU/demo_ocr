import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from dataset import ListDataset
from dataset import char2token
from dataset import Batch
from model import make_model
from tqdm import tqdm
import numpy as np
from dataset import *
import os
# from predict import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    device = "cuda"
else:
    devide = "cpu"
print(device)

# src_mask=Variable(torch.from_numpy(np.ones([1, 1, 36], dtype=np.bool)).cuda())
SIZE=96

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)).cuda() / norm
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data * norm

def run_epoch(dataloader, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, (imgs, labels_y, labels,y) in tqdm(enumerate(dataloader), total = len(dataloader)):
        # print(imgs)
        batch = Batch(imgs.cuda(), labels_y.cuda(), labels.cuda())
        out = model(batch.imgs, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        # if i % 50 == 1:
        #     elapsed = time.time() - start
        #     print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
        #             (i, loss / batch.ntokens, tokens / elapsed))
        #     start = time.time()
        #     tokens = 0

    return total_loss / total_tokens

def eval_model(dataloader, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    acc_sq = 0
    acc_char = []
    for i, (imgs, labels_y, labels, ground_truth) in tqdm(enumerate(dataloader), total = len(dataloader)):
        # print(imgs)
        batch = Batch(imgs.cuda(), labels_y.cuda(), labels.cuda())
        out = model(batch.imgs, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        # img = torch.from_numpy(img).float().unsqueeze(0).cuda()
        memory = model.encode(imgs.cuda().float(), batch.src_mask)
        # print(memory[1,:,:])
        src_mask=Variable(torch.from_numpy(np.ones([1, 1, 377], dtype=np.bool)).cuda())
        re = []
        for i, img in enumerate(memory):
            # memory = model.encode(imgs[i], batch.src_mask)
            ys = torch.ones(1, 1).fill_(1).long().cuda()
            for i in range(36 - 1):
                out = model.decode(img, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1)).long().cuda()))
                prob = model.generator(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.data[0]
                ys = torch.cat([ys,torch.ones(1, 1).long().cuda().fill_(next_word)], dim=1)
                if token2char[next_word.item()] == '>':
                    break
            ret = ys.cpu().numpy()[0]
            out = [token2char[i] for i in ret]
            out = "".join(out[1:-1])
            # print(type(out))
            re.append(out)
            # pred = greedy_decode(img)
            # if out == y[i]:
        re = np.array(re)
        ground_truth = np.array(ground_truth)
        acc_sq+=np.sum(re==ground_truth)

        # accuracy = []
        for index, label in enumerate(ground_truth):
            # print("oaky")
            prediction = re[index]
            total_count = len(label)
            correct_count = 0
            # print("label: ", label)
            # print("prediction: ", prediction)
            for i, tmp in enumerate(prediction):
                if tmp == prediction[i]:
                    correct_count += 1

            acc_char.append(correct_count / total_count)

        

        #Calculate acc char
        

        # print(acc)
        # print(re)
        # print(y)
    acc_char = np.mean(np.array(acc_char).astype(np.float32), axis=0)
    acc_sq = acc_sq/len(dataloader.dataset)
        # if i % 50 == 1:
        #     elapsed = time.time() - start
        #     print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
        #             (i, loss / batch.ntokens, tokens / elapsed))
        #     start = time.time()
        #     tokens = 0
    return total_loss / total_tokens, acc_sq, acc_char




def train():
    batch_size = 50
    train_dataloader = torch.utils.data.DataLoader(ListDataset('IC15/train/gt.txt', training=True), batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(ListDataset('IC15/test/gt.txt', training=False), batch_size=batch_size, shuffle=False, num_workers=0)
    model = make_model(len(char2token))
    # model.load_state_dict(torch.load('checkpoint.pth'))
    model.cuda()
    criterion = LabelSmoothing(size=len(char2token), padding_idx=0, smoothing=0.1)
    criterion.cuda()
    model_opt = NoamOpt(model.tgt_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-4))
    best_acc=0
    for epoch in range(10000):
        print("epoch: ", epoch)
        model.train()
        train_loss = run_epoch(train_dataloader, model,
              SimpleLossCompute(model.generator, criterion, model_opt))

        model.eval()
        test_loss, acc_sq, acc_char = eval_model(val_dataloader, model,
              SimpleLossCompute(model.generator, criterion, None))
        
        print("train_loss:", train_loss)
        print("test_loss:", test_loss.item)
        print("acc_char:", acc_char)
        print("acc_seq:", acc_sq)
        if acc_sq > best_acc:
            best_acc = acc_sq
            torch.save(model.state_dict(), '%08d_%f.pth'%(epoch, test_loss))

if __name__=='__main__':
    train()





