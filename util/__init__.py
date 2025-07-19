import torch
from torch.utils.data import Dataset
import os

class PTFileDataset(Dataset):
    def __init__(self, opt,mode='train'):
        # 加载.pt文件中的数据和标签
        root='./data'
        if opt.airpls==True:
            if opt.SG==True:
                file='airpls+SG'
            else:
                file='airpls'
        else:
            if opt.SG==True:
                file='SG'
            else:
                file='original'

        self.data, self.labels = torch.load(os.path.join(root,file,mode+'.pt'))

    def __len__(self):
        # 返回数据集的大小
        return len(self.data)

    def __getitem__(self, idx):
        # 根据索引返回单个样本和标签
        return self.data[idx], self.labels[idx]
