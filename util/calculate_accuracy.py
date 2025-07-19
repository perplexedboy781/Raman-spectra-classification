import torch
from sklearn.metrics import confusion_matrix
import numpy as np
def cal_acc(model,loader,device):
    for data, label in loader:
        data, label = data.to(device), label.to(device)

        logits = model(data)
        pred = torch.gt(logits, 0.5).int()

        crocret= torch.eq(pred, label).float().sum().item()

        total_num= data.size(0) * label.size(-1)
        acc=crocret/total_num
        return acc

def cal_cm(model,loader,device):
    for data, label in loader:
        data= data.to(device)

        logits = model(data)
        pred = torch.gt(logits, 0.5).int().to('cpu').detach().numpy()
        label=label.detach().numpy()
        cm=[]
        for i in range(label.shape[1]):
            cm.append(confusion_matrix(label[:,i],pred[:,i]))
        return np.array(cm)