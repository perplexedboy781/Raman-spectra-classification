import torch
import os
from util import  PTFileDataset
from torch.utils.data import DataLoader
from util.model import CNN
from util.config import parser
from util.create_folder import create_folder_with_date
import pandas as pd
from util.Visualization import plot_loss,plot_acc,plot_multilabel_cf
from util.calculate_accuracy import cal_acc,cal_cm
from util.Save_model_hyperparameters import redirect_stdout_to_file
#模型训练的默认参数
opt = parser.parse_args()
#选择训练的设备
if torch.cuda.is_available():
    print('模型在GPU上训练')
    device='cuda:0'
else:
    print('模型在CPU上训练')
    device = 'cpu'

#加载数据
train_db=PTFileDataset(opt,mode='train')
train_loader=DataLoader(train_db,batch_size=opt.batch_size)
val_db=PTFileDataset(opt,mode='val')
val_loader=DataLoader(val_db,batch_size=500)
test_db=PTFileDataset(opt,mode='test')
test_loader=DataLoader(test_db,batch_size=500)
#选择模型
model=CNN().double().to(device)
#选择优化器
optimizer=torch.optim.Adam(model.parameters(),lr=opt.lr,weight_decay=opt.l2)
#选择损失函数
loss_fn=torch.nn.BCELoss().to(device)
#记录训练日志的路径
log_path=create_folder_with_date('./log')

#early stop
best_epoch,best_acc=0,0
#训练日志记录
log=[]
for epoch in range(opt.n_epochs):
    for data,label in train_loader:
        data,label=data.to(device),label.to(device)
        pred=model(data)
        loss=loss_fn(pred,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    val_acc=cal_acc(model,val_loader,device)
    if val_acc > best_acc:
        best_epoch=epoch
        best_acc=val_acc
        torch.save(model.state_dict(),os.path.join(log_path,'best.mdl'))
    log.append([epoch,val_acc,loss.item()])
    print(f'第{epoch}批次，验证集准确率:{val_acc},损失值{loss.item()}')
print('训练结束！')
#记录模型参数和训练超参数
with redirect_stdout_to_file(os.path.join(log_path,'model&hyperparameters.txt')):
    print(model)
    print(opt)
#记录训练日志
df=pd.DataFrame(log,columns=['epoch','val_acc','loss_value'])
df.to_csv(os.path.join(log_path,'train_log.csv'),index=False)
#训练过程可视化
plot_loss(df,log_path)
plot_acc(df,best_epoch,log_path)
test_acc=cal_acc(model,test_loader,device)
plot_multilabel_cf(cal_cm(model,test_loader,device),test_acc,log_path)