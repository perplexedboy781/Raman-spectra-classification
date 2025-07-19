import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #1X623
        self.sq1=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=6,kernel_size=5, stride=2,padding=1),
            #4X623
            nn.ReLU()

        )
        self.sq2=nn.Sequential(
            nn.Conv1d(in_channels=6,out_channels=16,kernel_size=5, stride=2,padding=1),
            nn.ReLU()
            #16X311
        )
        self.sq3=nn.Sequential(
            nn.Linear(2480,5)

        )
    def forward(self,X):
        X=self.sq1(X)
        X=self.sq2(X)
        #print(X.size)
        X = X.view(X.size(0), -1)
        X=self.sq3(X)
        output = torch.sigmoid(X)
        return output
if __name__=='__main__':
    a=torch.ones(5,1,623).double()
    print(a)
    model=CNN().double()
    x=model(a)

    print(x.size())
