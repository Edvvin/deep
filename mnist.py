import pandas as pd

datatrain = pd.read_csv('/home/edvvin/Desktop/mnist/mnist_train.csv')

datatrain_array = datatrain.values

ytrain = datatrain_array[:,0]
temp = datatrain_array[:,1:]
xtrain = []
for ar in temp:
    xic = []
    for i in range(28):
        xic.append(ar[i*28:(i+1)*28])
    xtrain.append([xic])


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


torch.manual_seed(1234)



hl = 100
lr = 0.001
num_epoch = 100
inl = 28*28
outl = 10

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8,kernel_size=5, stride=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(8, 8,kernel_size=3, stride=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc2 = nn.Linear(5*5*8,hl)
        self.fc3 = nn.Linear(hl, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
net = Net()
net.to(device)
crit = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

for epoch in range(num_epoch):
    X = Variable(torch.Tensor(xtrain).float()).to(device)
    Y = Variable(torch.Tensor(ytrain).long()).to(device)
    
    optimizer.zero_grad()
    out = net(X)
    loss = crit(out, Y)
    loss.backward()
    optimizer.step()
    
    if (epoch) % 10 == 0:
        print ('Epoch [%d/%d] Loss: %.4f'%(epoch+1, num_epoch, loss.data))
    
datatest = pd.read_csv('/home/edvvin/Desktop/mnist/mnist_test.csv')

datatest_array = datatest.values   

temp = datatest_array[:,1:]
xtest = []

for ar in temp:
    xic = []
    for i in range(28):
        xic.append(ar[i*28:(i+1)*28])
    xtest.append([xic])


ytest = datatest_array[:,0]

X = Variable(torch.Tensor(xtest).float().to(device))
Y = torch.Tensor(ytest).long().to(device)
out = net(X)
_, predicted = torch.max(out.data, 1)

print('Accuracy of the network %.4f %%' % (100 * torch.sum(Y==predicted) / 10000.0 ))
