import torch.nn as nn
import torch.nn.functional as F

class Net_FC(nn.Module):
    def __init__(self):
        super(Net_FC, self).__init__()
        self.conv1 = nn.Conv2d(1,8,3,1,1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8,16,5,1,2)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(16,32,3,1,1)
        self.bn32= nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32,10,1)

        self.gap = nn.AvgPool2d(7,7)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(self.bn2(x)))
        x = self.pool(x)
        x = self.relu(self.conv3(self.bn4(x)))
        x = self.pool(x)
        x = self.sig(self.conv4(self.bn32(x)))
        self.CAM = x.detach().numpy()
        x = self.gap(x) 
        x = x.view(-1,10)
        return x

    def return_CAM(self):
        return self.CAM

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)#28 28 -> 26 26
        self.conv2 = nn.Conv2d(8, 16, 3)#-> 24 24
        self.pool = nn.MaxPool2d(2, 2) #-> 12 12
        self.fc1 = nn.Linear(12*12*16, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 12*12*16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
