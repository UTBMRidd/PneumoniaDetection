import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
import torchvision

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1_1 = nn.Conv2d(3,64,3,padding=1)
        self.conv1_2 = nn.Conv2d(64,64,3,padding=1)
        self.pool1_1 = nn.MaxPool2d(2, 2)
        self.do1_1 = nn.Dropout(p=0.2)

        self.conv2_1 = nn.Conv2d(64,32,3,padding=1)
        self.conv2_2 = nn.Conv2d(32,32,3,padding=1)
        self.pool2_1 = nn.MaxPool2d(2, 2)
        self.do2_1 = nn.Dropout(p=0.2)
        
        self.conv3_1 = nn.Conv2d(32,16,3,padding=1)
        self.conv3_2 = nn.Conv2d(16,16,3,padding=1)
        self.pool3_1 = nn.MaxPool2d(2, 2)
        self.do3_1 = nn.Dropout(p=0.2)

        self.fc5_1 = Flatten()
        self.fc5_2 = nn.Linear(5184, 128)
        self.fc5_3 = nn.Dropout(p=0.2)
        self.fc5_4 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = F.relu(self.conv1_1(x))
        x = self.pool1_1(F.relu(self.conv1_2(x)))
        x = self.do1_1(x)

        x = F.relu(self.conv2_1(x))
        x = self.pool2_1(F.relu(self.conv2_2(x)))
        x = self.do2_1(x)

        x = F.relu(self.conv3_1(x))
        x = self.pool3_1(F.relu(self.conv3_2(x)))
        x = self.do3_1(x)

        x = self.fc5_1(x)
        x = F.relu(self.fc5_2(x))
        x = self.fc5_3(x)
        x = F.softmax(self.fc5_4(x))
        
        return x

net = torch.load('./weights/model.pt')
net.cuda()
test_X = torch.from_numpy(np.load('./data_preprocess/array/test_X.npy'))
test_Y = torch.from_numpy(np.load('./data_preprocess/array/test_Y.npy'))

test = utils.TensorDataset(test_X,test_Y)
test_data = utils.DataLoader(test)

result_predict = np.zeros((test_X.shape[0],1))
net.train(False)
for i, data in enumerate(test_data, 0):
    inputs, labels = data
    inputs = inputs.float()
    labels = labels.long()
    inputs = inputs.cuda()
    labels = labels.cuda()
    outputs = net(inputs)
    _,predicted = torch.max(outputs,1)
    result_predict[i] = predicted.cpu()

np.save('./results/test-set',result_predict)