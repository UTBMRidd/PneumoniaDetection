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
net = Net()
net.cuda()

criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
batch_size = 16
nb_epochs = 10

train_X = torch.from_numpy(np.load('./data_preprocess/array/train_X.npy'))
train_Y = torch.from_numpy(np.load('./data_preprocess/array/train_Y.npy'))

val_X = torch.from_numpy(np.load('./data_preprocess/array/val_X.npy'))
val_Y = torch.from_numpy(np.load('./data_preprocess/array/val_Y.npy'))

my_dataset = utils.TensorDataset(train_X,train_Y)
my_dataloader = utils.DataLoader(my_dataset,batch_size=16,shuffle=True) 

val = utils.TensorDataset(val_X,val_Y)
val_data = utils.DataLoader(val)

for epoch in range(nb_epochs):
    running_loss = 0.0
    val_counter = 0
    for i, data in enumerate(my_dataloader, 0):
        inputs, labels = data
        inputs = inputs.float()
        labels = labels.long()
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    net.train(False)
    for i, data in enumerate(val_data, 0):
        inputs, labels = data
        inputs = inputs.float()
        labels = labels.long()
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = net(inputs)
        if outputs[0,labels[0]] > 0.5:
            val_counter = val_counter + 1
    print('Validation accuracy : ')
    print((val_counter / 16)*100)
    print('Loss Value : ')
    print((running_loss / train_X.shape[0]))
    net.train(True)
    
print('Finished Training')
torch.save(net, './weights/model.pt')