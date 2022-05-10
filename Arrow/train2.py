import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from main import init_dataset

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=7):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16*150*150, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
in_chnnel = 1
num_classes = 7
learning_rate = 0.001
batch_size = 32
num_epochs = 10
save_point = 1800
# Load Data
dataset = init_dataset(csv_file='D:/Training/WINTEC/label_7.csv', root_dir='D:/Training/WINTEC/Train_7',
                transform = transforms.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [24854, 6000])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
test_dataset = init_dataset(csv_file='D:/Training/WINTEC/label_test_7.csv', root_dir='D:/Training/WINTEC/insp_test1',
                            transform = transforms.ToTensor())
r_train_set, r_test_set = torch.utils.data.random_split(test_dataset,[3303,1000])
r_test_loader = DataLoader(dataset=r_train_set, batch_size=batch_size, shuffle=True)
# Initialize nework
model = CNN().to(device)
model.load_state_dict(torch.load('D:/Training/WINTEC/pt/test_7.pt'))
model.eval()
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# checkpoint = torch.load('D:/Training/WINTEC/pt/test_7.pt')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
# # model.eval()
# model.train()

def check_accuracy(loader, model):
    # if loader.dataset.train:
    #     print("Checking accuracy on training data")
    # else:
    #     print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            #print("test")
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100}')

    model.train()
    return num_correct

for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, target) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        #if data == None:
            #continue
        target = target.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, target)

        # backward
        optimizer.zero_grad()
        loss.backward()

        #gradient decent or adam step
        optimizer.step()
        #print("traing....")
    check_every_epoch = check_accuracy(r_test_loader, model)
    if check_every_epoch > save_point:
        torch.save(model.state_dict(), 'D:/Training/WINTEC/pt/test_7.pt')
        save_point = check_every_epoch
        print("save model...")


# Check accuracy on training & test to see how good our model

torch.save(model.state_dict(), 'D:/Training/WINTEC/pt/final_7.pt')

print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test_Set")
check_accuracy(test_loader, model)