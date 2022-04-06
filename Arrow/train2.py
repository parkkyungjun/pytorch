import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from main import runse

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
    def __init__(self, in_channels=1, num_classes=4):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16*30*30, num_classes)

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
num_classes = 4
learning_rate = 0.001
batch_size = 32
num_epochs = 1000

# Load Data
dataset = runse(csv_file='C:/Download/runes/runes.csv', root_dir='C:/Download/runes',
                transform = transforms.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [133, 30])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Initialize nework
model = CNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, target) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        target = target.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, target)

        # backward
        optimizer.zero_grad()
        loss.backward()

        #gradient decent or adam step
        optimizer.step()

# Check accuracy on training & test to see how good our model

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

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100}')

    model.train()

print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test_Set")
check_accuracy(test_loader, model)