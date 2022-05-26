import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
<<<<<<< HEAD
from main import runse
from model import NET
from model import CNNclassification
=======
#from main import runse
>>>>>>> cdf96ea6a41f66fbfdeb1057def4173cc5f085fc

def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_chnnel = 3
num_classes = 11
learning_rate = 1e-2
batch_size = 8
num_epochs = 10
#no1 = 29
load_model = False

# Load Data
dataset = runse(csv_file='C:/momentum/train.csv', root_dir='C:/momentum/train',
                transform=transforms.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [858, 0])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Model
# model = NET(in_chnnel, num_classes).to(device)
model = torchvision.models.googlenet(pretrained=True).to(device)
# model = CNNclassification().to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    model = torch.load("C:/momentum/pt/save.pt")

# Train Network


def check_accuracy(loader, model):
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

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100}')

    model.train()
    return num_correct


for epoch in range(num_epochs):
    losses = []
    pathes = []
    label = []
    for batch_idx, (data, target, path) in enumerate(train_loader):
        # Get data to cuda if possible
        pathes.extend(np.array(path))
        label.extend(np.array(target))

        data = data.to(device=device)
        target = target.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, target)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient decent or adam step
        optimizer.step()

    a = pd.DataFrame({'file_name': pathes, 'label': label})
    a = a.sort_values(by=['file_name'], axis=0)
    a.to_csv(f'C:/momentum/save{epoch}.csv', index=False)
    print(f'Cost at epoch {epoch} is {sum(losses)/len(losses)}')
    # check = check_accuracy(test_loader, model)
    # if check > no1:
    #     no1 = check
    #     print("save")
    #     torch.save(model, "C:/Download/runes/no1_checkpoint3.pt")

torch.save(model, "C:/momentum/pt/save.pt")
print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test_Set")
#check_accuracy(test_loader, model)