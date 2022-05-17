import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from dataset import init_dataset

def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
in_chnnel = 3
num_classes = 11
learning_rate = 1e-3
batch_size = 16
num_epochs = 100
load_model = True
no1 = 150

# Load Data
dataset = init_dataset(csv_file='train.csv', root_dir='train',
                transform=transforms.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [700, 158])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

sub_dataset = init_dataset(csv_file='test.csv', root_dir='test',
                           transform=transforms.ToTensor())
sub_loader = DataLoader(dataset=sub_dataset, batch_size=batch_size, shuffle=False)
# Model
model = torchvision.models.googlenet(pretrained=True)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    model.load_state_dict(torch.load("16/suhwa_154.pt"))

# Train Network
def inference(loader, model):
    model.eval()
    sub = pd.DataFrame(pd.read_csv("test.csv"))
    sub['label'] = 0
    i = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)

            scores = model(x)
            _, prediction = scores.max(1)
            for j in prediction.tolist():
                #print(j)
                sub['label'][i] = str(j)
                i += 1

    sub.to_csv("test.csv", index=False)

#inference(sub_loader, model)

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

    for batch_idx, (data, target) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        target = target.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, target)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        #gradient decent or adam step
        optimizer.step()
    print(f'Cost at epoch {epoch} is {sum(losses)/len(losses)}')
    check = check_accuracy(test_loader, model)
    if check > no1:
        no1 = check
        print("save")
        torch.save(model.state_dict(), f"16/suhwa_{no1}.pt")

torch.save(model.state_dict(), "16/suhwa_last.pt")
print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test_Set")
check_accuracy(test_loader, model)