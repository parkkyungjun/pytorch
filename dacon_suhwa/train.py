import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import os
from torch.utils.data import DataLoader
from dataset import init_dataset
import albumentations as A


def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


# createFolder('/Users/aaron/Desktop/test')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_chnnel = 3
num_classes = 11
learning_rate = 1e-3
batch_size = 8
num_epochs = 11
load_model = True
no1 = 825
train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
    ]
)
# Load Data
dataset = init_dataset(csv_file='train.csv', root_dir='train',
                       transform=train_transform)
testset = init_dataset(csv_file='flip.csv', root_dir='flip',
                       transform=transforms.ToTensor())
#train_set, test_set = torch.utils.data.random_split(dataset, [857, 1])
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

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
    model.load_state_dict(torch.load("googlenet_8/suhwa_155.pt"))


# Train Network
def inference(loader, model):
    model.eval()
    test = pd.DataFrame(pd.read_csv("test.csv"))
    label = []
    submission = test.copy()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)

            scores = model(x)
            _, prediction = scores.max(1)
            label += prediction.tolist()

    for i in range(len(label)):
        if label[i] == 10:
            label[i] = '10-1'
        elif label[i] == 0:
            label[i] = '10-2'

    submission['label'] = label
    submission['label'] = submission['label'].apply(lambda x: str(x))

    submission.to_csv('submission.csv', index=False)


inference(sub_loader, model)

# def check_accuracy(loader, model):
#     num_correct = 0
#     num_samples = 0
#     model.eval()
#
#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device=device)
#             y = y.to(device=device)
#
#             scores = model(x)
#             _, predictions = scores.max(1)
#             num_correct += (predictions == y).sum()
#             num_samples += predictions.size(0)
#
#         print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100}')
#
#     model.train()
#     return num_correct
#
#
# for epoch in range(num_epochs):
#     losses = []
#     #model.train()
#
#     for batch_idx, (data, target) in enumerate(train_loader):
#         # Get data to cuda if possible
#         data = data.to(device=device)
#         target = target.to(device=device)
#
#         # forward
#         scores = model(data)
#         loss = criterion(scores, target)
#
#         losses.append(loss.item())
#
#         # backward
#         optimizer.zero_grad()
#         loss.backward()
#
#         # gradient decent or adam step
#         optimizer.step()
#     print(f'Cost at epoch {epoch} is {sum(losses) / len(losses)}')
#     if epoch % 10 == 0:
#         check = check_accuracy(test_loader, model)
#         if check > no1:
#           no1 = check
#           print("save")
#           torch.save(model.state_dict(), f"googlenet_8/suhwa_{no1}.pt")
#
# torch.save(model.state_dict(), "googlenet_8/suhwa_last.pt")
# print("Checking accuracy on Training Set")
# check_accuracy(train_loader, model)
#
# print("Checking accuracy on Test_Set")
# check_accuracy(test_loader, model)
