import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from main import runse

def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
in_chnnel = 3
num_classes = 4
learning_rate = 1e-3
batch_size = 32
num_epochs = 100
no1 = 29
load_model = True

# Load Data
dataset = runse(csv_file='C:/Download/runes/runes.csv', root_dir='C:/Download/runes/all',
                transform = transforms.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [133, 30])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Model
model = torchvision.models.googlenet(pretrained=True)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#if load_model:
    #load_checkpoint(torch.load("C:/Download/runes/no1_checkpoint.pth.tar"))
model.load_state_dict(torch.load("C:/Download/runes/no1_checkpoint2.pt"))
#model.eval()
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

# for epoch in range(num_epochs):
#     losses = []
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
#         #gradient decent or adam step
#         optimizer.step()
#     print(f'Cost at epoch {epoch} is {sum(losses)/len(losses)}')
#     check = check_accuracy(test_loader, model)
#     if check > no1:
#         no1 = check
#         #checkpoint = {'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict()}
#         print("save")
#         torch.save(model.state_dict(), "C:/Download/runes/no1_checkpoint2.pt")

print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test_Set")
check_accuracy(test_loader, model)