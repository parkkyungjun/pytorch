import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from ddddd import init_dataset
import torch.nn.functional as nnf
from ddddd import __differ__
from mmm import NET
import timm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Hyperparameters
in_chnnels = 3
num_classes = 11
learning_rate = 1e-5
batch_size = 1
num_epochs = 10
load_model = True
model_name = 'google'
model_name2 = 'regnet'

train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(distortion_scale=.15, p=.15, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.RandomRotation(degrees=20, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Normalize(mean=[0.5878, 0.5398, 0.4853],
                             std=[0.1505, 0.1592, 0.1703]),
    ]
)

sub_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5915, 0.5468, 0.4935],
                             std=[0.1549, 0.1642, 0.1756]),
    ]
)

# Load Data
dataset = init_dataset(csv_file='D:/asddas/user_data/train.csv', root_dir='D:/dacon_suhwa/train_299',
                       transform=train_transform)
# train_set, test_set = torch.utils.data.random_split(dataset, [858, 0])
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

sub_dataset = init_dataset(csv_file='r_test.csv', root_dir='D:/dacon_suhwa/test_299',
                           transform=sub_transform)
sub_loader = DataLoader(dataset=sub_dataset, batch_size=batch_size, shuffle=False)

# Model
model = torchvision.models.googlenet().to(device)
model2 = torchvision.models.regnet_y_1_6gf().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True)

if load_model:
    model.load_state_dict(torch.load(f"pt/206_{model_name}.pt"))
    model2.load_state_dict(torch.load(f"pt/206_{model_name2}.pt"))


# Train Network
def inference(loader, model):
    model.eval()
    test = pd.DataFrame(pd.read_csv("test.csv"))
    label = []
    submission = test.copy()
    i = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            scores = model(x)

            _, prediction = scores.max(1)

            label += prediction.tolist()
            i += 1
    for i in range(len(label)):
        if label[i] == 10:
            label[i] = '10-1'
        elif label[i] == 0:
            label[i] = '10-2'

    submission['label'] = label
    submission['label'] = submission['label'].apply(lambda x: str(x))

    submission.to_csv('submission.csv', index=False)

def resize_tensor(x, b, c):
    x = nnf.interpolate(x, size=(b, b), mode='bicubic', align_corners=False)
    t = transforms.CenterCrop((c, c))
    x = t(x)
    return x

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    model2.eval()
    index = 0
    with torch.no_grad():
        for x, y in loader:
            top2 = []
            X = x.clone().detach()
            xx = x.clone().detach()
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)

            for i in scores:
                top2.append(sorted(i[:11], reverse=True)[1])

            _, predictions = scores.max(1)

            if _ - top2[0] < 2 :

                scores2 = model(resize_tensor(X, 350, 299).to(device=device))
                _2, predictions2 = scores2.max(1)

                for i in scores2:
                    top2.append(sorted(i[:11], reverse=True)[1])

                predictions = predictions2

                if predictions != 2 and predictions != 3 and predictions != 7:
                    scores = model2(x)
                    _, predictions = scores.max(1)

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

            index += 1
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100}')

    model.train()
    model2.train()
    return num_correct


# for epoch in range(num_epochs):
#     losses = []
#
#     for batch_idx, (data, targets) in enumerate(train_loader):
#         # Get data to cuda if possible
#         data = data.to(device=device)
#         targets = targets.to(device=device)
#
#         # forward
#         # scores = model(data)
#         aux1, aux2, scores = model(data)
#         loss1 = criterion(scores, targets)
#         loss2 = criterion(aux1, targets)
#         loss3 = criterion(aux2, targets)
#         loss = loss1 + 0.3*(loss2)
#
#         losses.append(loss.item())
#
#         # backward
#         optimizer.zero_grad()
#         loss.backward()
#
#         # gradient decent or adam step
#         optimizer.step()
#     mean_loss = sum(losses)/len(losses)
#     scheduler.step(mean_loss)
#     print(f'Cost at epoch {epoch} is {sum(losses) / len(losses)}')
#     if load_model:
#         a = check_accuracy(sub_loader, model)
#         if a >= 205:
#             torch.save(model.state_dict(), f'pt/{a}_{model_name}.pt')
#     else:
#         if epoch > 25:
#             a = check_accuracy(sub_loader, model)
#             if a >= 205:
#                 torch.save(model.state_dict(), f'pt/{a}_{model_name}.pt')
#
# torch.save(model.state_dict(), f"pt/{model_name}.pt")

# train()
# check_accuracy(train_loader, model)
check_accuracy(sub_loader, model)
# inference(sub_loader, model)
# __differ__()