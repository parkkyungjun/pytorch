import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

    def forward(self, x):
        return self.conv(x)


class NET(nn.Module):
    def __init__(self, in_channels=3, out_channels=11, features=[32, 64, 128, 256]):
        super(NET, self).__init__()

        self.downs = nn.ModuleList()

        for feature in features:
            self.downs.append(CNN(in_channels, feature))
            in_channels = feature

        self.fc1 = nn.Linear(14*14*256, 1000)
        self.fc2 = nn.Linear(1000, out_channels)

        self.fc = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
        )

    def forward(self, x):
        for down in self.downs:
            x = down(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        #x = self.fc1(x)

        return x