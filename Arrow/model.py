import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.conv(x)


class NET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NET, self).__init__()
        features = [32, 64, 128, 256]
        self.downs = nn.ModuleList()

        for feature in features:
            self.downs.append(CNN(in_channels, feature))
            in_channels = feature

        self.fc1 = nn.Linear(14*14*256, 1000)
        self.fc2 = nn.Linear(1000, 11)

    def forward(self, x):
        for down in self.downs:
            x = down(x)

        x = x.reshape(x.shape[0], -1)

        return self.fc2(self.fc1(x))


class CNNclassification(nn.Module):
    def __init__(self):
        super(CNNclassification, self).__init__()
        self.keep_prob = 0.5  ## dropout에서 쓰임
        self.layer1 = nn.Sequential(
            ##채널=1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  ## 절반으로 줄어듬

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(15 * 15 * 64, 363)
        # torch.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')

        self.layer4 = nn.Sequential(
            self.fc1,
            nn.ReLU(),  ##ReLU Sigmoid
            nn.Dropout(p=1 - self.keep_prob))

        self.fc2 = nn.Linear(363, 4)
        # nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.fc2(out)
        return out