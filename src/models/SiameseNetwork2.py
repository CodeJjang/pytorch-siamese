import torch.nn as nn


class SiameseNetwork2(nn.Module):
    def __init__(self):
        super(SiameseNetwork2, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(2, stride=2))

        self.fc1 = nn.Sequential(
            nn.Linear(64 * 4 * 4, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 128),
            nn.Linear(128, 2))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2=None):
        if input2 is None:
            return self.forward_once(input1)
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
