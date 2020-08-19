import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2))

        self.fc1 = nn.Sequential(
            nn.Linear(64 * 4 * 4, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 128),
            nn.Linear(128, 2))

    def _forward_siamese_head(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2=None, input3=None):
        if input2 is None and input3 is None:
            return self._forward_siamese_head(input1)
        output1 = self._forward_siamese_head(input1)
        output2 = self._forward_siamese_head(input2)
        output3 = self._forward_siamese_head(input3)
        return output1, output2, output3
