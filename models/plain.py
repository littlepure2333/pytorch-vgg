"""Plain 18/34 in Pytorch."""
import torch
import torch.nn as nn

cfg = {
    'plain18': [64, 64, 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512],
    'plain34': [64, 64, 64, 64, 64, 64, 'M', 128, 128, 128, 128, 128, 128, 128, 'M', 256, 256, 256, 256, 256, 256, 256,
                256, 256, 256, 256, 'M', 512, 512, 512, 512, 512]
}


class Plain(nn.Module):
    def __init__(self, plain_name, num_classes):
        super(Plain, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features = self._make_layers(cfg[plain_name])
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, stride=2, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = Plain('plain18', 10)
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(y.size())


if __name__ == '__main__':
    test()
