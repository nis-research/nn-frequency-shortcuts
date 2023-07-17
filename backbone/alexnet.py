import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=11):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1,bias=False),
            
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(96, 256, kernel_size=3, padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),

            nn.Conv2d(384, 384, kernel_size=3, padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),

            nn.Conv2d(384, 256, kernel_size=3, padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096,bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes,bias=False)
        )

    def forward(self, x):
        
        x = self.features(x.float())
        enc = x.view(x.size(0), 256 * 2 * 2)
        prediction = self.classifier(enc)
        out = enc
        return out,enc,prediction