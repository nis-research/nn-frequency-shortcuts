import torch
import torch.nn as nn
import torch.nn.functional as F

  

class ResNet(nn.Module):
    def __init__(self, block_en, num_blocks,num_class):
        super(ResNet, self).__init__()
      
        self.in_planes = 64
        self.num_class = num_class
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block_en, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block_en, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block_en, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block_en, 512, num_blocks[3], stride=2)
        
        self.features = nn.Sequential(self.conv1, self.bn1,self.relu,self.layer1,self.layer2,self.layer3,self.layer4)
        

        # self.sm = nn.Softmax(dim=1)
        self.classifier = nn.Linear(512*block_en.expansion,self.num_class)
             
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,shortcut=True))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)

    
    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        enc = self.features(x)
        # print(enc.size())
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        enc = F.avg_pool2d(enc, enc.size(2))
        enc = enc.view(enc.size(0), -1) # flatten
        
        prediction = self.classifier(enc)
        # prediction = self.sm(prediction)
        return enc, prediction
