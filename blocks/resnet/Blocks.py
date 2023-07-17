import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,shortcut=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut_flag = shortcut
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
          
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut_flag == True:
            out += self.shortcut(x)
    
        out = F.relu(out)

        return out


class Upconvblock(nn.Module):
    expansion = 16
    def __init__(self,in_planes,output_channels, stride=1):
        super(Upconvblock,self).__init__()
        if stride == 1:
            self.scaleup = nn.Conv2d(in_planes,output_channels,kernel_size=3,padding=1)
        elif stride == 2:
            self.scaleup = nn.ConvTranspose2d(in_planes,output_channels,kernel_size=2,stride=stride)
        # self.scaleup = nn.ConvTranspose2d(in_planes,output_channels,kernel_size=2,stride=2)
        # self.scaleup = nn.Upsample(scale_factor=stride,mode="bilinear",align_corners=True)
        self.conv1 = nn.Conv2d(output_channels,output_channels,stride = 1,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels,output_channels,stride=1,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)
          
    def forward(self, x):
        x = self.scaleup(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1,shortcut=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.shortcut_flag = shortcut
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
          
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out =  F.relu(self.bn2(self.conv2(out)))
    
        out = self.bn3(self.conv3(out))
        if self.shortcut_flag == True:
            out += self.shortcut(x)
    
        out = F.relu(out)
        return out