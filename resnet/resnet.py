import torch

class block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsampling=None,stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = torch.nn.Conv2d(in_channels,out_channels, kernel_size=1,stride=1,padding=0)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels,out_channels, kernel_size=3,stride=stride,padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.conv3 = torch.nn.Conv2d(out_channels,out_channels*self.expansion, kernel_size=1,stride=1,padding=0)
        self.bn3 = torch.nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = torch.nn.ReLU()
        self.identity_downsampling = identity_downsampling
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsampling is not None:
            identity = self.identity_downsampling(identity)

        x += identity
        x = self.relu(x)
        return x
    
class Resnet(torch.nn.Module): 
    def __init__(self, block, layers, image_channels, num_classes):
        # layers gives number of reuses of block [3,4,6,3]
        # image_channels here is 1 (b/w)
        # num_classes here is 3 (no, sphere, vort)
        super(Resnet, self).__init__()

        # initial layers
        self.in_channels = 64 
        self.conv1 = torch.nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3) 
        self.bn1 = torch.nn.BatchNorm2d(64) 
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        # Resnet layers come now
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.fc = torch.nn.Linear(512*4, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsampling = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsampling = torch.nn.Sequential(torch.nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride),
                                                        torch.nn.BatchNorm2d(out_channels*4))
        
        layers.append(block(self.in_channels,out_channels, identity_downsampling, stride))
        self.in_channels = out_channels*4

        for i in range(num_residual_blocks-1):
            layers.append(block(self.in_channels,out_channels))
        
        return torch.nn.Sequential(*layers)

def ResNet50(img_channels=1, num_classes=3):
    return Resnet(block, [3,4,6,3], img_channels, num_classes)