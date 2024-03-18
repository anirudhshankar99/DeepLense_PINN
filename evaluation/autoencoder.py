import torchvision.models as models

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
        self.stride = stride
        
    def forward(self, x):
        identity = x.clone()
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
    
class inverse_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, identity_upsampling=None, stride=1):
        super(inverse_block, self).__init__()
        self.contraction = 2
        self.tconv1 = torch.nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1,padding=0)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.tconv2 = torch.nn.ConvTranspose2d(in_channels=out_channels,out_channels=out_channels,kernel_size=2,stride=stride)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.tconv3 = torch.nn.ConvTranspose2d(in_channels=out_channels,out_channels=out_channels//self.contraction,kernel_size=1,stride=1,padding=0)
        self.bn3 = torch.nn.BatchNorm2d(out_channels//self.contraction)
        self.relu = torch.nn.ReLU()
        self.identity_upsampling = identity_upsampling
        self.stride = stride

    def forward(self, x):
        identity = x.clone()
        x = self.tconv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.tconv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.tconv3(x)
        x = self.bn3(x)
        if self.identity_upsampling is not None:
            identity = self.identity_upsampling(identity)
        if x.shape != identity.shape:
            x = torch.nn.AdaptiveAvgPool2d((identity.shape[2],identity.shape[3]))(x)
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

class Encoder(torch.nn.Module):
    def __init__(self, block, in_channels, latent_dim):
        super(Encoder, self).__init__()
        self.model = Resnet(block,[2,2,2,2],in_channels,latent_dim)

    def forward(self, x):
        x = self.model(x)
        return x
    
class Decoder(torch.nn.Module):
    def __init__(self, inverse_block, layers, latent_dim, image_channels) -> None:
        super(Decoder,self).__init__()
        self.fc = torch.nn.Linear(latent_dim, 512*4)
        self.bn1 = torch.nn.BatchNorm1d(512*4)
        self.in_channels = 32
        self.layer1 = self._make_layer(inverse_block, layers[0], out_channels=32, stride=2)
        self.layer2 = self._make_layer(inverse_block, layers[1], out_channels=16, stride=2)
        self.layer3 = self._make_layer(inverse_block, layers[2], out_channels=8, stride=2)
        self.layer4 = self._make_layer(inverse_block, layers[3], out_channels=4, stride=2)
        self.layer5 = self._make_layer(inverse_block, 1, out_channels=2, stride=2)

        self.avgpoolf = torch.nn.AdaptiveAvgPool2d((150,150))

    def forward(self, x):
        x = self.fc(x)
        x = self.bn1(x)
        x = x.view(x.shape[0],32,8,8)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpoolf(x)

        return x

    
    def _make_layer(self, inverse_block, num_residual_blocks, out_channels, stride):
        identity_upsampling = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 2:
            identity_upsampling = torch.nn.Sequential(torch.nn.ConvTranspose2d(self.in_channels, out_channels//2, kernel_size=2, stride=stride),
                                                        torch.nn.BatchNorm2d(out_channels//2))
        
        layers.append(inverse_block(self.in_channels,out_channels, identity_upsampling, stride))
        self.in_channels = out_channels//2

        for i in range(num_residual_blocks-1):
            layers.append(inverse_block(self.in_channels,out_channels))
        
        return torch.nn.Sequential(*layers)

class AutoEncoder(torch.nn.Module):
    def __init__(self, in_channels, latent_dim) -> None:
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(block, in_channels=in_channels, latent_dim= latent_dim)
        self.decoder = Decoder(inverse_block, [3,3,3,3], latent_dim, 1)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

class Classifier(torch.nn.Module):
    def __init__(self, latent_dim, out_classes) -> None:
        super(Classifier, self).__init__()
        self.fc1 = torch.nn.Linear(latent_dim,latent_dim//2)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(latent_dim//2,latent_dim//4)
        self.fc3 = torch.nn.Linear(latent_dim//4,out_classes)
        self.softmax = torch.nn.Softmax(1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)

        return x

def encoder(in_channels, latent_dim):
    return Encoder(block, in_channels, latent_dim)

def classifier(latent_dim, out_classes):
    return Classifier(latent_dim, out_classes)