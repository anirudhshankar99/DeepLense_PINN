import torch
import torchvision.models as models
from torchvision.transforms import v2
import torch
from einops.layers.torch import Rearrange
from einops import repeat

class block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsampling=None,stride=1):
        """
        ResidualBlock module

        :param in_channels: number of incoming channels in the image (here 1, as b/w image)
        :param out_channels: number of outgoing channels (here 3, equal to the number of classes)
        :param identity_downsampling: whether identity downsampling needs to be applied, for the residual sum to be consistent in channel count
        :param stride: stride to be used, default 1 as stride is used in some layers to decrease image shape, but not always
        """
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
        """
        Forward propagation

        :param x: Tensor to be propagated through the ResidualBlock module
        :return: Tensor after passing through the ResidualBlock module
        """
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

class Resnet(torch.nn.Module): 
    def __init__(self, block, layers, image_channels, num_classes):
        """
        Implementation of the ResNet model whose specificity is determined by the layers param

        :class block: implements the Residual block module
        :param layers: [4] array containing number of repeats in each of the 4 residual layers
        :param image_channels: number of incoming channels in the image (here 1, as b/w image)
        :param num_classes: number of outgoing channels (here 3, equal to the number of classes)
        """
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
        """
        Forward propagation

        :param x: Image to be classified
        :return: Prediction of the ResNet model
        """
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
        """
        Makes a layer with the required number of residual blocks

        :class block: Implements the ResidualBlock module
        :param num_residual_blocks: Number of residual blocks in the layer
        :param out_channels: Number of channels the processed image needs to be in when leaving the layer
        :param stride: Stride to be used in some convolution layers
        :return: Sequential module containing the layer
        """
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

def ResNet18(image_channels=1, num_classes=3):
    """
    Returns an object of the ResNet18 model

    :param image_channels: Number of channels in the image (here 1, as the image is b/w)
    :param num_classes: Number of classes to be classified into
    :return: ResNet18 object
    """
    return Resnet(block, [2,2,2,2], image_channels, num_classes)

class Resnet_simple(torch.nn.Module):
    def __init__(self, fc_in = 25088) -> None:
        """
        Resnet_simple model class

        :param fc_in: Optional number of in_channels of the fc layer that creates the latent space
        """
        super(Resnet_simple, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.mp1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv3 = torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.mp2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv4 = torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3)
        self.bn4 = torch.nn.BatchNorm2d(32)
        self.mp3 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv5 = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.bn5 = torch.nn.BatchNorm2d(64)
        self.conv6 = torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1)
        self.bn6 = torch.nn.BatchNorm2d(64)
        self.conv7 = torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1)
        self.bn7 = torch.nn.BatchNorm2d(64)
        self.conv8 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.bn8 = torch.nn.BatchNorm2d(128)
        self.fc1 = torch.nn.Linear(in_features=fc_in, out_features=256)
        self.dr = torch.nn.Dropout(p=0.3)
        self.fc2 = torch.nn.Linear(in_features=256, out_features=32)
        self.fc3 = torch.nn.Linear(in_features=32, out_features=3)
        self.convx1 = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=1)
        self.convx2 = torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1)

    def forward(self, x):
        """
        Forward propagation

        :param x: Image to be classified
        :return: Class predicted by the model
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.mp1(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.mp2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.mp3(x)

        res = self.convx1(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = x + res
        x = self.relu(x)

        res = self.convx2(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)
        x = x + res

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)

        x = x.view(x.shape[0],-1)

        x = self.fc1(x)
        x = self.dr(x)
        x = self.fc2(x)
        x = self.fc3(x)
        

        return torch.nn.Softmax(1)(x)
    
class LensAutoEncoder(torch.nn.Module):
    def __init__(self, in_shape, device) -> None:
        """
        The Physics-Informer module that transforms the image as given by the Single Isothermal Sphere equation, method (1)

        :param in_shape: Shape of the image (assuming its a square)
        :param device: Device to place certain tensors in (cuda or cpu)
        """
        super(LensAutoEncoder, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.in_shape = in_shape
        self.device = device
        self.k_min = 0.8
        self.k_max = 1.2
        self.pos_x = torch.tensor([[_ for _ in range(-in_shape//2,(in_shape)//2)] for __ in range(in_shape)]).to(device)
        self.pos_y = torch.transpose(self.pos_x,0,1)
        
        self.pos_x = self.pos_x.view(-1)
        self.pos_y = self.pos_y.flatten()
        
        self.origin_mask = (self.pos_x != 0) | (self.pos_y != 0)
        self.pos_x_nz = self.pos_x[self.origin_mask]
        self.pos_y_nz = self.pos_y[self.origin_mask]
        self.r_nz = torch.sqrt(self.pos_x_nz**2+self.pos_y_nz**2)
        self.eps = 1e-9
        
        
    
    def forward(self, x, x_true):
        """
        Forward propagation

        :param x: The k(x,y) tensor inferred from a different module that will be used to transform the image
        :param x_true: Image to be transformed by k(x,y) through the Single Isothermal Sphere equation, method (1)
        :return: Transformed image
        """
        BATCH_SIZE = x.shape[0]
        x = x.view(BATCH_SIZE,-1)
        k = self.sigmoid(x)
        x = x_true
        
        k = self.k_min + (self.k_max - self.k_min)*k
        k_nz = k[:,self.origin_mask].to(self.device)

        lensed_x_nz = (self.pos_x_nz[None, None, None, :] - k_nz.view(BATCH_SIZE, 1, 1, -1) * self.pos_x_nz[None, None, None, :] / self.r_nz)
        lensed_y_nz = (self.pos_y_nz[None, None, None, :] - k_nz.view(BATCH_SIZE, 1, 1, -1) * self.pos_y_nz[None, None, None, :] / self.r_nz)

        lensed_indices = torch.round(lensed_x_nz + (self.in_shape // 2)).long() * self.in_shape + torch.round(lensed_y_nz + (self.in_shape // 2)).long() # i*row_length + j
        lensed_indices = lensed_indices.to(self.device)
        
        out_image = torch.zeros(BATCH_SIZE,self.in_shape, self.in_shape).view(BATCH_SIZE,-1).to(self.device)

        zeroes_origin_masked = out_image.gather(1, lensed_indices.view(BATCH_SIZE,-1))

        x = x.view(BATCH_SIZE, -1)
        in_true_values = x[:, self.origin_mask]

        updates = torch.where(zeroes_origin_masked == 0, in_true_values, (zeroes_origin_masked + in_true_values)/2)
        out_image.scatter_(1, lensed_indices.view(BATCH_SIZE, -1), updates)

        max_values, _ = out_image.max(dim=0, keepdim=True)
        max_values, _ = max_values.max(dim=1, keepdim=True)

        out_image = out_image/(max_values + self.eps)
        out_image = out_image.view(BATCH_SIZE,1,self.in_shape,self.in_shape)
        return out_image
    
class LensAutoEncoder2(torch.nn.Module):
    def __init__(self, in_shape, device) -> None:
        """
        The Physics-Informer module that transforms the image as given by the Single Isothermal Sphere equation, method (2)

        :param in_shape: Shape of the image (assuming its a square)
        :param device: Device to place certain tensors in (cuda or cpu)
        """
        super(LensAutoEncoder2, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.in_shape = in_shape
        self.device = device
        self.k_min = 0.8
        self.k_max = 1.2
        self.pos_x = torch.tensor([[_ for _ in range(-in_shape//2,(in_shape)//2)] for __ in range(in_shape)]).to(device)
        self.pos_y = torch.transpose(self.pos_x,0,1)
        
        self.pos_x = self.pos_x.view(-1)
        self.pos_y = self.pos_y.flatten()
        
        self.origin_mask = (self.pos_x != 0) | (self.pos_y != 0)
        self.pos_x_nz = self.pos_x[self.origin_mask]
        self.pos_y_nz = self.pos_y[self.origin_mask]
        self.r_nz = torch.sqrt(self.pos_x_nz**2+self.pos_y_nz**2)
        self.eps = 1e-9
        
    
    def forward(self, x, x_true):
        """
        Forward propagation

        :param x: The k(x,y) tensor inferred from a different module that will be used to transform the image
        :param x_true: Image to be transformed by k(x,y) through the Single Isothermal Sphere equation, method (1)
        :return: Transformed image
        """
        BATCH_SIZE = x.shape[0]
        x = x.view(BATCH_SIZE,-1)
        k = self.sigmoid(x)
        x = x_true
        
        k = self.k_min + (self.k_max - self.k_min)*k
        k_nz = k[:,self.origin_mask].to(self.device)

        lensed_x_nz = (self.pos_x_nz[None, None, None, :] - k_nz.view(BATCH_SIZE, 1, 1, -1) * self.pos_x_nz[None, None, None, :] / self.r_nz)
        lensed_y_nz = (self.pos_y_nz[None, None, None, :] - k_nz.view(BATCH_SIZE, 1, 1, -1) * self.pos_y_nz[None, None, None, :] / self.r_nz)

        lensed_indices = torch.round(lensed_x_nz + (self.in_shape // 2)).long() * self.in_shape + torch.round(lensed_y_nz + (self.in_shape // 2)).long() # i*row_length + j
        lensed_indices = lensed_indices.to(self.device)
        
        out_image = torch.zeros(BATCH_SIZE,self.in_shape, self.in_shape).view(BATCH_SIZE,-1).to(self.device)

        x = x.view(BATCH_SIZE, -1)
        in_true_values = x[:, self.origin_mask]
        out_image.scatter_add_(1, lensed_indices.view(BATCH_SIZE, -1), in_true_values)

        max_values, _ = out_image.max(dim=0, keepdim=True)
        max_values, _ = max_values.max(dim=1, keepdim=True)

        out_image = out_image/(max_values + self.eps)
        out_image = out_image.view(BATCH_SIZE,1,self.in_shape,self.in_shape)
        return out_image
