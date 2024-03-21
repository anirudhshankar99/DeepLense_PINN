import torch

class Encoder(torch.nn.Module):
    def __init__(self, fc_in = 25088, latent_dim = 512) -> None:
        """
        Encoder module to represent the image in the latent space

        :param fc_in: in_channels for the fc layer that maps to the latent space
        :param latent_dim: latent space dimension
        """
        super(Encoder, self).__init__()

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
        self.fc1 = torch.nn.Linear(in_features=fc_in, out_features=latent_dim)

        self.convx1 = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=1)
        self.convx2 = torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1)
        self.dr = torch.nn.Dropout(0.1)

    def forward(self, x):
        """
        Forward propagation

        :param x: Image to be encoded
        :return: Encoded latent space
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

        return x
    
class Decoder(torch.nn.Module):
    def __init__(self, latent_dim=512) -> None:
        """
        Decoder module to recreate the image from the latent space

        :param latent_dim: latent space dimension 
        """
        super(Decoder, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=latent_dim, out_features=2048)
        self.tconv1 = torch.nn.ConvTranspose2d(in_channels=8,out_channels=8,stride=3,kernel_size=3)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(in_channels=8,out_channels=8,kernel_size=3,padding=1)
        self.tconv2 = torch.nn.ConvTranspose2d(in_channels=8,out_channels=4,kernel_size=4,stride=3)
        self.bn2 = torch.nn.BatchNorm2d(4)
        self.conv2 = torch.nn.Conv2d(in_channels=4,out_channels=4,kernel_size=3,padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=4,out_channels=1,kernel_size=4,padding=4)

    def forward(self, x):
        """
        Forward propagation
        
        :param x: Latent space for decoding
        :return: Decoded image
        """
        x = self.fc1(x)
        x = x.view(x.shape[0],8,16,16)

        x = self.tconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        res = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x + res

        x = self.tconv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        res = x.clone()
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = x + res
        x = self.conv3(x)

        return x
    
class AutoEncoder(torch.nn.Module):
    def __init__(self, dropout=None) -> None:
        """
        Autoencoder module

        :param dropout: Dropout fraction imposed on the latent space
        """
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        if dropout:
            self.dr = torch.nn.Dropout(dropout)
        else: self.dr = None 
        self.decoder = Decoder()
    
    def forward(self, x):
        """
        Forward propagation

        :param x: Image to be encoded into the latent space
        :return: Image decoded from the latent space
        """
        x = self.encoder(x)
        if self.dr: x = self.dr(x)
        x = self.decoder(x)

        return x