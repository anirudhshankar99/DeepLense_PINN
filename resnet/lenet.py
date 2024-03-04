import numpy as np
from PIL import Image
import torch

class LeNet(torch.nn.Module):
    def __init__(self, numChannels, classes):
		# call the parent constructor
        super(LeNet, self).__init__()
		# initialize first set of CONV => RELU => POOL layers
        self.conv1 = torch.nn.Conv2d(in_channels=numChannels, out_channels=20,
			kernel_size=(7, 7))
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(4, 4), stride=(3, 3))
		# initialize second set of CONV => RELU => POOL layers
        self.conv2 = torch.nn.Conv2d(in_channels=20, out_channels=50,
			kernel_size=(5, 5))
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(4, 4), stride=(3, 3))
		# initialize first (and only) set of FC => RELU layers
        self.fc1 = torch.nn.Linear(in_features=9800, out_features=1000)
        # self.fc1 = torch.nn.Linear(in_features=57800, out_features=500)
        # self.fc1 = torch.nn.Linear(in_features=57800, out_features=12500)
        self.relu3 = torch.nn.ReLU()

        # self.fc15 = torch.nn.Linear(in_features=12500, out_features=500)
        # self.relu4 = torch.nn.ReLU()
		# initialize our softmax classifier
        self.fc2 = torch.nn.Linear(in_features=1000, out_features=classes)
        self.logSoftmax = torch.nn.LogSoftmax(dim=1)
			
    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        # x = self.fc15(x)
        # x = self.relu4(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output