## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

from torchvision import models


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        ### Inputsize (batch_size, 224, 224)
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # Only using the pooling layer for Reduction
        # ((224 - 5 + 2 * 2) / 1) + 1 = 224 
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2) # After pooling: 32x112x112
        
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2, bias=False) # After pooling: 64x56x56
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3  = nn.Conv2d(64, 128, 5, padding=2, bias=False) # after pooling: 128x28x28
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4  = nn.Conv2d(128, 256, 3, padding=1, bias=False) # after pooling: 256x14x14
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5  = nn.Conv2d(256, 512, 3, padding=1, bias=False) # after pooling: 512x7x7
        self.bn5 = nn.BatchNorm2d(512)
        
        # Pooling Layer
        self.pool = nn.MaxPool2d(2, 2) # Kernel 2x2 with a stride of 2
        
        # Classifier
        self.fc1 = nn.Linear(512*7*7, 10000)
        self.drop1 = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(10000, 2000)
        self.drop2 = nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(2000, 136)
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # Passing tensor to convolutional layers
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(self.bn2(F.leaky_relu(self.conv2(x))))
        x = self.pool(self.bn3(F.leaky_relu(self.conv3(x))))
        x = self.pool(self.bn4(F.leaky_relu(self.conv4(x))))
        x = self.pool(self.bn5(F.leaky_relu(self.conv5(x))))
        
        # flattening for fc layers
        x = x.flatten(start_dim=1)
        
        # Passing tensor to classifier
        x = F.leaky_relu(self.fc1(self.drop1(x)))
        x = F.leaky_relu(self.fc2(self.drop2(x)))
        
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x


class BigNet(nn.Module):
    
    def __init__(self, freeze_params=False) -> None:
        super(BigNet, self).__init__()
        
        # Preparing Input for Resnet - could've changed the first layer aswell
        self.pre_conv = nn.Conv2d(1, 3, 7, padding=3, bias=False)
        self.pre_bn = nn.BatchNorm2d(3)
        
        self.resnet_model = models.resnet50(pretrained=True)
        fc_in_features = self.resnet_model.fc.in_features
        
        if freeze_params:
            for param in self.resnet_model.parameters():
                param.requires_grad = False
            
        self.resnet_model.fc = nn.Sequential(
            nn.Linear(fc_in_features, 1024),                             
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 136))

    
    def forward(self, x):
        x = self.pre_bn(self.pre_conv(x))
        x = self.resnet_model(x)
        
        return x