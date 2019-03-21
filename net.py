import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.fc6 = nn.Linear(256 * 6 * 6 * 2, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 4)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc6(x))
        x = self.drop(x)
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x)) * 10
        return x

class GOTURN_AlexNET(nn.Module):

    def __init__(self):
        super(GOTURN_AlexNET, self).__init__()
        self.prev_conv1 = nn.Conv2d(3, 96, 11, stride=4, padding=2) #in 244 out 55
        self.prev_pool1 = nn.MaxPool2d(3, 2)  # in: 55 out: 27
        self.prev_norm1 = nn.BatchNorm2d(96)
        self.prev_conv2 = nn.Conv2d(96, 256, 5, padding=2) #in: 27 out: 27
        self.prev_pool2 = nn.MaxPool2d(3, 2)  # in: 27 out: 13
        self.prev_norm2 = nn.BatchNorm2d(256)
        self.prev_conv3 = nn.Conv2d(256, 384, 3, padding=1)  # in: 13 out: 13
        self.prev_conv4 = nn.Conv2d(384, 384, 3, padding=1)  # in: 13 out: 13
        self.prev_conv5 = nn.Conv2d(384, 256, 3, padding=1)  # in: 13 out: 13
        self.prev_pool5 = nn.MaxPool2d(3, 2)  # in: 13 out: 6

        self.curr_conv1 = nn.Conv2d(3, 96, 11, stride=4, padding=2) #in 244 out 55
        self.curr_pool1 = nn.MaxPool2d(3, 2)  # in: 55 out: 27
        self.curr_norm1 = nn.BatchNorm2d(96)
        self.curr_conv2 = nn.Conv2d(96, 256, 5, padding=2) #in: 27 out: 27
        self.curr_pool2 = nn.MaxPool2d(3, 2)  # in: 27 out: 13
        self.curr_norm2 = nn.BatchNorm2d(256)
        self.curr_conv3 = nn.Conv2d(256, 384, 3, padding=1)  # in: 13 out: 13
        self.curr_conv4 = nn.Conv2d(384, 384, 3, padding=1)  # in: 13 out: 13
        self.curr_conv5 = nn.Conv2d(384, 256, 3, padding=1)  # in: 13 out: 13
        self.curr_pool5 = nn.MaxPool2d(3, 2)  # in: 13 out: 6


        self.classifier = Classifier()


    def forward(self, x_prev, x_curr):

        # prev_img
        x_prev = F.relu(self.prev_conv1(x_prev.float()))
        x_prev = self.prev_pool1(x_prev)
        x_prev = self.prev_norm1(x_prev)
        x_prev = F.relu(self.prev_conv2(x_prev))
        x_prev = self.prev_pool2(x_prev)
        x_prev = self.prev_norm2(x_prev)
        x_prev = F.relu(self.prev_conv3(x_prev))
        x_prev = F.relu(self.prev_conv4(x_prev))
        x_prev = F.relu(self.prev_conv5(x_prev))
        x_prev = self.prev_pool5(x_prev)
        x_prev = x_prev.view(-1, 256 * 6 * 6)

        #curr_img
        x_curr = F.relu(self.prev_conv1(x_curr.float()))
        x_curr = self.prev_pool1(x_curr)
        x_curr = self.prev_norm1(x_curr)
        x_curr = F.relu(self.prev_conv2(x_curr))
        x_curr = self.prev_pool2(x_curr)
        x_curr = self.prev_norm2(x_curr)
        x_curr = F.relu(self.prev_conv3(x_curr))
        x_curr = F.relu(self.prev_conv4(x_curr))
        x_curr = F.relu(self.prev_conv5(x_curr))
        x_curr = self.prev_pool5(x_curr)
        x_curr = x_curr.view(-1, 256 * 6 * 6)

        x = torch.cat((x_prev, x_curr), 1)

        return self.classifier(x)

    def num_outputs(self):
        return 4
