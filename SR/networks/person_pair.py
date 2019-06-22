import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from resnet_v1 import ResNet, Bottleneck

class person_pair(nn.Module):
    def __init__(self, num_classes = 3):
        super(person_pair, self).__init__()
        self.resnet101_union = ResNet(Bottleneck, [3, 4, 23, 3])
        self.resnet101_a = ResNet(Bottleneck, [3, 4, 23, 3])
        self.resnet101_b = self.resnet101_a

        # for param in self.resnet101_union.parameters():
            # param.requires_grad = False
        # for param in self.resnet101_a.parameters():
            # param.requires_grad = False
        # for param in self.resnet101_b.parameters():
            # param.requires_grad = False
            
        self.bboxes = nn.Linear(10, 256)
        self.fc6 = nn.Linear(2048+2048+2048+256, 4096)
        self.fc7 = nn.Linear(4096, num_classes)
        self.ReLU = nn.ReLU(False)
        self.Dropout = nn.Dropout()

        self._initialize_weights()

    # x1 = union, x2 = object1, x3 = object2, x4 = bbox geometric info
    def forward(self, x1, x2, x3, x4): 
        x1 = self.resnet101_union(x1)
        x2 = self.resnet101_a(x2)
        x3 = self.resnet101_b(x3)
        x4 = self.bboxes(x4)

        x = torch.cat((x4, x1, x2, x3), 1)
        x = self.Dropout(x)
        fc6 = self.fc6(x)
        x = self.ReLU(fc6)
        x = self.Dropout(x)
        x = self.fc7(x)

        # return x
        return x,x1,x2,x3

    def _initialize_weights(self):
        self.fc6.apply(weights_init_classifier)
        self.fc7.apply(weights_init_classifier)
        ckpt = torch.load('resnet101-5d3b4d8f.pth')
        model_dict = self.resnet101_union.state_dict()
        model_pre_dict = {k: v for k, v in ckpt.items() if k in model_dict}
        for key ,val in ckpt.items() :
            if key not in model_dict:
                print ('skip :',key)
        model_dict.update(model_pre_dict)
        self.resnet101_union.load_state_dict(model_dict)
        self.resnet101_a.load_state_dict(model_dict)
        self.resnet101_b.load_state_dict(model_dict)      


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0.0)              