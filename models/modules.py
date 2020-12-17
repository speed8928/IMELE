from collections import OrderedDict
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import model_zoo
import copy
import numpy as np
from models import senet
from models import resnet
from models import densenet

import os, sys
import torchvision.models as models
import torch.autograd.variable as Variable
import scipy.io as sio


class _UpProjection(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_UpProjection, self).__init__()

        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)

        self.conv2 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode='bilinear',align_corners=False)
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))

        out = self.relu(bran1 + bran2)

        return out

class E_resnet(nn.Module):

    def __init__(self, original_model, num_features = 2048):
        super(E_resnet, self).__init__()        
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
       

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        return x_block1, x_block2, x_block3, x_block4

class E_densenet(nn.Module):

    def __init__(self, original_model, num_features = 2208):
        super(E_densenet, self).__init__()        
        self.features = original_model.features

    def forward(self, x):
        x01 = self.features[0](x)
        x02 = self.features[1](x01)
        x03 = self.features[2](x02)
        x04 = self.features[3](x03)

        x_block1 = self.features[4](x04)
        x_block1 = self.features[5][0](x_block1)
        x_block1 = self.features[5][1](x_block1)
        x_block1 = self.features[5][2](x_block1)
        x_tran1 = self.features[5][3](x_block1)

        x_block2 = self.features[6](x_tran1)
        x_block2 = self.features[7][0](x_block2)
        x_block2 = self.features[7][1](x_block2)
        x_block2 = self.features[7][2](x_block2)
        x_tran2 = self.features[7][3](x_block2)

        x_block3 = self.features[8](x_tran2)
        x_block3 = self.features[9][0](x_block3)
        x_block3 = self.features[9][1](x_block3)
        x_block3 = self.features[9][2](x_block3)
        x_tran3 = self.features[9][3](x_block3)

        x_block4 = self.features[10](x_tran3)
        x_block4 = F.relu(self.features[11](x_block4))

        return x_block1, x_block2, x_block3, x_block4

class E_senet(nn.Module):

    def __init__(self, original_model, num_features = 2048):
        super(E_senet, self).__init__()

        self.base = nn.Sequential(*list(original_model.children())[:-3])

        #self.conv = nn.Conv2d(3, 64 , kernel_size=5, stride=1, bias=False)
        #self.bn = nn.BatchNorm2d(64)
      
        self.pool = nn.MaxPool2d(3, stride=2,ceil_mode=True)
        self.down = _UpProjection(64,128)

    def forward(self, x):
        #conv_x = F.relu(self.conv(x))
        #conv_x = self.bn(conv_x) 

        #summary(self.base, input_size=(3, 440, 440))
        x_block0 = self.base[0][0:6](x)
        x = self.base[0][6:](x_block0)
        

        # x = self.Harm(x)
        # x = self.pool(x)
        # x = self.down(x,(110,110))

        x_block1 = self.base[1](x)
        x_block2 = self.base[2](x_block1)
        x_block3 = self.base[3](x_block2)
        x_block4 = self.base[4](x_block3)
        return x_block0,  x_block1, x_block2, x_block3, x_block4



class D2(nn.Module):

    def __init__(self, num_features = 2048):
        super(D2, self).__init__()

        self.conv = nn.Conv2d(num_features, num_features //
                                   2, kernel_size=1, stride=1, bias=False)
        num_features = num_features // 2
        self.bn = nn.BatchNorm2d(num_features)
        self.up1 = _UpProjection(1024, 512) #out 512 channels
       
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.up2 = _UpProjection(512, 256)

        self.conv2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.up3 = _UpProjection(256,128)

        self.conv3 = nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.up4 = _UpProjection(128,64)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1 ,bias=False)
       


    def forward(self,x_block0, x_block1, x_block2, x_block3, x_block4):

        x_d0 = F.relu(self.bn(self.conv(x_block4)))


        x_d1 = self.up1(x_d0, [x_block3.size(2), x_block3.size(3)])
        x_block3 = F.relu(self.bn1(self.conv1(x_block3)))#512
        cx_d1 = torch.cat((x_d1,x_block3),1)#512
        cx_d1 = F.relu(self.bn1(self.conv1(cx_d1)))#512
        

        x_d2 = self.up2(cx_d1, [x_block2.size(2), x_block2.size(3)])
        x_block2 = F.relu(self.bn2(self.conv2(x_block2)))
        cx_d2 = torch.cat((x_d2,x_block2),1)
        cx_d2 = F.relu(self.bn2(self.conv2(cx_d1)))



        x_d3 = self.up3(cx_d2, [x_block1.size(2), x_block1.size(3)])
        
        x_block1 = F.relu(self.bn3(self.conv3(x_block1)))
        cx_d3 = torch.cat((x_d3,x_block1),1)
        cx_d3 = F.relu(self.bn3(self.conv3(cx_d3)))

        

        x_d4 = self.up4(cx_d3, [x_block1.size(2)*2, x_block1.size(3)*2])



        cx_d4 = torch.cat((x_d4,x_block0),1)
        cx_d4 = F.relu(self.bn3(self.conv4(cx_d4)))


        return cx_d4 #128 chanel

# class D(nn.Module):

#     def __init__(self, num_features = 2048, skip_con = False):
#         super(D, self).__init__()
#         self.skip_con = skip_con
#         if skip_con == False: 
#             self.conv = nn.Conv2d(num_features, num_features //
#                                    2, kernel_size=1, stride=1, bias=False)
#             num_features = num_features // 2
#         else:
#             self.conv = nn.Conv2d(num_features, num_features, kernel_size=1, stride=1, bias=False)
#             num_features = num_features // 2
      

#         self.bn = nn.BatchNorm2d(num_features)


       
#         self.up1 = _UpProjection(
#             num_input_features=num_features, num_output_features=num_features // 2)
#         num_features = num_features // 2

#         self.up2 = _UpProjection(
#             num_input_features=num_features, num_output_features=num_features // 2)
#         num_features = num_features // 2

#         self.up3 = _UpProjection(
#             num_input_features=num_features, num_output_features=num_features // 2)
#         num_features = num_features // 2

#         self.up4 = _UpProjection(
#             num_input_features=num_features, num_output_features=num_features // 2)
#         num_features = num_features // 2

#         self.conv1 = nn.Conv2d(, , kernel_size=1, stride=1, bias=False)
#         self.conv2 = nn.Conv2d(, , kernel_size=1, stride=1, bias=False)
#         self.conv3 = nn.Conv2d(, , kernel_size=1, stride=1, bias=False)


#     def forward(self, x_block1, x_block2, x_block3, x_block4):

#         if self.skip_con == False:
#             x_d0 = F.relu(self.bn(self.conv(x_block4)))
#             x_d1 = self.up1(x_d0, [x_block3.size(2), x_block3.size(3)])
#             x_d2 = self.up2(x_d1, [x_block2.size(2), x_block2.size(3)])
#             x_d3 = self.up3(x_d2, [x_block1.size(2), x_block1.size(3)])
#             x_d4 = self.up4(x_d3, [x_block1.size(2)*2, x_block1.size(3)*2])
#         else:
#             print('flag')
#             x_d0 = F.relu(self.bn(self.conv(x_block4)))
#             x_d1 = self.up1(x_d0, [x_block3.size(2), x_block3.size(3)])
#             cx_d1 = torch.cat((x_d1,x_block3),1)

#             x_d2 = self.up2(cx_d1, [x_block2.size(2), x_block2.size(3)])
#             cx_d2 = torch.cat((x_d2,x_block2),1)
#             x_d3 = self.up3(cx_d2, [x_block1.size(2), x_block1.size(3)])
#             cx_d3 = torch.cat((x_d3,x_block1),1)
#             x_d4 = self.up4(cx_d3, [x_block1.size(2)*2, x_block1.size(3)*2])



        

#         return x_d4







class MFF(nn.Module):

    def __init__(self, block_channel, num_features=64):

        super(MFF, self).__init__()

        self.up0 = _UpProjection(
            num_input_features=64, num_output_features=16)

        self.up1 = _UpProjection(
            num_input_features=block_channel[0], num_output_features=16)
        
        self.up2 = _UpProjection(
            num_input_features=block_channel[1], num_output_features=16)
       
        self.up3 = _UpProjection(
            num_input_features=block_channel[2], num_output_features=16)
       
        self.up4 = _UpProjection(
            num_input_features=block_channel[3], num_output_features=16)

        self.conv = nn.Conv2d(
            80, 80, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(80)
        

    def forward(self,x_block0, x_block1, x_block2, x_block3, x_block4, size):

        x_m0 = self.up0(x_block0, size)
        x_m1 = self.up1(x_block1, size)
        x_m2 = self.up2(x_block2, size)
        x_m3 = self.up3(x_block3, size)
        x_m4 = self.up4(x_block4, size)

        x = self.bn(self.conv(torch.cat((x_m0, x_m1, x_m2, x_m3, x_m4), 1)))
        x = F.relu(x)

        return x


class R(nn.Module):
    def __init__(self, block_channel):

        super(R, self).__init__()
        

    
        self.conv0 = nn.Conv2d(208, 144,
                               kernel_size=1, stride=1)
    
        self.bn0 = nn.BatchNorm2d(144)

        self.conv1 = nn.Conv2d(144, 144,
                               kernel_size=5, stride=1, padding=2, bias=True)
        self.bn1 = nn.BatchNorm2d(144)

        self.conv2 = nn.Conv2d(144, 144, kernel_size=5, stride=1, padding=2, bias=True)

        self.bn2 = nn.BatchNorm2d(144)

        self.conv3 = nn.Conv2d(144, 72, kernel_size=3, padding=1, stride=1)

        self.bn3 = nn.BatchNorm2d(72)

        self.conv4 = nn.Conv2d(72, 1, kernel_size=1, stride=1)



    def forward(self, x):


        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)


        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)


        x4 = self.conv4(x3)


        return x4
