import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import glob 
from models import modules, net, resnet, densenet, senet
import loaddata
import util
import numpy as np
import sobel
import argparse
import cv2
from PIL import Image
from tensorboard_logger import configure, log_value
import pandas as pd
import os
import csv
import re
def main():
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)

    parser = argparse.ArgumentParser()
   
  
    parser.add_argument("--model")
    parser.add_argument("--csv")
    parser.add_argument("--outfile")
    args = parser.parse_args()

    md = glob.glob(args.model+'/*.tar')

    md.sort(key=natural_keys)
  

    for x in md:
        x = str(x)

        model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
        #model = torch.nn.DataParallel(model,device_ids=[0,1]).cuda()
        state_dict = torch.load(x)['state_dict']
        model.load_state_dict(state_dict, strict=False)
        test_loader = loaddata.getTestingData(1, args.csv)
        test(test_loader, model, args)



def test(test_loader, model, args):
    
    losses = AverageMeter()
    model.eval()
    model.cuda()
    totalNumber = 0
    errorSum = {'MSE': 0, 'RMSE': 0, 'MAE': 0,'SSIM':0}

    for i, sample_batched in enumerate(test_loader):
        image, depth = sample_batched['image'], sample_batched['depth']
        depth = depth.cuda(non_blocking=True)
        image = image.cuda()
        output = model(image)

        output = torch.nn.functional.interpolate(output,size=(440,440),mode='bilinear')




        batchSize = depth.size(0)
        testing_loss(depth,output,losses,batchSize)


        totalNumber = totalNumber + batchSize

       

        errors = util.evaluateError(output, depth,i,batchSize)

        errorSum = util.addErrors(errorSum, errors, batchSize)
        averageError = util.averageErrors(errorSum, totalNumber)
     

    averageError['RMSE'] = np.sqrt(averageError['MSE'])
    loss = float((losses.avg).data.cpu().numpy())



    print('Model Loss {loss:.4f}\t'
        'MSE {mse:.4f}\t'
        'RMSE {rmse:.4f}\t'
        'MAE {mae:.4f}\t'
        'SSIM {ssim:.4f}\t'.format(loss=loss,mse=averageError['MSE']\
            ,rmse=averageError['RMSE'],mae=averageError['MAE'],\
            ssim=averageError['SSIM']))






def testing_loss(depth , output, losses, batchSize):
    
    ones = torch.ones(depth.size(0), 1, depth.size(2),depth.size(3)).float().cuda()
    get_gradient = sobel.Sobel().cuda()
    cos = nn.CosineSimilarity(dim=1, eps=0)
    depth_grad = get_gradient(depth)
    output_grad = get_gradient(output)
    depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
    depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
    output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
    output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)
    depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
    output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

    loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()

    loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
    loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
    loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()
    loss = loss_depth + loss_normal + (loss_dx + loss_dy)
    losses.update(loss.data, batchSize)





def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained=None)
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]  


if __name__ == '__main__':
    main()
