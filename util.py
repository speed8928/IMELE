#original script: https://github.com/fangchangma/sparse-to-dense/blob/master/utils.lua
import torch
import math
import numpy as np
from ssim import pytorch_ssim
from PIL import Image
import cv2
import torch
def lg10(x):
    return torch.div(torch.log(x), math.log(10))

def maxOfTwo(x, y):
    z = x.clone()
    maskYLarger = torch.lt(x, y)
    z[maskYLarger.detach()] = y[maskYLarger.detach()]
    return z

def nValid(x):
    return torch.sum(torch.eq(x, x).float())

def nNanElement(x):
    return torch.sum(torch.ne(x, x).float())

def getNanMask(x):
    return torch.ne(x, x)

def setNanToZero(input, target):
    nanMask = getNanMask(target)
   
    nValidElement = nValid(target)

    _input = input.clone()
    _target = target.clone()

    _input[nanMask] = 0
    _target[nanMask] = 0

    #_input = torch.where(_input < torch.tensor(4), torch.tensor(0), _input)
    #_target = torch.where(_target < torch.tensor(4), torch.tensor(0), _target)


    return _input, _target, nanMask, nValidElement


def evaluateError(output, target, idx, batches):

    errors = {'MSE': 0, 'RMSE': 0, 'MAE': 0,'SSIM':0}
                                                                                                                                                                                                                                                                                                                                                                    
    _output, _target, nanMask, nValidElement = setNanToZero(output, target)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
   

    if (nValidElement.data.cpu().numpy() > 0):


        


        output_0_1 = _output.cpu().detach().numpy()
        target_0_1 = _target.cpu().detach().numpy()



        #x = np.reshape(output_0_1,[500,500])
        #x = x *100000
        #x = x.astype('uint16');
     
        #cv2.imwrite(str(idx)+'_out.png',x)
        
        output_0_1= output_0_1*100
        target_0_1 = target_0_1*100


        idx_zero = np.where(target_0_1 <=1)
        output_0_1[idx_zero] = 0
        #target_0_1[idx_zero] = 0
        output_0_1[np.where(output_0_1>=30)] = 0

       
        output_0_1 = torch.from_numpy(output_0_1).float().to(device)
        target_0_1 = torch.from_numpy(target_0_1).float().to(device)
        




        

        diffMatrix = torch.abs((output_0_1) - (target_0_1))




        IMsize = target_0_1.shape[2]*target_0_1.shape[3]

        errors['MSE'] = torch.sum(torch.pow(diffMatrix, 2)) / IMsize / batches
        errors['MAE'] = torch.sum(diffMatrix) / IMsize / batches
        ssim_loss = pytorch_ssim.SSIM(window_size = 15)
        errors['SSIM'] = ssim_loss(_output,_target)
       
       

       
       

        errors['MSE'] = float(errors['MSE'].data.cpu().numpy())
        errors['SSIM'] = float(errors['SSIM'].data.cpu().numpy())
        errors['MAE'] = float(errors['MAE'].data.cpu().numpy())
        #errors['SSIM'] = float(errors['SSIM'])

    return errors


def addErrors(errorSum, errors, batchSize):
    errorSum['MSE']=errorSum['MSE'] + errors['MSE'] * batchSize
    errorSum['SSIM']=errorSum['SSIM'] + errors['SSIM'] * batchSize
    errorSum['MAE']=errorSum['MAE'] + errors['MAE'] * batchSize

    return errorSum


def averageErrors(errorSum, N):

    averageError= {'MSE': 0, 'RMSE': 0, 'MAE': 0,'SSIM':0}
    averageError['MSE'] = errorSum['MSE'] / N
    averageError['SSIM'] = errorSum['SSIM'] / N
    averageError['MAE'] = errorSum['MAE'] / N


    return averageError

def feature_plot(feats,w,h):

    feats = feats.cpu().detach().numpy()
    channel = feats.shape[1]
    feats = np.reshape(feats,(channel,w,h))
    for idx in range(16):
        m = feats[idx]*10000
        m = m.astype('uint16')
        cv2.imwrite('Fusion_features/'+str(idx)+'_mf_features.png',m)
        
    
    return feats
  






	
