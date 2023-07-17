from locale import currency
import sys
from tkinter.tix import COLUMN
sys.path.insert(0,'/home/wangs1/')

import torch
import numpy as np
import torchvision
from torchvision.transforms import transforms
import torch.fft as fft
import argparse
from torchmetrics import ConfusionMatrix
from torchvision.datasets import ImageFolder
from torchmetrics.functional import accuracy
import HFC.backbone.resnet as resnet
import HFC.backbone.alexnet as alexnet
from HFC.blocks.decoder import Decoder
from HFC.blocks.resnet.Blocks import Upconvblock
sys.path.insert(0,'./')
from HFC.oldtrain import Model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def main(args):
    
    model_path = args.model_path

    if args.backbone_model == 'resnet18':
        from HFC.blocks.resnet.Blocks import BasicBlock
        # backbone_model = resnet.ResNet(BasicBlock,[2,2,2,2],args.num_class)
    elif args.backbone_model == 'resnet50':
        from HFC.blocks.resnet.Blocks import Bottleneck
        # backbone_model = resnet.ResNet(Bottleneck,[3,4,6,3],args.num_class)
    elif args.backbone_model == 'densenet121':
        from HFC.blocks.densenet.Blocks import Bottleneck
        # backbone_model = densenet.DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, reduction=0.5, num_classes=args.num_class)
    elif args.backbone_model == 'densenet169':
        from HFC.blocks.densenet.Blocks import Bottleneck
        # backbone_model = densenet.DenseNet(Bottleneck, [6,12,32,32], growth_rate=32, reduction=0.5, num_classes=args.num_class)
    # elif args.backbone_model == 'alexnet':


    model = Model.load_from_checkpoint(model_path)
    model.to(device)
    model.eval()
    model.freeze()
    encoder = model.backbone_model

    confmat = ConfusionMatrix(num_classes=10)

    mean =  [0.479838, 0.470448, 0.429404]
    std = [0.258143, 0.252662, 0.272406]
    transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),transforms.Normalize(mean, std)])
    data_test =  ImageFolder('../datasets/ImageNet/val/',transform=transform)


    test_loader = torch.utils.data.DataLoader(data_test, batch_size= 16, shuffle=False,num_workers=2)
    total = 0
    Matrix2 = torch.zeros((10,10))
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        _, y_hat = encoder(x)
        # _, predicted = torch.max(y_hat.data,1)
        total += y.size(0)
        # correct += (predicted == y).sum().item()
    # total_acc = float(correct/total)
    
        Matrix2 += confmat(y_hat.cpu(), y.cpu())
    print('Confusion Metrix on testing set:')
    print(Matrix2)

    for mask_i in range(10):
        print('TP_f/P -- class %d' % mask_i)  
        delta1 = (Matrix2[mask_i,mask_i])/sum(Matrix2[mask_i,:])
        print(delta1)

        print('FP_f/N -- class %d' % mask_i)  
        delta2 = (sum(Matrix2[:,mask_i])-Matrix2[mask_i,mask_i])/(sum(sum(Matrix2))-sum(Matrix2[mask_i,:]))
        print(delta2)


    batchsize = 1
    
    testset = ImageFolder('../datasets/ImageNet/val/',transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batchsize, shuffle=False)
    
    with open(args.m_path+'.pkl', 'rb') as f:
        all = pickle.load(f)

    for mask_i in all:
        print('Using mask %d' %mask_i)
        mask = np.array(all[mask_i]) #map
        print(len(mask[mask==1]))
        # mask[16,16] = 0
        # mat = np.zeros((10,10))
        mat = torch.zeros((10,10))
        for x,y in test_loader:
                size = x.size()
                x1=x
                y1 = torch.zeros(size,dtype=torch.complex128)
                #print(y1.shape())
                y1 = fft.fftshift(fft.fft2(x1))    
                for num_s in range(size[0]):
                    for channel in range(3):
                        y1[num_s,channel,:,:] = y1[num_s,channel,:,:] * mask  

               
                x1 = fft.ifft2(fft.ifftshift(y1))
                x1 = torch.real(x1)
                x1 = torch.Tensor(x1).to(device)
                _, y_hat = encoder(x1)
                
                # _, predicted = torch.max(y_hat.data,1)
                mat += confmat(y_hat.cpu(), y.cpu())
                # mat[y,predicted] += 1
                

        print(mat)
        
        print('TP_f/P -- class %d' % mask_i)  
        delta1 = (mat[mask_i,mask_i])/sum(Matrix2[mask_i,:])
        print(delta1)

        print('FP_f/N -- class %d' % mask_i)  
        delta2 = (sum(mat[:,mask_i])-mat[mask_i,mask_i])/(sum(sum(Matrix2))-sum(Matrix2[mask_i,:]))
        print(delta2)
        
    print('------------------without contributing frequs')

    for mask_i in range(10):
        print('Using mask %d' %mask_i)
        mask = np.array(all[mask_i]) #map
        print(len(mask[mask==1]))
        mask = 1-mask
        mask[112,112] = 1
        mat = torch.zeros((10,10))
        
        for x,y in test_loader:
                size = x.size()
                x1=x
                y1 = torch.zeros(size,dtype=torch.complex128)
                
                
                y1 = fft.fftshift(fft.fft2(x1))    
                for num_s in range(size[0]):
                    for channel in range(3):
                        y1[num_s,channel,:,:] = y1[num_s,channel,:,:] * mask  
             

                
                x1 = fft.ifft2(fft.ifftshift(y1))
                x1 = torch.real(x1)
                x1 = torch.Tensor(x1).to(device)
                _, y_hat = encoder(x1)

                mat += confmat(y_hat.cpu(), y.cpu())
                

        print(mat) 
        print('TP_f/P -- class %d' % mask_i)  
        delta1 = (mat[mask_i,mask_i])/sum(Matrix2[mask_i,:])
        print(delta1)

        print('FP_f/N -- class %d' % mask_i)  
        # print((sum(sum(Matrix2))-sum(Matrix2[mask_i,:])))
        delta2 = (sum(mat[:,mask_i])-mat[mask_i,mask_i])/(sum(sum(Matrix2))-sum(Matrix2[mask_i,:]))
        print(delta2)
        # print('Delta 1: (TP_f-TP_org)/P -- class %d' % mask_i)  
        # delta1 = (mat[mask_i,mask_i]-Matrix2[mask_i,mask_i])/sum(Matrix2[mask_i,:])
        # print(delta1)

        # print('Delta 2: (FP_f-FP_org)/N -- class %d' % mask_i)  
        # # print((sum(sum(Matrix2))-sum(Matrix2[mask_i,:])))
        # delta2 = (sum(mat[:,mask_i])-mat[mask_i,mask_i]-sum(Matrix2[:,mask_i]) + Matrix2[mask_i,mask_i])/(sum(sum(Matrix2))-sum(Matrix2[mask_i,:]))
        # print(delta2)
     
       
        
    
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone_model', type=str, default='resnet18',
                        help='model ')
    parser.add_argument('--model_path', type=str, default='None',
                        help='path of the model')
    parser.add_argument('--m_path', type=str, default='./',
                        help='path of the msk')

 

    args = parser.parse_args()

    main(args)
