import torch
import numpy as np
from torchvision.transforms import transforms
import torch.fft as fft
import argparse
from torchmetrics import ConfusionMatrix
from torchvision.datasets import ImageFolder
import pickle

import sys
sys.path.insert(0,'/home/wangs1/nn-frequency-shortcuts/')
from train import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def main(args):
    
    model_path = args.model_path

    if args.backbone_model == 'resnet18':
        from blocks.resnet.Blocks import BasicBlock
    elif args.backbone_model == 'resnet50':
        from blocks.resnet.Blocks import Bottleneck

    model = Model.load_from_checkpoint(model_path)
    model.to(device)
    model.eval()
    model.freeze()
    encoder = model.backbone_model

    confmat = ConfusionMatrix(num_classes=10)
    #  model performance on original dataset
    mean =  [0.479838, 0.470448, 0.429404]
    std = [0.258143, 0.252662, 0.272406]
    transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),transforms.Normalize(mean, std)])
    data_test =  ImageFolder('./data/ImageNet/val/',transform=transform)

    
    test_loader = torch.utils.data.DataLoader(data_test, batch_size= 16, shuffle=False,num_workers=2)
    total = 0
    Matrix2 = torch.zeros((10,10))
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        y_hat = encoder(x)
        total += y.size(0)
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

    #  model performance on DFM-filtered datasets
    batchsize = 16
    testset = ImageFolder('./data/ImageNet/val/',transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batchsize, shuffle=False)
    
    with open(args.m_path+'.pkl', 'rb') as f:
        all = pickle.load(f)

    for mask_i in all:
        print('Using mask %d' %mask_i)
        mask = np.array(all[mask_i]) #map
        print(len(mask[mask==1]))
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
            y_hat = encoder(x1)
            mat += confmat(y_hat.cpu(), y.cpu())
                
        print(mat)
        
        print('TP_f/P -- class %d' % mask_i)  
        delta1 = (mat[mask_i,mask_i])/sum(Matrix2[mask_i,:])
        print(delta1)

        print('FP_f/N -- class %d' % mask_i)  
        delta2 = (sum(mat[:,mask_i])-mat[mask_i,mask_i])/(sum(sum(Matrix2))-sum(Matrix2[mask_i,:]))
        print(delta2)     
            
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
