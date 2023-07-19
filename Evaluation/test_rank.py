import sys
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
import torch.fft as fft
import argparse
from torchmetrics import ConfusionMatrix
from train import Model
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def main(args):
      model_path = args.model_path

      if args.backbone_model == 'resnet18':
            from HFC.blocks.resnet.Blocks import BasicBlock
      elif args.backbone_model == 'resnet50':
            from HFC.blocks.resnet.Blocks import Bottleneck
         


      model = Model.load_from_checkpoint(model_path)
      model.to(device)
      model.eval()
      model.freeze()
      encoder = model.backbone_model

      confmat = ConfusionMatrix(num_classes=10)
      size = 224
      transform=transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor(),transforms.Normalize([0.479838, 0.470448, 0.429404], [0.258143, 0.252662, 0.272406])])
      # Model performance on the original test set
      Matrix1 = torch.zeros((10,10))
      data_test =  ImageFolder('./datasets/ImageNet/val/',transform=transform)
      test_loader = torch.utils.data.DataLoader(data_test, batch_size= 32, shuffle=False,num_workers=4)
      for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            _, y_hat = encoder(x)

            Matrix1 += confmat(y_hat.cpu(), y.cpu())
      print(Matrix1)
      
      # Testing importance of each frequency
      batchsize = 100
      test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=batchsize, shuffle=False)
      result_prediction = {}
      result_loss = {}
      criterion1 = nn.CrossEntropyLoss()
      for test_class in ([0,1,2,3,4,5,6,7,8,9]):
            prection_matrix = torch.zeros(size,size)
            loss_matrix = torch.zeros(size,size)
            patch_size = args.patch_size
            image_size = 224

            for r in range(int(image_size/patch_size)):
            
                  for c in range(int(image_size/patch_size/2)+1):
                        mask = torch.ones((image_size,image_size))
                        mask[patch_size*0:patch_size*(0+1),int(image_size/2+patch_size):]=0 

                        mask[patch_size*r:patch_size*(r+1),patch_size*c:patch_size*(c+1)] =  0
                        if int(image_size/patch_size)-r<int(image_size/patch_size) and int(image_size/patch_size)-c<int(image_size/patch_size):
                              mask[224-patch_size*(r):224-patch_size*(r-1),224-patch_size*(c):224-patch_size*(c-1)] =  0
                        
                  
                        # correct = 0
                        loss = 0
                        for x,y in test_loader:
                              
                              x1=x
                              sizex = x1.size()
                              reference_class = torch.ones(sizex[0])*test_class
                              if  (y.to(device) == reference_class.to(device)).int().sum()>0:
                        
                                    y1 = torch.zeros(sizex,dtype=torch.complex128)
                                    y1 = fft.fftshift(fft.fft2(x1))
                                    for num_s in range(sizex[0]):
                                          for channel in range(3):
                                                y1[num_s,channel,:,:] = y1[num_s,channel,:,:] * mask                    

                                    x1 = fft.ifft2(fft.ifftshift(y1))
                                    x1 = torch.real(x1)
                                    x1 = torch.Tensor(x1).to(device)
                                 
                                    _, y_hat = encoder(x1)
                                    _, predicted = torch.max(y_hat.data,1)

                                    correct_predictions = (predicted == y.to(device))
                                    correct_predictions = correct_predictions.int()
                                   
                                    # selecting images of the corresponding class
                                    tested_classes = (y.to(device) == reference_class.to(device))
                                    tested_classes = tested_classes.int()
                                    
                                    # correct += (tested_classes*correct_predictions).sum().item()
                                    tc = torch.unsqueeze(tested_classes,1)
                                    test_cla = torch.cat((tc,tc,tc,tc,tc,tc,tc,tc,tc,tc),1).to(device)
                                  
                                    loss += criterion1(test_cla*y_hat,tested_classes*y.to(device))
                                    
                  
                        # prection_matrix[patch_size*r:patch_size*(r+1),patch_size*c:patch_size*(c+1)] = correct/50.0
                        loss_matrix[patch_size*r:patch_size*(r+1),patch_size*c:patch_size*(c+1)] = loss
                        if int(image_size/patch_size)-r<int(image_size/patch_size) and int(image_size/patch_size)-c<int(image_size/patch_size):
                              # prection_matrix[224-patch_size*(r):224-patch_size*(r-1),224-patch_size*(c):224-patch_size*(c-1)] = correct/50.0
                              loss_matrix[224-patch_size*(r):224-patch_size*(r-1),224-patch_size*(c):224-patch_size*(c-1)] = loss
                        result_prediction.update({test_class:prection_matrix})
                        result_loss.update({test_class:loss_matrix})
         
                  
                  

           
      # with open(args.backbone_model+'prediction'+str(args.patch_size)+'.pkl', 'wb') as f:
      #       pickle.dump(result_prediction, f)
      # f.close()
      with open(args.backbone_model+'loss'+str(args.patch_size)+'.pkl', 'wb') as f:
            pickle.dump(result_loss, f)
      f.close()
      
if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument('--backbone_model', type=str, default='resnet18',
                              help='model ')
      parser.add_argument('--model_path', type=str, default='None',
                              help='path of the model')
      parser.add_argument('--patch_size', type=int, default=1,
                              help='patch_size')
      

      
      args = parser.parse_args() 

      main(args)



