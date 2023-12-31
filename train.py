import torch
import torch.nn as nn
import argparse
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
import timm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import sys
sys.path.insert(0,'/home/wangs1/nn-frequency-shortcuts/')
from data.Synthetic import Synthetic
import backbone.resnet as resnet
import backbone.vgg as vgg
import backbone.alexnet as alexnet


class Model(LightningModule):
    def __init__(self,backbone_model, lr,num_class,dataset,image_size,special):
        super(Model, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.dataset = dataset
        self.num_class = num_class
        self.image_size = image_size
        self.backbone_model = backbone_model
        self.special = special
        
    def forward(self, x):
        # enc, prediction = self.backbone_model(x)
        prediction = self.backbone_model(x)
       
        return prediction
     

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), self.lr,
                                momentum=0.9, nesterov=True,
                                weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min',verbose=True, factor=0.1)
        return {'optimizer': optimizer, 
                'lr_scheduler':scheduler,
                'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        x, y = batch

        criterion1 = nn.CrossEntropyLoss()
       
        # _, y_hat = self(x)
        y_hat = self(x)
        #print(y_hat)
        loss1 = criterion1(y_hat, y)
        loss = loss1
        
        _, predicted = torch.max(y_hat.data,1) 
        self.log_dict({'train_classification_loss': loss1}, on_epoch=True,on_step=True)
        self.log_dict({'train_loss': loss}, on_epoch=True,on_step=True)
        return {"loss": loss,'epoch_preds': predicted, 'epoch_targets': y}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        criterion1 = nn.CrossEntropyLoss()
      
        # _, y_hat = self(x)
        y_hat = self(x)
        # print(y_hat)
        # print(y_hat.size())
        loss1 = criterion1(y_hat, y)
        self.val_loss = loss1
        
        _, predicted = torch.max(y_hat.data,1) 
        self.log_dict( {'val_loss':  self.val_loss}, on_epoch=True,on_step=True)

        return  {'epoch_preds': predicted, 'epoch_targets': y} #self.val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        # _, y_hat = self(x)
        y_hat = self(x)
        # print(y_hat.size())
          
        _, predicted = torch.max(y_hat.data,1)
        
        return {'batch_preds': predicted, 'batch_targets': y}
        
    
    def test_step_end(self, output_results):
        
        self.test_acc(output_results['batch_preds'], output_results['batch_targets'])
        self.log_dict( {'test_acc': self.test_acc}, on_epoch=True,on_step=False)
        
    def training_epoch_end(self, output_results):
        # print(output_results)
        self.train_acc(output_results[0]['epoch_preds'], output_results[0]['epoch_targets'])
        self.log_dict({"train_acc": self.train_acc}, on_epoch=True, on_step=False)

    def validation_epoch_end(self, output_results):
        # print(output_results)
        self.val_acc(output_results[0]['epoch_preds'], output_results[0]['epoch_targets'])
        self.log_dict({"valid_acc": self.val_acc}, on_epoch=True, on_step=False)
        # print(acc)
        # return val_accuracy

    def setup(self, stage):
        if self.dataset == 'synthetic':
            transform_train = transforms.Compose([
                transforms.Pad(4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.498, 0.498, 0.498], [0.172, 0.173042, 0.173])
                # normalize
            ])
            transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.498, 0.498, 0.498], [0.172, 0.173042, 0.173])])
            data_train  = Synthetic('./data',train=True,complex=self.special, transform=transform_train,band = '')
            data_test = Synthetic('./data',train=False,complex=self.special, transform=transform,band = '')
        elif self.dataset == 'imagenet10':
            transform_train = transforms.Compose([
                transforms.Pad(4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(self.image_size),
                # transforms.AugMix(),# transforms.AutoAugment(), # change here to add other augmentations
                transforms.ToTensor(),
                transforms.Normalize([0.479838, 0.470448, 0.429404], [0.258143, 0.252662, 0.272406])
                # normalize
            ])
            transform=transforms.Compose([transforms.Resize((self.image_size,self.image_size)), transforms.ToTensor(),transforms.Normalize([0.479838, 0.470448, 0.429404], [0.258143, 0.252662, 0.272406])])
            data_train  = ImageFolder('./data/ImageNet/train/',transform=transform_train)
            data_test =  ImageFolder('./data/ImageNet/val/',transform=transform)
        elif self.dataset == 'imagenet10_style':
            transform_train = transforms.Compose([
                transforms.Pad(4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.479838, 0.470448, 0.429404], [0.258143, 0.252662, 0.272406])
            ])
            transform=transforms.Compose([transforms.Resize((self.image_size,self.image_size)), transforms.ToTensor(),transforms.Normalize([0.479838, 0.470448, 0.429404], [0.258143, 0.252662, 0.272406])])
            data_train  = ImageFolder('./data/ImageNet_style/train/',transform=transform_train)
            data_test =  ImageFolder('./data/ImageNet_style/val/',transform=transform)
    
        # train/val split
        data_train2, data_val =  torch.utils.data.random_split(data_train, [int(len(data_train)*0.9), len(data_train)-int(len(data_train)*0.9)])

        # assign to use in dataloaders
        self.train_dataset = data_train2
        self.val_dataset = data_val
        self.test_dataset = data_test

    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=64, shuffle=True)#,num_workers=2)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=64, shuffle=False)#,num_workers=2)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=64)#,num_workers=2)


def main(args):
    backbone = ['resnet18', 'resnet34', 'resnet50','resnet101', 'alex', 'ViT', 'vgg16'] 
    print(torch.cuda.device_count())
    if args.backbone_model == 'resnet18':
        from blocks.resnet.Blocks import BasicBlock
        backbone_model = resnet.ResNet(BasicBlock,[2,2,2,2],args.num_class)
    elif args.backbone_model == 'resnet34':
        from blocks.resnet.Blocks import BasicBlock
        backbone_model = resnet.ResNet(BasicBlock, [3,4,6,3],args.num_class)
    elif args.backbone_model == 'resnet50':
        from blocks.resnet.Blocks import Bottleneck
        backbone_model = resnet.ResNet(Bottleneck,[3,4,6,3],args.num_class)
    elif args.backbone_model == 'resnet101':
        from blocks.resnet.Blocks import Bottleneck
        backbone_model = resnet.ResNet(Bottleneck[3,4,23,3],args.num_class)
    elif args.backbone_model == 'alex':
        backbone_model = alexnet.AlexNet(args.num_class)
    elif args.backbone_model == 'ViT':
        backbone_model = timm.create_model('vit_base_patch8_224', pretrained=False)
    

    logger = TensorBoardLogger(args.save_dir, name=args.backbone_model)
    
    model = Model(backbone_model, args.lr,args.num_class,args.dataset,args.image_size, args.special)
    maxepoch = 200
    checkpoints_callback = ModelCheckpoint(save_last=True,save_top_k=-1)
    trainer = pl.Trainer(enable_progress_bar=False,logger=logger, callbacks=[checkpoints_callback], gpus=-1, max_epochs=maxepoch) # accelerator='dp',
    trainer.fit(model)
    trainer.test()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write parameters')
    parser.add_argument('--backbone_model', type=str,
                    help='backbone_model')
    parser.add_argument('--image_size', type=int, default= 32,
                    help='size of images in dataset')
    parser.add_argument('--num_class', type=int, default= 10,
                    help='number of classes in dataset')
    parser.add_argument('--dataset', type=str, default='imagenet10',
                    help='dataset')
    parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')            
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--special', required=False, default=None,
                        help='selecting synthetic dataset')
   
    args = parser.parse_args()
    if not os.path.exists(args.save_dir+'/'+args.backbone_model):
        os.makedirs(args.save_dir+'/'+args.backbone_model)
    print('make the directory')
    
    main(args)