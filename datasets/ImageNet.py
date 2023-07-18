import os
import numpy as np
from torch.utils.data.dataset import Dataset
import torch
from PIL import Image
class ImageNet(Dataset):
    def __init__(self, root,train = True, transform = None):
        super(ImageNet).__init__()
        if train == True:
            self.labels_path = os.path.join(root,'ImageNet','select_tiny_label_train.npy')
            self.root_dir = os.path.join(root,'ImageNet','select_tiny_train.npy')
        elif train == 'test':
            self.labels_path = os.path.join(root,'ImageNet','select_tiny_label_val.npy')
            self.root_dir = os.path.join(root,'ImageNet','select_tiny_val.npy')
        
        self.transform = transform

        self.data = np.load(self.root_dir, allow_pickle=True)
        self.targets = np.load(self.labels_path, allow_pickle=True) 
        # self.data = self.data.transpose((0, 3, 1, 2))

    
    def __len__(self): 
        return len(self.data)

    def __getitem__(self, index):
        
        img = self.data[index]        
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        target = torch.tensor(int(self.targets[index]))
        
        return img, target