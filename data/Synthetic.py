import os
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import torch

class Synthetic(Dataset):
    def __init__(self, root_dir,train=True, transform = None, complex = '', band = '',t=True):
        super(Synthetic).__init__()
        
        if train is False:
            
            self.labels_path = os.path.join(root_dir,'synthetic','test_label'+complex +'.npy')
            self.root_dir = os.path.join(root_dir,'synthetic','test_data'+complex +band+'.npy')
        else:
            self.labels_path = os.path.join(root_dir,'synthetic','train_label'+complex+'.npy')
            self.root_dir = os.path.join(root_dir,'synthetic','train_data'+complex+band+'.npy')
    

        print(self.root_dir)
        self.transform = transform
        self.data = np.load(self.root_dir, allow_pickle=True)
        self.targets = np.load(self.labels_path, allow_pickle=True) 
        self.band = band
        self.t = t
        # self.data = self.data.transpose((0, 3, 1, 2))

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.t:
            img = self.data[index].permute(1,2,0).numpy()
        else:
            img = self.data[index] 

        if self.transform is not None:
            # print(np.max(img))
            # print(np.min(img))
            img = np.clip(img,0,1)
            img = img*255 
            img = Image.fromarray(img.astype(np.uint8),mode='RGB')
          
            img = self.transform(img)

        target = self.targets[index]
        
        return img, torch.tensor(target, dtype=torch.long)