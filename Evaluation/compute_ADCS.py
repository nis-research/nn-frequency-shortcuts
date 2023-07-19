from scipy import signal
from scipy.ndimage import gaussian_filter 
import numpy.fft as fft
import numpy as np
import matplotlib.pyplot as plt 
import torchvision
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

def distance(i, j, imageSize, r1,r2):
    dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
    if dis < r2 and dis >=r1:
        return 1.0
    else:
        return 0

def mask_radial(img, r1,r2):
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r1=r1,r2=r2)
    return mask


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

    
   
Energy = {}

mean = [0.479838, 0.470448, 0.429404]
std = [0.258143, 0.252662, 0.272406]
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),transforms.Normalize(mean, std)]) 
batchsize = 1
data_test =  ImageFolder('./datasets/ImageNet/val/',transform=transform) # data path to be changed
test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=batchsize, shuffle=False)
img_size =224
for x,y in test_loader:
    x1=x[0]
    y1 = np.zeros((img_size,img_size,3),dtype=np.complex128)
    for j in range(3):
        y1[:,:,j] = fft.fftshift(fft.fft2(x1[j,:,:]))
    y1[y1==0] = 12e-12
    log_y1 = np.abs(y1)
    if y.item()  in Energy:
        Energy[y.item()] += log_y1
    else:
        Energy.update({y.item():log_y1})


fig, axs = plt.subplots(2,5,sharex=True,sharey=True)
fig.set_figheight(8)
fig.set_figwidth(20)
for j in range(10):
    olp = np.zeros((img_size,img_size))
    for i in range(10):
        diff = Energy[j] - Energy[i]
        diff = rgb2gray(diff)
        diff[diff>0] = 1
        diff[diff<=0] = -1 
        olp += diff
    if j >=5:
        axs[1,j-5].imshow(olp,cmap='jet',vmin=-9,vmax=9)
        axs[1,j-5].axis('off')
        axs[1,j-5].set_title('Class: %d' %j)
    else:
        axs[0,j].imshow(olp,cmap='jet',vmin=-9,vmax=9)
        axs[0,j].axis('off')
        axs[0,j].set_title('Class: %d' %j) 
plt.rcParams.update({'font.size': 25})
plt.savefig('ADCS_imagenet10.pdf')