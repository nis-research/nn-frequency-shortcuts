import pickle
import matplotlib.pyplot as plt 
import numpy as np
dir = './DFMs/'
all = {}
count = 0
imagenet_classes = ['Airliner','Wagon','Humming\n Bird','Siamese\n Cat','Ox','Golden\n Retriever','Tailed\n Frog','Zebra','Container\n Ship','Trailer\n Truck']
for m_path in (['resnet18_loss2acc5','resnet18autoaug_loss2acc5','resnet18augmix_loss2acc5','resnet18_style_v2loss2acc5','resnet50_loss2acc5','vgg16_loss2acc5']):
    with open(dir+m_path+'.pkl', 'rb') as f:
        alexd = pickle.load(f)
        all.update({count:alexd})
    count += 1

name = [ '' ,'AutoAugment','AugMix', 'Stylized-IN', '','']
fig, axs = plt.subplots(len(name),10,sharex=True,sharey=True)
fig.set_figheight(15)
fig.set_figwidth(15)
r=0
for model in range(len(name)):
    all_mask = all[model]
    for mask_i in range(10):
        map = np.array(all_mask[mask_i])
        # print(mask_i)
        # print(map.shape)
        axs[r,mask_i].imshow(map,cmap='gray')
        if r == 0:
            axs[0,mask_i].set_title(imagenet_classes[mask_i])#+'\n'+str(delta[model][mask_i])) 
        # else:
            # axs[r,mask_i].set_title(str(delta[model][mask_i]))
        axs[r,mask_i].set_yticks([])
        axs[r,mask_i].set_xticks([])
    if r==0:
        axs[r,0].set_ylabel('ResNet18 \n')
    elif r == 1:
        axs[r,0].set_ylabel('ResNet18 \n'+name[model])
    elif r == 2:
        axs[r,0].set_ylabel('ResNet18 \n'+name[model])
    elif r == 3:
        axs[r,0].set_ylabel('ResNet18 \n'+name[model])
    elif r == 4:
        axs[r,0].set_ylabel('ResNet50 \n'+name[model])
    elif r == 5:
        axs[r,0].set_ylabel('VGG16  \n'+name[model])
    
    r += 1
plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.8, wspace=0.05, hspace=-0.85)
plt.savefig('masks_rank5.pdf',bbox_inches='tight')
