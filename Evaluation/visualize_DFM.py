import pickle
import argparse
import matplotlib.pyplot as plt 
import numpy as np

def main(args):
    dir = './DFMs/'
    
   
    imagenet_classes = ['Airliner','Wagon','Humming\n Bird','Siamese\n Cat','Ox','Golden\n Retriever','Tailed\n Frog','Zebra','Container\n Ship','Trailer\n Truck']
    m_path = args.DFMs+'.pkl'
    with open(dir+m_path, 'rb') as f:
        all_mask = pickle.load(f)
    fig, axs = plt.subplots(1,10,sharex=True,sharey=True)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    for mask_i in range(len(all_mask)):
        map = np.array(all_mask[mask_i])
        axs[mask_i].imshow(map,cmap='gray')
        axs[mask_i].set_title(imagenet_classes[mask_i])
        axs[mask_i].set_yticks([])
        axs[mask_i].set_xticks([])

    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.8, wspace=0.05, hspace=-0.85)
    plt.savefig(dir + args.DFMs + '.pdf',bbox_inches='tight')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DFMs', type=str, default='resnet18_DFM_1',
                            help='File name of DFMs')
    
    args = parser.parse_args() 

    main(args)