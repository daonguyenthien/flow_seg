import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch

from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from torchvision.utils import save_image

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    #u,v = flow_viz.flow_to_image(flo)
    #print(u.flatten())
    #print(v.shape)
    cv2.imwrite("1_flow.png",flo)
    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    
    return flo


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        print(images)
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            #print(image1.shape)
            flow_low, flow_up,img_seg = model(image1, image2, iters=12, test_mode=True)
            #img_seg = soft_skel(img_seg,2)
            '''
            print(img_seg.cpu().detach().numpy().shape)
            img_seg= img_seg.cpu().detach().numpy()
            
            output_image = np.array(img_seg[0,:,:,:])
            print(output_image.shape)
            output_image = helpers.reverse_one_hot(output_image)
            #print(output_image.shape)
            out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
            '''
            
            save_image(img_seg ,'1_seg.png')
            #cv2.imwrite("1.png",img_seg.cpu().detach().numpy() )
            img_flo = viz(image1, flow_up)
            img_seg = np.array(img_seg.cpu().numpy())
            #_,c,w,h = img_seg.shape
            print(img_seg.shape)
            #img_seg = img_seg.transpose(0,3,1,2).reshape(w,h,3)
            #cv2.imwrite("1_disp.png",img_seg )
            print(img_flo.shape)
            print(img_seg.shape)
            #final = cv2.bitwise_and(img_flo[0:w,0:h,:],img_flo[0:w,0:h,:],mask = img_seg)
            #cv2.imwrite("1_final.png",final )
            #segment(viz(image1, flow_up))
            #xent_img(viz(image1, flow_up))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default = "./checkpoints/9001_raft-kitti.pth",help="restore checkpoint")
    parser.add_argument('--path',default = "./demo-frames/" ,help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
