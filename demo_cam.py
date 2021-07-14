import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import time

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3)/2)
frame_height = int(cap.get(4))
face_cascade = cv2.CascadeClassifier("Face_detec.xml")
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

tim = 0
DEVICE = 'cuda'

def load_image(imfile):
    img = np.array((imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo,image1_real):
    tim = time.time()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    img = cv2.cvtColor(image1_real, cv2.COLOR_RGB2BGR )
    #cv2.imshow('frame',img/255)
    # map flow to rgb image
    #print(img.shape)
    (x_tt,y_tt,_) = image1_real.shape
    u,v = flow_viz.flow_to_image(flo)
    u = u.flatten()
    v = v.flatten()
    gray = cv2.cvtColor(image1_real, cv2.COLOR_RGB2GRAY)
    try:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(image1_real,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.circle(image1_real, (int(x+w/2), int(y+h/2)), 5, (255, 255, 255), -1)
            print(math.sqrt(u[int(x_tt*(y-1)+x)]*u[int(x_tt*(y-1)+x)]+v[int(x_tt*(y-1)+x)]*v[int(x_tt*(y-1)+x)]))
    except:
        pass

    #flo = cv2.imread(flo)
    #img_flo = np.concatenate([img, flo], axis=0)
    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()
    
    #out.write(flo)

    # Display the resulting frame    
    
    cv2.imshow('frame',image1_real)
    # Press Q on keyboard to stop recording
    if cv2.waitKey(40) & 0xFF == ord('q'):
        pass
    print(-tim + time.time())


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
       
        while True:
            tim = time.time()
            ret,frame = cap.read()
            if frame is not None:
                #frame = cv2.flip(frame,1)
                (h,w,c) = frame.shape
                image1_real = frame[:,:int(w/2)]


                image2 = frame[:,int((w/2) ):]
            
                image1 = load_image(image1_real)

                image2 = load_image(image2)

                #padder = InputPadder(image1.shape)
                #image1, image2 = padder.pad(image1, image2)
            
                flow_low, flow_up = model(image1, image2, iters=10, test_mode=True)
                print(flow_up.shape)
                viz(image1, flow_up,image1_real )




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default = "./checkpoints/3801_raft-kitti.pth",help="restore checkpoint")
    parser.add_argument('--path',default = "./demo-frames/" ,help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
