# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 00:06:53 2021

@author: black
"""

import torch
import torch.nn.functional as F
import torchvision as tv
from torchvision import models
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import cv2
import random
from models import VGGNet,DenseNet,AlexNet,googleNet
import time
    


def plot_img(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300,300),(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    if detections.shape[2] == 0:
        return image
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence>0.5:
            box = detections[0,0,i,3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            img = image[startY:endY,startX:endX,:]
            im = cv2.resize(img,(img_size,img_size))
            im = np.array(im)/255.0
            im = im.reshape(1,3,124,124).astype(np.float32)
            im = Variable(torch.from_numpy(im)).cuda()
            result = model(im)
            result = result.cpu().data
        
            if result[0][0]>result[0][1]:
                label_Y = 'Mask'
            else:
                label_Y = 'Without Mask'
        
            if label_Y == 'Mask':
                image = cv2.rectangle(image, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
            else:
                image = cv2.rectangle(image, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
            cv2.putText(image,label_Y , (startX, endY-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36,255,12), 2)
    return image

def get_video(model):
    cap = cv2.VideoCapture(0)
    cap_video=cv2.VideoCapture(0)
    k=0
    ret,start_frame=cap_video.read()
    time_cost = 0
    while(cap_video.isOpened()):
        start = time.time()
        ret,frame=cap_video.read()
        # frame = np.round(0.9*frame + 0.1*start_frame)
        if ret:

            frame = plot_img(frame)
            cv2.imshow('frame',frame)
            cv2.waitKey(1)
            end = time.time()
            k=k+1
        time_cost += (end-start)
        start_frame = frame
        print('average time cost per frame:{}'.format(time_cost/k))
    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindow()

def process_video(path,output_path,model):
    cap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path,fourcc, 20.0, (width,height))

    while(cap.isOpened()):
        ret,frame = cap.read()
        if ret:

            frame = plot_img(frame)
            out.write(frame)
            end = time.time()
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release() 
    out.release()

    
if __name__ == '__main__':
    model = torch.load('model/model_GoogLeNet.pkl')

    caffeModel = "res10_300x300_ssd_iter_140000.caffemodel"
    prototextPath = "deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(prototextPath,caffeModel)
    img_size = 124
    get_video(model)
    # process_video('mask.mp4','google_output.avi',model)