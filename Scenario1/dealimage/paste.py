# coding: utf-8
import cv2
import shutil
import os
import numpy as np


for line in open("E:/data/maskscene2/tasks.txt"):#scenemask,mask,maskscene
    name = line.strip('\n')

    #img_back = cv2.imread('E:/data/actionnoise25/' + name + '.jpg')  # actionnoise25,50,blur
    #img = cv2.imread('E:/data/process2/'+name+'.jpg')#actionmosaic,process1,process2
    img = cv2.imread('E:/data/blur/'+name+'.jpg')#actionnoise25,50,blur,actionallmosaic
    img_back = cv2.imread('E:/data/process2/'+name+'.jpg')#actionmosaic,process1,process2

    #缩放
    #rows,cols,channels = img_back.shape
    img_back = cv2.resize(img_back,(336,336))
    #cv2.imshow('img_back',img_back)

    img = cv2.resize(img,(336,336))
    #cv2.imshow('img',img)

    rows,cols,channels = img.shape #一定是前景图片的

    #mask
    rootdir = 'E:/data/maskscene2/'+name #scenemask,mask,maskscene
    list = os.listdir(rootdir)
    mask =cv2.imread(rootdir+'/'+list[0],1)
    #mask = cv2.imread(rootdir + '/' + list[0], 0)

    mask_r = cv2.resize(mask, (cols, rows))
    hsv = cv2.cvtColor(mask_r, cv2.COLOR_BGR2HSV)

    #lower_green = np.array([0, 0, 221])
    #upper_green = np.array([180, 30, 255])
    lower_green = np.array([0, 0, 0])
    upper_green = np.array([180, 255, 46])

    mask_r = cv2.inRange(hsv, lower_green, upper_green)
    #mask_r = 255 - mask_r

    #腐蚀膨胀
    erode = cv2.erode(mask_r,None,iterations=1)
    #cv2.imshow('erode',erode)
    dilate = cv2.dilate(erode,None,iterations=1)
    #cv2.imshow('dilate', dilate)
    #cv2.waitKey(0)

    #遍历替换
    center=[0,0]
    for i in range(rows):
        for j in range(cols):
            if dilate[i,j]==0:
                img_back[center[0]+i,center[1]+j]=img[i,j]

    #cv2.imshow('res',img_back)
    cv2.imwrite('E:/data/process3/%s.jpg'%name,img_back)#process1,process2,process3


for i in range(0,100):
    if i<10:
        imname = "0"+str(i)+".jpg"
    else:
        imname = str(i)+".jpg"

    path='E:/data/process3/'+imname #process1,process2,process3

    if os.path.isfile(path):
        continue
    else:
        src = 'E:/data/process2/'+imname #actionmosaic,process1,process2
        dst = 'E:/data/process3/'+imname #process1,process2,process3
        shutil.copyfile(src, dst)