# coding: utf-8
import cv2
from PIL import Image
import os

face = cv2.CascadeClassifier('E:/dealimage/detection_models/haarcascade_frontalface_default.xml')

image_path='E:/data/facetasks.txt'

box_file = 'E:/data/facebox.txt'
box_f = open(box_file, 'a')

for line in open(image_path):
    curline = line.strip('\n')
    sample_image=cv2.imread('E:/data/majorperson/'+curline)

    faces = face.detectMultiScale(sample_image, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))

    count = 0
    for (x, y, w, h) in faces:
        x=x-15
        y=y-15
        w=w+15
        h=h+15
        x2 = x + w
        y2 = y + h

        image = Image.open('E:/data/majorperson/'+curline)  # open只能打开一个具体的文件，而不能打开一个文件夹，否则会报denied error
        img = image.crop((x, y, x2, y2))  # crop只能接受一个值，必须将四个值用括号括起来
        tempname=curline.strip('.jpg')
        img.save('E:/data/cropface/%s'%tempname+'_%d.jpg'%count)
        box_f.write('%s'%tempname+'_%d.jpg'%count+','+str(x)+','+str(y)+','+str(w)+','+str(h)+'\n')
        count=count+1
