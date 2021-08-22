# coding: utf-8
import cv2
import linecache
from PIL import Image

#马赛克
def do_mosaic(frame, x, y, w, h, neighbor=9):
    """
    马赛克的实现原理是把图像上某个像素点一定范围邻域内的所有点用邻域内左上像素点的颜色代替，这样可以模糊细节，但是可以保留大体的轮廓。
    :param frame: opencv frame
    :param int x : 马赛克左顶点
    :param int y: 马赛克右顶点
    :param int w: 马赛克宽
    :param int h: 马赛克高
    :param int neighbor: 马赛克每一块的宽
    """
    fh, fw = frame.shape[0], frame.shape[1]
    if (y + h > fh) or (x + w > fw):
        return
    for i in range(0, h - neighbor, neighbor):  # 关键点0 减去neightbour 防止溢出
        for j in range(0, w - neighbor, neighbor):
            rect = [j + x, i + y, neighbor, neighbor]
            color = frame[i + y][j + x].tolist()  # 关键点1 tolist
            left_up = (rect[0], rect[1])
            right_down = (rect[0] + neighbor - 1, rect[1] + neighbor - 1)  # 关键点2 减去一个像素
            cv2.rectangle(frame, left_up, right_down, color, -1)



#box_file = 'E:/data/minorbox.txt'
box_file = 'E:/data/vrdtarget2.txt'#vrdtarget,action_box

temp = linecache.getline(box_file, 1)
print(temp)

countline=0

for line in open(box_file):
    boxline = line.strip("\n")
    curline = line.split(",")
    im_name = curline[0]

    x1 = int(float(curline[1]))
    y1 = int(float(curline[2]))
    x2 = int(float(curline[3]))
    y2 = int(float(curline[4]))
    """
    x1 = int(float(curline[2]))
    y1 = int(float(curline[3]))
    x2 = int(float(curline[4]))
    y2 = int(float(curline[5]))
    """

    if countline==0:
        #root = "E:/data/fakeperson"
        root = "E:/data/personmosaic"
        im2 = cv2.imread(root + "/" + im_name, 1)
        do_mosaic(im2, x1, y1, x2 - x1, y2 - y1)
        #root = "E:/data/personmosaic/"
        root = "E:/data/actionmosaic3/"
        cv2.imwrite(root + im_name, im2)
        countline=countline+1
        print(im_name, x1, y1, x2, y2)
    else:
        lastline = linecache.getline(box_file, countline)
        print(lastline)
        last_im = lastline.strip('\n').split(',')[0]
        print(last_im)

        if last_im==im_name:
            #root="E:/data/personmosaic/"
            root = "E:/data/actionmosaic3/"
            im2 = cv2.imread(root + im_name, 1)
            do_mosaic(im2, x1, y1, x2 - x1, y2 - y1)
            cv2.imwrite(root + im_name, im2)
            countline = countline + 1
            print(im_name, x1, y1, x2, y2)
            print('same')
        else:
            #root = "E:/data/fakeperson"
            root = "E:/data/personmosaic"
            im2 = cv2.imread(root + "/" + im_name, 1)
            do_mosaic(im2, x1, y1, x2 - x1, y2 - y1)
            #root = "E:/data/personmosaic/"
            root = "E:/data/actionmosaic3/"
            cv2.imwrite(root + im_name, im2)
            countline = countline + 1
            print(im_name, x1, y1, x2, y2)




    """
    objectlist=["bench","bicycle"]
    if object in objectlist:
        im2 = cv2.imread(root + "/" + im_name, 1)
        root="E:/mosaic/"
        do_mosaic(im2, x1, y1, x2 - x1, y2 - y1)
        cv2.imwrite(root+im_name, im2)
    """