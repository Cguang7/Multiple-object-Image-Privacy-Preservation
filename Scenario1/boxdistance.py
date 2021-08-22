# coding: utf-8
import linecache
from scipy.spatial import distance as dist

personfile = "E:/data/actionperson.txt"
objectfile = "E:/data/actionobject.txt"
boxfile = "E:/data/action_box.txt"
b_f=open(boxfile,'a')


last = "00.jpg"
last2 = "00.jpg"
countset = []

count = 0
for i in range(1, 384):
    objectstring = linecache.getline(objectfile,i)
    curline = objectstring.strip('\n').split(',')
    name = curline[0]
    count = count + 1

    if name != last2:
        countset.append(count-1)
        count = 1

    last2 = name


flag = 0
start=1
temp=0
for line in open(personfile):
    curline = line.strip('\n').split(',')
    name = curline[0]
    print(name)

    if name!=last:
        if temp!=0:
            start=start+countset[flag]
            flag = flag+1

    last = name
    count = curline[1]
    px1 = int(float(curline[2]))
    py1 = int(float(curline[3]))
    px2 = int(float(curline[4]))
    py2 = int(float(curline[5]))
    
        
    min=2000000
    box=""
    temp=0
    for j in range(start, 383):
        objline = linecache.getline(objectfile, j)
        curline2 = objline.strip('\n').split(',')
        name2 = curline2[0]
        ox1 = int(float(curline2[2]))
        oy1 = int(float(curline2[3]))
        ox2 = int(float(curline2[4]))
        oy2 = int(float(curline2[5]))

        if name==name2:
            temp=temp+1
            (xA, yA) = (px1+(px2-px1)*0.5, py1+(py2-py1)*0.5)
            print((xA, yA))
            (xB, yB) = (ox1 + (ox2 - ox1)*0.5, oy1 + (oy2 - oy1)*0.5)
            print((xB, yB))

            # 计算中心点之间的欧式距离
            D = dist.euclidean((xA,yA), (xB, yB))
            print(D)
            
            if D<=min:
                min=D
                box=objline

        else:
            break

    b_f.write(box)






b_f.close()