# coding: utf-8
from PIL import Image

box_file = 'E:/data/facebox.txt'
count = 0
lastname=""

for line in open(box_file):
    boxline = line.strip("\n")
    curline = line.split(",")
    im_name = curline[0]
    person_name=im_name.split("_")
    p_name=person_name[0]+'_'+person_name[1]

    x = int(float(curline[1]))
    y = int(float(curline[2]))
    w = int(float(curline[3]))
    h = int(float(curline[4]))

    if lastname==p_name:
        img = Image.open('E:/data/fakemajorperson/'+p_name+'.jpg')
        image = Image.open('E:/data/fake/'+im_name)
        image.resize((w, h))
        reimage = image.resize((w, h))

        img.paste(reimage, (x, y, x+w, y+h))  # 用来覆盖，位置

        img.save('E:/data/fakemajorperson/'+p_name+'.jpg')
        lastname=p_name
    else:
        img = Image.open('E:/data/majorperson/' + p_name + '.jpg')
        print(p_name)
        image = Image.open('E:/data/fake/' + im_name)
        print(im_name)
        reimage=image.resize((w, h))

        img.paste(reimage, (x, y, x+w, y+h))  # 用来覆盖，位置

        img.save('E:/data/fakemajorperson/' + p_name + '.jpg')
        lastname = p_name





