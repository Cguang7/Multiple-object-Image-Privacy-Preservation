# coding: utf-8
from PIL import Image
import linecache

box_file = 'E:/data/majorbox.txt'
person_file = 'E:/data/fakemajorpersontasks.txt'
count = 0
lastname=""
countline=1

for line in open(box_file):
    boxline = line.strip("\n")
    curline = line.split(",")
    im_name = curline[0]
    i_name=im_name.strip('.jpg')

    f_name=i_name+'_'+str(count)
    count=count+1
    print(f_name+'.jpg')

    fake=linecache.getline(person_file, countline)

    fakeline = fake.strip("\n")
    p_name=fakeline.strip('.jpg')
    person_fake = p_name.split("_")
    print(p_name)

    if f_name==p_name:
        x = int(float(curline[1]))
        y = int(float(curline[2]))
        x2 = int(float(curline[3]))
        y2 = int(float(curline[4]))
        w=x2-x
        h=y2-y

        if lastname==person_fake[0]:
            img = Image.open('E:/data/fakeperson/'+i_name+'.jpg')
            image = Image.open('E:/data/fakemajorperson/'+p_name+'.jpg')
            image.resize((w, h))
            reimage = image.resize((w, h))
            print(p_name)

            img.paste(reimage, (x, y, x+w, y+h))  # 用来覆盖，位置

            img.save('E:/data/fakeperson/'+i_name+'.jpg')
            lastname=person_fake[0]
        else:
            img = Image.open('E:/data/person/' + i_name + '.jpg')
            print(p_name)
            image = Image.open('E:/data/fakemajorperson/' +p_name+'.jpg')
            print(i_name)
            reimage=image.resize((w, h))

            img.paste(reimage, (x, y, x+w, y+h))  # 用来覆盖，位置

            img.save('E:/data/fakeperson/' + i_name + '.jpg')
            lastname = person_fake[0]

        countline=countline+1

    else:
        continue