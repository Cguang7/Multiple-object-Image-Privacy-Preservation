# coding: utf-8
import scipy.misc as misc
import random
import numpy as np

MOSAIC_RANGE = [2, 7]

# 打码函数
def drawMask(imname):
    # print(file_name)
    file_name = "E:/data/action/"+imname
    img = misc.imread(file_name).astype(dtype=np.float)
    img_out = img.copy()
    row, col, channel = img.shape
    half_patch = np.random.randint(MOSAIC_RANGE[0], MOSAIC_RANGE[1] + 1, 1)[0]
    for i in range(half_patch, row - 1 - half_patch, half_patch):
        for j in range(half_patch, col - 1 - half_patch, half_patch):
            k1 = random.random() - 0.5
            k2 = random.random() - 0.5
            m = np.floor(k1 * (half_patch * 2 + 1))
            n = np.floor(k2 * (half_patch * 2 + 1))
            h = int((i + m) % row)
            w = int((j + n) % col)
            img_out[i - half_patch:i + half_patch, j - half_patch:j + half_patch, :] = img[h, w, :]

    misc.imsave('E:/data/actionallmosaic/'+imname, img_out)



if __name__ == "__main__":
    for line in open('E:/data/persontasks.txt'):
        iname=line.strip('\n')
        drawMask(iname)
