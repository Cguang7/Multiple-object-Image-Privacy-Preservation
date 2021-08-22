# coding: utf-8
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import linecache
import os


CLS2IDX = {0: 'airport_terminal',
 1: 'landing_field',
 2: 'airplane_cabin',
 3: 'amusement_park',
 4: 'skating_rink',
 5: 'arena/performance',
 6: 'art_room',
 7: 'assembly_line',
 8: 'baseball_field',
 9: 'football_field',
 10: 'soccer_field',
 11: 'volleyball_court',
 12: 'golf_course',
 13: 'athletic_field',
 14: 'ski_slope',
 15: 'basketball_court',
 16: 'gymnasium',
 17: 'bowling_alley',
 18: 'swimming_pool',
 19: 'boxing_ring',
 20: 'racecourse',
 21: 'farm/farm_field',
 22: 'orchard/vegetable',
 23: 'pasture',
 24: 'countryside',
 25: 'greenhouse',
 26: 'television_studio',
 27: 'templeeast_asia',
 28: 'pavilion',
 29: 'tower',
 30: 'palace',
 31: 'church',
 32: 'street',
 33: 'dining_room',
 34: 'coffee_shop',
 35: 'kitchen',
 36: 'plaza',
 37: 'laboratory',
 38: 'bar',
 39: 'conference_room',
 40: 'office',
 41: 'hospital',
 42: 'ticket_booth',
 43: 'campsite',
 44: 'music_studio',
 45: 'elevator/staircase',
 46: 'garden',
 47: 'construction_site',
 48: 'general_store',
 49: 'specialized_shops',
 50: 'bazaar',
 51: 'library/bookstore',
 52: 'classroom	',
 53: 'ocean/beach',
 54: 'firefighting',
 55: 'gas_station',
 56: 'landfill',
 57: 'balcony',
 58: 'recreation_room',
 59: 'discotheque',
 60: 'museum',
 61: 'desert/sand',
 62: 'raft',
 63: 'forest',
 64: 'bridge',
 65: 'residential_neighborhood',
 66: 'auto_showroom',
 67: 'lake/river',
 68: 'aquarium',
 69: 'aqueduct',
 70: 'banquet_hall',
 71: 'bedchamber',
 72: 'mountain',
 73: 'station/platform',
 74: 'lawn',
 75: 'nursery',
 76: 'beauty_salon',
 77: 'repair_shop',
 78: 'rodeo',
 79: 'igloo,ice_engraving'}

from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from baselines.ViT.ViT_explanation_generator import LRP

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])


# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


# initialize ViT pretrained
model = vit_LRP(pretrained=True).cuda()
model.eval()
attribution_generator = LRP(model)


def generate_visualization(original_image, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(),
                                                                 method="transformer_attribution",
                                                                 index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
                transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
                image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def print_top_classes(predictions, **kwargs):
    # Print Top-5 predictions
    prob = torch.softmax(predictions, dim=1)
    class_indices = predictions.data.topk(5, dim=1)[1][0].tolist()
    max_str_len = 0
    class_names = []
    for cls_idx in class_indices:
        class_names.append(CLS2IDX[cls_idx])
        if len(CLS2IDX[cls_idx]) > max_str_len:
            max_str_len = len(CLS2IDX[cls_idx])

    print('Top 5 classes:')
    for cls_idx in class_indices:
        output_string = '\t{} : {}'.format(cls_idx, CLS2IDX[cls_idx])
        output_string += ' ' * (max_str_len - len(CLS2IDX[cls_idx])) + '\t\t'
        output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
        print(output_string)

linenum=1
for line in open('E:/data/persontasks.txt'):
    imname=line.strip('\n')
    target = imname.strip('.jpg')
    image = Image.open('E:/data/personmosaic/'+imname)#E:/data/action/
    dog_cat_image = transform(image)

    scene=linecache.getline('E:/data/scene2.txt',linenum).strip('\n')
    id = int(scene)
    linenum=linenum+1


    output = model(dog_cat_image.unsqueeze(0).cuda())
    print_top_classes(output)

# cat - the predicted class
#cat = generate_visualization(dog_cat_image)

# dog
# generate visualization for class 243: 'bull mastiff'


    dog = generate_visualization(dog_cat_image, class_index=id)

    hsv = cv2.cvtColor(dog, cv2.COLOR_BGR2HSV)

    lower_green = np.array([11,43,46])
    upper_green = np.array([155, 255, 255])

    green_mask = cv2.inRange(hsv,lower_green,upper_green)

    path = 'E:/data/maskscene2/' + target
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        cv2.imwrite(path + '/' + "%s.jpg" % scene, green_mask)
    else:
        cv2.imwrite(path + '/' + "%s.jpg" % scene, green_mask)




#plt.imshow(cat)
#plt.axis('off')  # 关掉坐标轴为 off
#plt.show()


























"""
import io
import os
from PIL import Image
from config import get

# 数据集根目录
DATA_ROOT = 'signs'

# 标签List
LABEL_MAP = get('LABEL_MAP')


# 标注生成函数
def generate_annotation(mode):
    # 建立标注文件
    with open('{}/{}.txt'.format(DATA_ROOT, mode), 'w') as f:
        # 对应每个用途的数据文件夹，train/valid/test
        train_dir = '{}/{}'.format(DATA_ROOT, mode)

        # 遍历文件夹，获取里面的分类文件夹
        for path in os.listdir(train_dir):
            # 标签对应的数字索引，实际标注的时候直接使用数字索引
            label_index = LABEL_MAP.index(path)

            # 图像样本所在的路径
            image_path = '{}/{}'.format(train_dir, path)

            # 遍历所有图像
            for image in os.listdir(image_path):
                # 图像完整路径和名称
                image_file = '{}/{}'.format(image_path, image)

                try:
                    # 验证图片格式是否ok
                    with open(image_file, 'rb') as f_img:
                        image = Image.open(io.BytesIO(f_img.read()))
                        image.load()

                        if image.mode == 'RGB':
                            f.write('{}\t{}\n'.format(image_file, label_index))
                except:
                    continue


generate_annotation('train')  # 生成训练集标注文件
generate_annotation('valid')  # 生成验证集标注文件
generate_annotation('test')  # 生成测试集标注文件
"""

