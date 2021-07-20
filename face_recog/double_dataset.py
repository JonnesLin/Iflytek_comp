import random
import torch
import torch.utils.data as data
import numpy as np
import torch.nn.functional as F

class DoubleDataset(data.Dataset):
    def __init__(self, dataset, img_transform, background_transform):
        # load dataset and transform
        self.dataset = dataset
        self.img_transform = img_transform
        self.background_transform = background_transform

    def __getitem__(self, item):
        """需要取两张图片img1, img2 进行混合"""
        lam = np.random.uniform(0, 1)
        # 1. 获取两张图片
        img1_index = item
        img2_index = random.randint(0, self.__len__()-1)
        # img3_index = random.randint(0, self.__len__()-1)
        
        img1, img1_label = self.dataset.__getitem__(img1_index)
        img2, img2_label = self.dataset.__getitem__(img2_index)
        
        img1_label = F.one_hot(torch.tensor(img1_label).long(), num_classes=100)*1.0
        img2_label = F.one_hot(torch.tensor(img2_label).long(), num_classes=100)*1.0
        
        # lam = np.random.beta(1, 1)
        lam = 0.5
        # 2. 对两张图片进行数据增强
        img1 = self.img_transform(img1)
        img2 = self.img_transform(img2)
        pro = np.random.uniform(0, 1)
        if pro > 0.5:
            return img1, img1_label
        
        bbx1, bby1, bbx2, bby2 = rand_bbox(img1.size(), 0.5)
        img1[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img1.size()[-1] * img1.size()[-2]))
        lb_onehot = img1_label * lam + img2_label * (1. - lam)
        
        return img1, lb_onehot
#         # img3 = self.img_transform(img3)
        
#         # 3. 创建一个画布(比两张图片大)
#         channel, w, h = img1.size()
#         background = torch.zeros((channel, int(3*w), int(3*h)))
#         background_size = background.size()

#         # 4. 将两张图片放到画布上
#         # 4.1 获取坐标
#         img1_location_x = random.randint(0, background_size[2]-img1.size()[2])
#         img1_location_y = random.randint(0, background_size[1]-img1.size()[1])
#         img2_location_x = random.randint(0, background_size[2]-img2.size()[2])
#         img2_location_y = random.randint(0, background_size[1]-img2.size()[1])
#         # img3_location_x = random.randint(0, background_size[2]-img3.size()[2])
#         # img3_location_y = random.randint(0, background_size[1]-img3.size()[1])
        
#         # 4.2 放置
#         background[:, img1_location_x:img1_location_x+img1.size()[2], img1_location_y:img1_location_y+img1.size()[1]] \
#             += img1[:, :, :]
#         background[:, img2_location_x:img2_location_x+img2.size()[2], img2_location_y:img2_location_y+img2.size()[1]] \
#             += img2[:, :, :]
#         # background[:, img3_location_x:img3_location_x+img3.size()[2], img3_location_y:img3_location_y+img3.size()[1]] \
#             # += img3[:, :, :]
        
#         # 5. 对background进行transform
#         background = self.background_transform(background)
#         if 
#         return background, img1_label, img2_label
        

    def __len__(self):
        return len(self.dataset)



def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2