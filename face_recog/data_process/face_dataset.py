from torch.utils.data import DataLoader,Dataset
import os
import torch
import numpy as np
import glob
import cv2
from PIL import Image

np.random.seed(1)
all_images = glob.glob('data/Datawhale_人脸情绪识别_数据集/train/*/*')
all_images = np.array(all_images)
np.random.shuffle(all_images)
    
class Face_dataset(Dataset):
    def __init__(self, train=True, transform=None):
        train_len = int(0.9*len(all_images))
        if train:
            imgs = all_images[:train_len]
        else:
            imgs = all_images[train_len:]

        self.img_path = imgs
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    
    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        
        lbl_dict = {'angry': 0,
             'disgusted': 1,
             'fearful': 2,
             'happy': 3,
             'neutral': 4,
             'sad': 5,
             'surprised': 6}
        if self.transform is not None:
            img = self.transform(img)
        
        if 'test' in self.img_path[index]:
            return img, torch.from_numpy(np.array(0))
        else:
            lbl_int = lbl_dict[self.img_path[index].split('/')[-2]]
            return img, torch.from_numpy(np.array(lbl_int))
    
    def __len__(self):
        return len(self.img_path)
    
    

class Face_pred_dataset(Dataset):
    def __init__(self, pred_images, transform=None):
        
        self.img_path = pred_images
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    
    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        
        lbl_dict = {'angry': 0,
             'disgusted': 1,
             'fearful': 2,
             'happy': 3,
             'neutral': 4,
             'sad': 5,
             'surprised': 6}
        if self.transform is not None:
            img = self.transform(img)
        
        if 'test' in self.img_path[index]:
            return img, torch.from_numpy(np.array(0))
        else:
            lbl_int = lbl_dict[self.img_path[index].split('/')[-2]]
            return img, torch.from_numpy(np.array(lbl_int))
    
    def __len__(self):
        return len(self.img_path)