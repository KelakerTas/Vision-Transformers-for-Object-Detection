import torch
import cv2
import os
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'

print("Cuda Avaliable :", torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class CaltechDataset (torch.utils.data.Dataset):
    def __init__(self, root, resize_dim, train = False):
       
        self.root = root
        
        if train:
            self.imgs = sorted(os.listdir(os.path.join(root, "101_ObjectCategories/airplanes")))[0:640]
            self.annotations = sorted(os.listdir(os.path.join(root, "Annotations/Airplanes_Side_2")))[0:640]
        else:
            self.imgs = sorted(os.listdir(os.path.join(root, "101_ObjectCategories/airplanes")))[640:]
            self.annotations = sorted(os.listdir(os.path.join(root, "Annotations/Airplanes_Side_2")))[640:]
       
        self.dim = resize_dim
       
    def __getitem__(self, idx):
       
        image_path = os.path.join(self.root, "101_ObjectCategories/airplanes/", self.imgs[idx])
        annotation_path = os.path.join(self.root, "Annotations/Airplanes_Side_2/", self.annotations[idx])
       
        img = cv2.imread(image_path)
       
        h, w, _ = img.shape
       
        img = cv2.resize(img, (self.dim, self.dim), interpolation = cv2.INTER_AREA)
       
        annot = scipy.io.loadmat(annotation_path)["box_coord"][0]
       
        top_left_x, top_left_y = annot[2], annot[0]
        bottom_rigth_x, bottom_right_y = annot[3], annot[1]
       
        target = (float(top_left_x)/w, float(top_left_y)/h, float(bottom_rigth_x)/w, float(bottom_right_y)/h)
        
        target = torch.as_tensor(target)
        target = target.to(device)
        #target = torch.as_tensor(target).to(device)
        img = torch.as_tensor(img)
        img = img.to(device)
       
        return img, target
       
    def __len__(self):
        return len(self.imgs)