import random
import torchvision.utils as tvutils
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import cv2
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose,
    RandomCrop, Normalize, Resize
)

import torch
class ImageDataset_Train():
    def __init__(self, data_list, norm_train, abnorm_train, is_train=True):
        """
        Args:
            data_list (list): Containing [image path, grading]
        """
        self.data_list = data_list
        self.norm_train, self.abnorm_train = norm_train, abnorm_train

        self.is_train = is_train
        self.crop_path = '/home/compu/working/breast_project/dataset/ori_aug_test2/sort/'

        # IMG_SIZE = 256
        if is_train:           
            print('N_training data:', len(self.data_list))
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                ])
        else:
            print('N_validation data:', len(self.data_list))
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                ])


    def __len__(self):
        if self.is_train:
            # return 8
            return len(self.data_list) # 964
        else:
            # return 8            
            return len(self.data_list) # 108


    def __getitem__(self, idx):
        # if idx == 0:
        #     print('self.data_list[idx]:', self.data_list[idx])
        # print(idx, self.data_list[idx])
        patient_img_path = self.crop_path + str(self.data_list[idx][0])
        patient_label = int(self.data_list[idx][1])
        imgs_list = os.listdir(patient_img_path)
        imgs_list.sort()
        # print('patient_label:',patient_label)
        image0 = Image.open(patient_img_path + '/' + imgs_list[0]).convert('RGB')
        image1 = Image.open(patient_img_path + '/' + imgs_list[1]).convert('RGB')
        image2 = Image.open(patient_img_path + '/' + imgs_list[2]).convert('RGB')
        image3 = Image.open(patient_img_path + '/' + imgs_list[3]).convert('RGB')
        
        if self.transform:
            image0 = self.transform(image0)
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            image3 = self.transform(image3)
        
        label = torch.tensor(patient_label).unsqueeze(-1).long()

        if patient_label == 0:
            random_index = random.randint(0, len(self.abnorm_train)-1)
            negative_img_path = self.crop_path + str(self.abnorm_train[random_index][0])
            negative_label = int(self.abnorm_train[random_index][1])
        else:
            random_index = random.randint(0, len(self.norm_train)-1)
            negative_img_path = self.crop_path + str(self.norm_train[random_index][0])
            negative_label = int(self.norm_train[random_index][1])

        negative_imgs_list = os.listdir(negative_img_path)
        negative_imgs_list.sort()
        # print('patient_label:',patient_label)
        n_image0 = Image.open(negative_img_path + '/' + negative_imgs_list[0]).convert('RGB')
        n_image1 = Image.open(negative_img_path + '/' + negative_imgs_list[1]).convert('RGB')
        n_image2 = Image.open(negative_img_path + '/' + negative_imgs_list[2]).convert('RGB')
        n_image3 = Image.open(negative_img_path + '/' + negative_imgs_list[3]).convert('RGB')
        
        if self.transform:
            n_image0 = self.transform(n_image0)
            n_image1 = self.transform(n_image1)
            n_image2 = self.transform(n_image2)
            n_image3 = self.transform(n_image3)

        if patient_label == 0:
            random_index = random.randint(0, len(self.norm_train) - 1)
            pos_img_path = self.crop_path + str(self.norm_train[random_index][0])
            pos_label = int(self.norm_train[random_index][1])
        else:
            random_index = random.randint(0, len(self.abnorm_train) - 1)
            pos_img_path = self.crop_path + str(self.abnorm_train[random_index][0])
            pos_label = int(self.norm_train[random_index][1])

        pos_imgs_list = os.listdir(pos_img_path)
        pos_imgs_list.sort()
        # print('patient_label:',patient_label)
        p_image0 = Image.open(pos_img_path + '/' + pos_imgs_list[0]).convert('RGB')
        p_image1 = Image.open(pos_img_path + '/' + pos_imgs_list[1]).convert('RGB')
        p_image2 = Image.open(pos_img_path + '/' + pos_imgs_list[2]).convert('RGB')
        p_image3 = Image.open(pos_img_path + '/' + pos_imgs_list[3]).convert('RGB')

        if self.transform:
            p_image0 = self.transform(p_image0)
            p_image1 = self.transform(p_image1)
            p_image2 = self.transform(p_image2)
            p_image3 = self.transform(p_image3)


        return [image0, image1, image2, image3], label, [n_image0, n_image1, n_image2, n_image3], negative_label, [p_image0, p_image1, p_image2, p_image3], pos_label

class ImageDataset():
    def __init__(self, data_list, is_train=True):
        """
        Args:
            data_list (list): Containing [image path, grading]
        """
        self.data_list = data_list
        self.is_train = is_train

        self.crop_path = '/home/compu/working/breast_project/dataset/ori_aug_test2/sort/'

        # IMG_SIZE = 256
        if is_train:           
            print('N_training data:', len(self.data_list))
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                ])
        else:
            print('N_validation data:', len(self.data_list))
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                ])


    def __len__(self):
        if self.is_train:
            # return 8
            return len(self.data_list)
        else:
            # return 8            
            return len(self.data_list)


    def __getitem__(self, idx):
        # if idx == 0:
        #     print('self.data_list[idx]:', self.data_list[idx])
        # print(idx, self.data_list[idx])
        patient_img_path = self.crop_path + str(self.data_list[idx][0])
        patient_label = int(self.data_list[idx][1])
        imgs_list = os.listdir(patient_img_path)
        imgs_list.sort()
        # print('patient_label:',patient_label)
        image0 = Image.open(patient_img_path + '/' + imgs_list[0]).convert('RGB')
        image1 = Image.open(patient_img_path + '/' + imgs_list[1]).convert('RGB')
        image2 = Image.open(patient_img_path + '/' + imgs_list[2]).convert('RGB')
        image3 = Image.open(patient_img_path + '/' + imgs_list[3]).convert('RGB')
        
        if self.transform:
            image0 = self.transform(image0)
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            image3 = self.transform(image3)
        
        id = str(self.data_list[idx][0])
        label = torch.tensor(patient_label).unsqueeze(-1).long()

        return [image0, image1, image2, image3], label, id
