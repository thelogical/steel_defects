from torchvision import transforms
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import torch

class Dataset:
    def __init__(self,proj_path,batch_size,gpu=False):
        if not gpu:
            self.device = 'cpu'
        else:
            self.device = 'cuda'
        self.im_path = proj_path+'/train'
        self.train_labels_path = proj_path+'/train_names.txt'
        self.test_labels_path = proj_path+'/test_names.txt'
        self.batch_size = batch_size
        self.trans = transforms.Compose([transforms.Grayscale(),transforms.ToTensor(),transforms.Normalize(mean=[0.344636027949075],std=[0.1961965408964816])])
        self.transf = transforms.ToTensor()
        self.start = 0
        self.end = 0
        self.proj_path = proj_path
        f = open(self.train_labels_path)
        self.train_labels = f.read().splitlines()
        f.close()
        f = open(self.test_labels_path)
        self.test_labels = f.read().splitlines()
        f.close()
        self.number_train_images = len(self.train_labels)
        self.number_test_images = len(self.test_labels)


    def get_mask(self,imdata):
        if type(imdata) == float:
            return np.zeros((256,1600)).astype('uint8')
        data = imdata.split(" ")
        pixels, offsets = data[::2], data[1::2]
        px = []
        for i in range(len(pixels)):
            off = offsets[i]
            px.append(int(pixels[i]) - 1)
            for j in range(1, int(off)):
                px.append(int(pixels[i]) + j - 1)
        Map = np.zeros(256*1600).astype('uint8')
        Map[px] = 1
        Map = Map.reshape(1600, 256).T
        return Map

    def get_masks_boxes_labels(self,labels):
        all_masks = pd.read_csv(self.proj_path+'/train.csv')
        train_data = []
        for im in labels:
            t_data = {}
            train_masks = []
            img_boxes = []
            box_lbl = []
            rows = all_masks[all_masks.Image.str.contains(im[:-1])]
            for _,img in rows.iterrows():
                mk = self.get_mask(img['pixels'])
                contours, _ = cv2.findContours(mk.copy(), 1, 1)
                for cont in contours:
                    mp = np.zeros((256,1600))
                    x, y, w, h = cv2.boundingRect(cont)
                    img_boxes.append([x,y,x+w,y+h])
                    box_lbl.append(int(img['Image'][-1]))
                    mp[x:x+w,y:y+h] = mk[x:x+w,y:y+h]
                    train_masks.append(mp)
            t_data['boxes'] = torch.tensor(img_boxes).to(self.device).float()
            t_data['labels'] = torch.tensor(box_lbl).to(self.device)
            t_data['masks'] = torch.tensor(train_masks).to(self.device)
            train_data.append(t_data)
        return train_data


    def get_next_batch(self):
        if self.start > self.number_train_images:
            batch_labels = self.train_labels[self.start-self.batch_size:]
            self.start = 0

        else:
            self.end = self.start + self.batch_size
            batch_labels = self.train_labels[self.start:self.end]
            self.start+=self.batch_size
        batch_images = {}
        batch_images['name'] = []
        batch_images['image'] = []
        for i in range(self.batch_size):
            im_data = Image.open(self.proj_path+'/train/'+batch_labels[i])
            im_data = self.transf(im_data)
            batch_images['name'].append(batch_labels[i])
            batch_images['image'].append(im_data)
        batch_images['image'] = torch.stack(batch_images['image'])
        return batch_images,self.get_masks_boxes_labels(batch_labels)

    def get_next_test_batch(self):
        if self.start > self.number_test_images:
            batch_labels = self.train_labels[self.start-self.batch_size:]
            self.start = 0

        else:
            self.end = self.start + self.batch_size
            batch_labels = self.test_labels[self.start:self.end]
            self.start+=self.batch_size
        batch_images = {}
        batch_images['name'] = []
        batch_images['image'] = []
        for i in range(self.batch_size):
            im_data = Image.open(self.proj_path+'/train/'+batch_labels[i])
            im_data = self.transf(im_data)
            batch_images['name'].append(batch_labels[i])
            batch_images['image'].append(im_data)
        batch_images['image'] = torch.stack(batch_images['image'])
        return batch_images,self.get_masks_boxes_labels(batch_labels)





