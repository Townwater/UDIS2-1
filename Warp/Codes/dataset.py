from torch.utils.data import Dataset
import  numpy as np
import cv2, torch
import os
import glob
from collections import OrderedDict
import random


class TrainDataset(Dataset):
    def __init__(self, data_path):

        self.width = 512
        self.height = 512
        self.train_path = data_path
        self.datas = OrderedDict()
        datas = glob.glob(os.path.join(self.train_path, '*'))
        
        for data in sorted(datas):
            # data_name = data.split('/')[-1]
            data_name = os.path.basename(data)
            if data_name == 'input1' or data_name == 'input2' :
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
        print(self.datas.keys())

    def __getitem__(self, index):
        
        # load image1
        input1 = cv2.imread(self.datas['input1']['image'][index])
        input1 = cv2.resize(input1, (self.width, self.height))
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        input1 = np.transpose(input1, [2, 0, 1])
        
        # load image2
        input2 = cv2.imread(self.datas['input2']['image'][index])
        input2 = cv2.resize(input2, (self.width, self.height))
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        input2 = np.transpose(input2, [2, 0, 1])
        
        # convert to tensor
        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)
        
        #print("fasdf")
        if_exchange = random.randint(0,1)
        if if_exchange == 0:
            #print(if_exchange)
            return (input1_tensor, input2_tensor)
        else:
            #print(if_exchange)
            return (input2_tensor, input1_tensor)

    def __len__(self):

        return len(self.datas['input1']['image'])

class TestDataset(Dataset):
    def __init__(self, data_path):

        self.width = 512
        self.height = 512
        self.test_path = data_path
        self.datas = OrderedDict()
        
        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            # data_name = data.split('/')[-1]
            data_name = os.path.basename(data)
            if data_name == 'input1' or data_name == 'input2' :
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
        print(self.datas.keys())

    def __getitem__(self, index):
        
        # load image1
        input1 = cv2.imread(self.datas['input1']['image'][index])
        #input1 = cv2.resize(input1, (self.width, self.height))
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        input1 = np.transpose(input1, [2, 0, 1])
        
        # load image2
        input2 = cv2.imread(self.datas['input2']['image'][index])
        #input2 = cv2.resize(input2, (self.width, self.height))
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        input2 = np.transpose(input2, [2, 0, 1])
        
        # convert to tensor
        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)

        return (input1_tensor, input2_tensor)

    def __len__(self):

        return len(self.datas['input1']['image'])


class SceneDataset(Dataset):
    def __init__(self, data_path, width=512, height=512):
        self.root_path = data_path
        self.width = width
        self.height = height
        # 一個 scene 是一個資料夾
        self.datas = OrderedDict()
        # self.scene_paths = glob.glob(os.path.join(self.root_path, '*'))
        self.scene_paths = sorted(glob.glob(os.path.join(self.root_path, '*')))
        self.scenes = []

        for scene_path in self.scene_paths:
            image_paths = sorted(glob.glob(os.path.join(scene_path, '*.jpg')))
            if len(image_paths) > 0:
                self.scenes.append(image_paths)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        image_paths = self.scenes[index]
        images = []

        for img_path in image_paths:
            img = cv2.imread(img_path)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img, (self.width, self.height))  # 可選
            img = img.astype(np.float32)/ 127.5 - 1.0
            img = np.transpose(img, (2, 0, 1))  # C, H, W
            images.append(torch.tensor(img, dtype=torch.float32))

        return torch.stack(images)  # shape: [num_images_in_scene, 3, H, W]

class FusionDataset(Dataset):
    def __init__(self, data_path):
        self.image_paths = sorted(glob.glob(os.path.join(data_path, '*.jpg')))
        self.width = 512
        self.height = 512

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = cv2.imread(img_path).astype(np.float32)
        img = (img / 127.5) - 1.0
        img = np.transpose(img, (2, 0, 1))  # C, H, W
        return torch.tensor(img), img_path