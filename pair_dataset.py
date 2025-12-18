import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
import numpy as np

class PairDataset(Dataset):
    def __init__(self, csv_path, train_path, width=512, height=512):
        """
        csv_path: CSV 檔路徑，包含 columns: image1, image2, label
        train_path: 影像根目錄
        width, height: resize 尺寸
        """
        self.train_path = train_path
        self.width = width
        self.height = height
        self.df = pd.read_csv(csv_path)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 讀圖並轉成 float32
        img1_path = os.path.join(self.train_path, row['image1'])
        img2_path = os.path.join(self.train_path, row['image2'])
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            raise FileNotFoundError(f"Cannot load image: {img1_path} or {img2_path}")

        # BGR -> RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # resize
        img1 = cv2.resize(img1, (self.width, self.height))
        img2 = cv2.resize(img2, (self.width, self.height))

        # float32 並 normalize [-1,1]
        img1 = img1.astype(np.float32) / 127.5 - 1.0
        img2 = img2.astype(np.float32) / 127.5 - 1.0

        # HWC -> CHW
        img1 = np.transpose(img1, (2,0,1))
        img2 = np.transpose(img2, (2,0,1))

        # 轉成 Tensor
        img1_tensor = torch.tensor(img1, dtype=torch.float32)
        img2_tensor = torch.tensor(img2, dtype=torch.float32)
        label_tensor = torch.tensor(row['label'], dtype=torch.float32)

        return img1_tensor, img2_tensor, label_tensor