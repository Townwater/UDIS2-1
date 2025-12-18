# coding: utf-8
import gc
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
from network import build_model, Network
from dataset import *
import sys
import os
import numpy as np
import skimage
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import cv2
from cross_and_swin import SimpleCrossAttention,compute_similarity_matrix
from torchvision.models.swin_transformer import swin_t, Swin_T_Weights
import torch
last_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
# print(last_path)
MODEL_DIR = os.path.join(last_path, 'model')
# print(MODEL_DIR)
def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.5)
    return


def select_best_pair(sim_matrix):
    N = sim_matrix.shape[0]
    # 創建遮罩：對角線設為極小值（避免選到自己）
    masked_sim = sim_matrix.clone()
    masked_sim.fill_diagonal_(-float('inf'))

    # 找最大值的索引
    max_idx = torch.argmax(masked_sim)
    i = max_idx // N
    j = max_idx % N
    return int(i), int(j)

def test(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # dataset
    # test_data = TestDataset(data_path=args.test_path)
    test_data = SceneDataset(data_path=args.test_path)
    # print(args.test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)

    # define the network
    net = Network()#build_model(args.model_name)
    if torch.cuda.is_available():
        net = net.cuda()

    #load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model'])
        print('load model from {}!'.format(model_path))
    else:
        print('No checkpoint found!')



    print("##################start testing#######################")
    psnr_list = []
    ssim_list = []
    net.eval()
    cross_attention_module = SimpleCrossAttention(dim=16).cuda()
    cross_attention_module.eval()
    swin_encoder = swin_t(weights=Swin_T_Weights.DEFAULT).features
    swin_encoder = swin_encoder.eval().cuda()
    # print(swin_encoder)
    for i, batch_value in enumerate(test_loader):
        # batch_value = batch_value.squeeze(0).cuda()
        # inpu1_tesnor = batch_value[0].float()
        # inpu2_tesnor = batch_value[1].float()
        batch_value = batch_value[0].cuda() 
        # === 相似度計算（例如用 Cross Attention 或 Swin Transformer encoder） ===
        print(f"i = {i} , batach_value.shape = {batch_value.shape}")
        sim_scores = compute_similarity_matrix(batch_value,cross_attention_module,swin_encoder)  # shape: (N, N)
        
        # 取得最相似的兩張圖片 idx1, idx2
        idx1, idx2 = select_best_pair(sim_scores)
        # batch_value = batch_value.squeeze(0).cuda()
        inpu1_tesnor = batch_value[idx1].unsqueeze(0) 
        inpu2_tesnor = batch_value[idx2].unsqueeze(0) 
        # batch_value = batch_value.unsqueeze(0).cuda()
        print(f"inpu1_tensor = {inpu1_tesnor.shape}, inpu2_tensor = {inpu2_tesnor.shape}")

        if torch.cuda.is_available():
            inpu1_tesnor = inpu1_tesnor.cuda()
            inpu2_tesnor = inpu2_tesnor.cuda()

            with torch.no_grad():
                
                batch_out = build_model(net, inpu1_tesnor, inpu2_tesnor, is_training=False)

            warp_mesh_mask = batch_out['warp_mesh_mask']
            warp_mesh = batch_out['warp_mesh']


            warp_mesh_np = ((warp_mesh[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
            warp_mesh_mask_np = warp_mesh_mask[0].cpu().detach().numpy().transpose(1,2,0)
            inpu1_np = ((inpu1_tesnor[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)

            del batch_out, warp_mesh_mask, warp_mesh
            del inpu1_tesnor, inpu2_tesnor
            gc.collect()
            torch.cuda.empty_cache()

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    # parser.add_argument('--test_path', type=str, default='/opt/data/private/nl/Data/UDIS-D/testing/')
    parser.add_argument('--test_path', type=str, default=r'D:\UDIS2\sorted_image')  
    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)
