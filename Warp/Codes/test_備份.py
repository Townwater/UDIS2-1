# coding: utf-8
import gc
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
from network import build_model, Network
from dataset import *
import os
import numpy as np
import skimage
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import cv2
# print(torch.cuda.is_available())  # 應該是 True
# print(torch.version.cuda)         # 應該顯示 CUDA 版本
# print(skimage.__version__)
import torch
# print("Torch Version:", torch.__version__)
# print("CUDA Version (linked):", torch.version.cuda)
# print("CUDA Available:", torch.cuda.is_available())
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


def test(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # dataset
    test_data = TestDataset(data_path=args.test_path)
    # test_data = SceneDataset(data_path=args.test_path)
    print(args.test_path)
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
    for i, batch_value in enumerate(test_loader):
        print(f"batch_value.shape = {batch_value.shape}")
        inpu1_tesnor = batch_value[0].float()
        inpu2_tesnor = batch_value[1].float()
        print(f"inpu1_tensor = {inpu1_tesnor.shape}")
        if torch.cuda.is_available():
            inpu1_tesnor = inpu1_tesnor.cuda()
            inpu2_tesnor = inpu2_tesnor.cuda()
            print(f"inpu1_tensor = {inpu1_tesnor.shape}, inpu2_tensor = {inpu2_tesnor.shape}")
            with torch.no_grad():
                batch_out = build_model(net, inpu1_tesnor, inpu2_tesnor, is_training=False)

            warp_mesh_mask = batch_out['warp_mesh_mask']
            warp_mesh = batch_out['warp_mesh']


            warp_mesh_np = ((warp_mesh[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
            warp_mesh_mask_np = warp_mesh_mask[0].cpu().detach().numpy().transpose(1,2,0)
            inpu1_np = ((inpu1_tesnor[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
            
            psnr = peak_signal_noise_ratio(inpu1_np * warp_mesh_mask_np,
                               warp_mesh_np * warp_mesh_mask_np,
                               data_range=255)

            ssim = structural_similarity(inpu1_np * warp_mesh_mask_np,
                                        warp_mesh_np * warp_mesh_mask_np,
                                        data_range=255,
                                        channel_axis=-1,
                                         win_size=3)
            


            print('i = {}, psnr = {:.6f}'.format( i+1, psnr))

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            del batch_out, warp_mesh_mask, warp_mesh
            del inpu1_tesnor, inpu2_tesnor
            gc.collect()
            torch.cuda.empty_cache()

    print("=================== Analysis ==================")
    print("psnr")
    psnr_list.sort(reverse = True)
    psnr_list_30 = psnr_list[0 : 331]
    psnr_list_60 = psnr_list[331: 663]
    psnr_list_100 = psnr_list[663: -1]
    print("top 30%", np.mean(psnr_list_30))
    print("top 30~60%", np.mean(psnr_list_60))
    print("top 60~100%", np.mean(psnr_list_100))
    print('average psnr:', np.mean(psnr_list))

    ssim_list.sort(reverse = True)
    ssim_list_30 = ssim_list[0 : 331]
    ssim_list_60 = ssim_list[331: 663]
    ssim_list_100 = ssim_list[663: -1]
    print("top 30%", np.mean(ssim_list_30))
    print("top 30~60%", np.mean(ssim_list_60))
    print("top 60~100%", np.mean(ssim_list_100))
    print('average ssim:', np.mean(ssim_list))
    print("##################end testing#######################")


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    # parser.add_argument('--test_path', type=str, default='/opt/data/private/nl/Data/UDIS-D/testing/')
    parser.add_argument('--test_path', type=str, default=r'd:\UDIS2\testing')
    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)
