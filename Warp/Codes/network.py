import torch
import torch.nn as nn
from Warp.Codes.utils import torch_DLT,torch_homo_transform,torch_tps_transform
from Warp.Codes import grid_res
import ssl
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.models as models

import torchvision.transforms as T
resize_512 = T.Resize((512,512))

grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W


# draw mesh on image
# warp: h*w*3
# f_local: grid_h*grid_w*2
def safe_inverse(tensor):
    try:
        return torch.inverse(tensor)
    except RuntimeError as e:
        if "cusolver error: 7" in str(e):
            # print("⚠️ GPU torch.inverse 失敗，改用 CPU 版本")
            return torch.inverse(tensor.cpu()).to(tensor.device)
        else:
            raise e
def draw_mesh_on_warp(warp, f_local):

    warp = np.ascontiguousarray(warp)

    point_color = (0, 255, 0) # BGR
    thickness = 2
    lineType = 8

    num = 1
    for i in range(grid_h+1):
        for j in range(grid_w+1):

            num = num + 1
            if j == grid_w and i == grid_h:
                continue
            elif j == grid_w:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
            elif i == grid_h:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)
            else :
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)

    return warp


#Covert global homo into mesh
def H2Mesh(H, rigid_mesh):

    H_inv = safe_inverse(H)
    ori_pt = rigid_mesh.reshape(rigid_mesh.size()[0], -1, 2)
    ones = torch.ones(rigid_mesh.size()[0], (grid_h+1)*(grid_w+1),1)
    if torch.cuda.is_available():
        ori_pt = ori_pt.cuda()
        ones = ones.cuda()

    ori_pt = torch.cat((ori_pt, ones), 2) # bs*(grid_h+1)*(grid_w+1)*3
    tar_pt = torch.matmul(H_inv, ori_pt.permute(0,2,1)) # bs*3*(grid_h+1)*(grid_w+1)

    mesh_x = torch.unsqueeze(tar_pt[:,0,:]/tar_pt[:,2,:], 2)
    mesh_y = torch.unsqueeze(tar_pt[:,1,:]/tar_pt[:,2,:], 2)
    mesh = torch.cat((mesh_x, mesh_y), 2).reshape([rigid_mesh.size()[0], grid_h+1, grid_w+1, 2])

    return mesh

# get rigid mesh
def get_rigid_mesh(batch_size, height, width):

    ww = torch.matmul(torch.ones([grid_h+1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w+1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h+1), 1), torch.ones([1, grid_w+1]))
    if torch.cuda.is_available():
        ww = ww.cuda()
        hh = hh.cuda()

    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)),2) # (grid_h+1)*(grid_w+1)*2
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return ori_pt

# normalize mesh from -1 ~ 1
def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = torch.stack([mesh_w, mesh_h], 3) # bs*(grid_h+1)*(grid_w+1)*2

    return norm_mesh.reshape([batch_size, -1, 2]) # bs*-1*2



# random augmentation
# it seems to do nothing to the performance
def data_aug(img1, img2):
    # Randomly shift brightness
    random_brightness = torch.randn(1).uniform_(0.7,1.3).cuda()
    img1_aug = img1 * random_brightness
    random_brightness = torch.randn(1).uniform_(0.7,1.3).cuda()
    img2_aug = img2 * random_brightness

    # Randomly shift color
    white = torch.ones([img1.size()[0], img1.size()[2], img1.size()[3]]).cuda()
    random_colors = torch.randn(3).uniform_(0.7,1.3).cuda()
    color_image = torch.stack([white * random_colors[i] for i in range(3)], axis=1)
    img1_aug  *= color_image

    random_colors = torch.randn(3).uniform_(0.7,1.3).cuda()
    color_image = torch.stack([white * random_colors[i] for i in range(3)], axis=1)
    img2_aug  *= color_image

    # clip
    img1_aug = torch.clamp(img1_aug, -1, 1)
    img2_aug = torch.clamp(img2_aug, -1, 1)

    return img1_aug, img2_aug

def scaled_resize_pad(image):
    _,_,h,w = image.shape
    scale = 512/max(h,w)
    new_h = int(h*scale)
    new_w = int(w*scale)
    resized_image = F.interpolate(image, size=(new_h, new_w), mode="bilinear")

    pad_h = 512 - new_h
    pad_w = 512 - new_w

    # 上下左右平均補，不讓影像偏移
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded_image = F.pad(resized_image, (pad_left, pad_right, pad_top, pad_bottom))
    return padded_image, scale
# for train.py / test.py
def build_model(net, input1_tensor, input2_tensor, is_training = True):
    batch_size, _, img_h, img_w = input1_tensor.size()

    # network
    if is_training == True:
        aug_input1_tensor, aug_input2_tensor = data_aug(input1_tensor, input2_tensor)
        H_motion, mesh_motion = net(aug_input1_tensor, aug_input2_tensor)
    else:
        H_motion, mesh_motion = net(input1_tensor, input2_tensor)

    H_motion = H_motion.reshape(-1, 4, 2)
    mesh_motion = mesh_motion.reshape(-1, grid_h+1, grid_w+1, 2)

    # initialize the source points bs x 4 x 2
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
    if torch.cuda.is_available():
        src_p = src_p.cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    # target points
    dst_p = src_p + H_motion
    # solve homo using DLT
    H = torch_DLT.tensor_DLT(src_p, dst_p)

    M_tensor = torch.tensor([[img_w / 2.0, 0., img_w / 2.0],
                      [0., img_h / 2.0, img_h / 2.0],
                      [0., 0., 1.]])

    if torch.cuda.is_available():
        M_tensor = M_tensor.cuda()

    M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
    M_tensor_inv = safe_inverse(M_tensor)
    M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)
    H_mat = torch.matmul(torch.matmul(M_tile_inv, H), M_tile)

    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        mask = mask.cuda()
    output_H = torch_homo_transform.transformer(torch.cat((input2_tensor, mask), 1), H_mat, (img_h, img_w))

    H_inv_mat = torch.matmul(torch.matmul(M_tile_inv, safe_inverse(H)), M_tile)
    
    
    output_H_inv = torch_homo_transform.transformer(torch.cat((input1_tensor, mask), 1), H_inv_mat, (img_h, img_w))

    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh = H2Mesh(H, rigid_mesh)
    mesh = ini_mesh + mesh_motion


    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh, img_h, img_w)

    output_tps = torch_tps_transform.transformer(torch.cat((input2_tensor, mask), 1), norm_mesh, norm_rigid_mesh, (img_h, img_w))
    warp_mesh = output_tps[:,0:3,...]
    warp_mesh_mask = output_tps[:,3:6,...]

    # calculate the overlapping regions to apply shape-preserving constraints
    overlap = torch_tps_transform.transformer(warp_mesh_mask, norm_rigid_mesh, norm_mesh, (img_h, img_w))
    overlap = overlap.permute(0, 2, 3, 1).unfold(1, int(img_h/grid_h), int(img_h/grid_h)).unfold(2, int(img_w/grid_w), int(img_w/grid_w))
    overlap = torch.mean(overlap.reshape(batch_size, grid_h, grid_w, -1), 3)
    overlap_one = torch.ones_like(overlap)
    overlap_zero = torch.zeros_like(overlap)
    overlap = torch.where(overlap<0.9, overlap_one, overlap_zero)


    out_dict = {}
    out_dict.update(output_H=output_H, output_H_inv = output_H_inv, warp_mesh = warp_mesh, warp_mesh_mask = warp_mesh_mask, mesh1 = rigid_mesh, mesh2 = mesh, overlap = overlap)


    return out_dict

# for train_ft.py
def build_new_ft_model(net, input1_tensor, input2_tensor):
    batch_size, _, img_h, img_w = input1_tensor.size()

    H_motion, mesh_motion = net(input1_tensor, input2_tensor)

    H_motion = H_motion.reshape(-1, 4, 2)
    #H_motion = torch.stack([H_motion[...,0]*img_w/512, H_motion[...,1]*img_h/512], 2)

    mesh_motion = mesh_motion.reshape(-1, grid_h+1, grid_w+1, 2)
    #mesh_motion = torch.stack([mesh_motion[...,0]*img_w/512, mesh_motion[...,1]*img_h/512], 3)

    # initialize the source points bs x 4 x 2
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
    if torch.cuda.is_available():
        src_p = src_p.cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    # target points
    dst_p = src_p + H_motion
    # solve homo using DLT
    H = torch_DLT.tensor_DLT(src_p, dst_p)


    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh = H2Mesh(H, rigid_mesh)
    mesh = ini_mesh + mesh_motion

    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh, img_h, img_w)

    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        mask = mask.cuda()
    output_tps = torch_tps_transform.transformer(torch.cat((input2_tensor, mask), 1), norm_mesh, norm_rigid_mesh, (img_h, img_w))
    warp_mesh = output_tps[:,0:3,...]
    warp_mesh_mask = output_tps[:,3:6,...]


    out_dict = {}
    out_dict.update(warp_mesh = warp_mesh, warp_mesh_mask = warp_mesh_mask, rigid_mesh = rigid_mesh, mesh = mesh)


    return out_dict

# for train_ft.py
def get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh):
    batch_size, _, img_h, img_w = input1_tensor.size()

    rigid_mesh = torch.stack([rigid_mesh[...,0]*img_w/512, rigid_mesh[...,1]*img_h/512], 3)
    mesh = torch.stack([mesh[...,0]*img_w/512, mesh[...,1]*img_h/512], 3)

    ######################################
    width_max = torch.max(mesh[...,0])
    width_max = torch.maximum(torch.tensor(img_w).cuda(), width_max)
    width_min = torch.min(mesh[...,0])
    width_min = torch.minimum(torch.tensor(0).cuda(), width_min)
    height_max = torch.max(mesh[...,1])
    height_max = torch.maximum(torch.tensor(img_h).cuda(), height_max)
    height_min = torch.min(mesh[...,1])
    height_min = torch.minimum(torch.tensor(0).cuda(), height_min)

    out_width = width_max - width_min
    out_height = height_max - height_min
    print(out_width)
    print(out_height)

    warp1 = torch.zeros([batch_size, 3, out_height.int(), out_width.int()]).cuda()
    warp1[:,:, int(torch.abs(height_min)):int(torch.abs(height_min))+img_h,  int(torch.abs(width_min)):int(torch.abs(width_min))+img_w] = (input1_tensor+1)*127.5

    mask1 = torch.zeros([batch_size, 3, out_height.int(), out_width.int()]).cuda()
    mask1[:,:, int(torch.abs(height_min)):int(torch.abs(height_min))+img_h,  int(torch.abs(width_min)):int(torch.abs(width_min))+img_w] = 255

    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        mask = mask.cuda()

    # get warped img2
    mesh_trans = torch.stack([mesh[...,0]-width_min, mesh[...,1]-height_min], 3)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh_trans, out_height, out_width)

    stitch_tps_out = torch_tps_transform.transformer(torch.cat([input2_tensor+1, mask], 1), norm_mesh, norm_rigid_mesh, (out_height.int(), out_width.int()))
    warp2 = stitch_tps_out[:,0:3,:,:]*127.5
    mask2 = stitch_tps_out[:,3:6,:,:]*255

    stitched = warp1*(warp1/(warp1+warp2+1e-6)) + warp2*(warp2/(warp1+warp2+1e-6))

    stitched_mesh = draw_mesh_on_warp(stitched[0].cpu().detach().numpy().transpose(1,2,0), mesh_trans[0].cpu().detach().numpy())

    out_dict = {}
    out_dict.update(warp1 = warp1, mask1 = mask1, warp2 = warp2, mask2 = mask2, stitched = stitched, stitched_mesh = stitched_mesh)

    return out_dict


# for test_output.py
def build_output_model(net, input1_tensor, input2_tensor, mask1 = None, mask2 = None):
    batch_size, _, img_h, img_w = input1_tensor.size()
    H_motion, mesh_motion = net(input1_tensor, input2_tensor, mask1, mask2)

    
    H_motion = H_motion.reshape(-1, 4, 2)
    H_motion = torch.stack([H_motion[...,0]*img_w/512, H_motion[...,1]*img_h/512], 2)
    mesh_motion = mesh_motion.reshape(-1, grid_h+1, grid_w+1, 2)
    mesh_motion = torch.stack([mesh_motion[...,0]*img_w/512, mesh_motion[...,1]*img_h/512], 3)

    # initialize the source points bs x 4 x 2
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
    if torch.cuda.is_available():
        src_p = src_p.cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    # target points
    dst_p = src_p + H_motion
    # solve homo using DLT
    H = torch_DLT.tensor_DLT(src_p, dst_p)


    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh = H2Mesh(H, rigid_mesh)
    mesh = ini_mesh + mesh_motion

    width_max = torch.max(mesh[...,0])
    width_max = torch.maximum(torch.tensor(img_w).cuda(), width_max)
    width_min = torch.min(mesh[...,0])
    width_min = torch.minimum(torch.tensor(0).cuda(), width_min)
    height_max = torch.max(mesh[...,1])
    height_max = torch.maximum(torch.tensor(img_h).cuda(), height_max)
    height_min = torch.min(mesh[...,1])
    height_min = torch.minimum(torch.tensor(0).cuda(), height_min)

    out_width = width_max - width_min
    out_height = height_max - height_min

    # get warped img1
    M_tensor = torch.tensor([[out_width / 2.0, 0., out_width / 2.0],
                      [0., out_height / 2.0, out_height / 2.0],
                      [0., 0., 1.]])
    N_tensor = torch.tensor([[img_w / 2.0, 0., img_w / 2.0],
                      [0., img_h / 2.0, img_h / 2.0],
                      [0., 0., 1.]])
    
    if torch.cuda.is_available():
        M_tensor = M_tensor.cuda()
        N_tensor = N_tensor.cuda()
    N_tensor_inv = torch.inverse(N_tensor)

    I_ = torch.tensor([[1., 0., width_min],
                      [0., 1., height_min],
                      [0., 0., 1.]])#.unsqueeze(0)
    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        I_ = I_.cuda()
        mask = mask.cuda()
    I_mat = torch.matmul(torch.matmul(N_tensor_inv, I_), M_tensor).unsqueeze(0)
    # print(f"out_height = {out_height}, out_width = {out_width}")
    homo_output = torch_homo_transform.transformer(torch.cat((input1_tensor+1, mask), 1), I_mat, (out_height.int(), out_width.int()))

    torch.cuda.empty_cache()
    # get warped img2
    mesh_trans = torch.stack([mesh[...,0]-width_min, mesh[...,1]-height_min], 3)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh_trans, out_height, out_width)
    tps_output = torch_tps_transform.transformer(torch.cat([input2_tensor+1, mask],1), norm_mesh, norm_rigid_mesh, (out_height.int(), out_width.int()))


    out_dict = {}
    out_dict.update(final_warp1=homo_output[:, 0:3, ...]-1, final_warp1_mask = homo_output[:, 3:6, ...], final_warp2=tps_output[:, 0:3, ...]-1, final_warp2_mask = tps_output[:, 3:6, ...], mesh1=rigid_mesh, mesh2=mesh_trans)

    return out_dict


# define and forward
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.regressNet1_part1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.regressNet1_part2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=4096, out_features=1024, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=8, bias=True)
        )


        self.regressNet2_part1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.regressNet2_part2 = nn.Sequential(
            nn.Linear(in_features=8192, out_features=4096, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=4096, out_features=2048, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=2048, out_features=(grid_w+1)*(grid_h+1)*2, bias=True)

        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        ssl._create_default_https_context = ssl._create_unverified_context
        resnet50_model = models.resnet.resnet50(pretrained=True)

        if torch.cuda.is_available():
            resnet50_model = resnet50_model.cuda()
        self.feature_extractor_stage1, self.feature_extractor_stage2 = self.get_res50_FeatureMap(resnet50_model)
        #-----------------------------------------

    def get_res50_FeatureMap(self, resnet50_model):

        layers_list = []

        layers_list.append(resnet50_model.conv1)
        layers_list.append(resnet50_model.bn1)
        layers_list.append(resnet50_model.relu)
        layers_list.append(resnet50_model.maxpool)
        layers_list.append(resnet50_model.layer1)
        layers_list.append(resnet50_model.layer2)

        feature_extractor_stage1 = nn.Sequential(*layers_list)

        feature_extractor_stage2 = nn.Sequential(resnet50_model.layer3)

        #layers_list.append(resnet50_model.layer3)

        return feature_extractor_stage1, feature_extractor_stage2

    # forward
    def forward(self, input1_tesnor, input2_tesnor, mask1 = None, mask2 = None):
        batch_size, _, img_h, img_w = input1_tesnor.size()

        feature_1_64 = self.feature_extractor_stage1(input1_tesnor)
        feature_1_32 = self.feature_extractor_stage2(feature_1_64)
        feature_2_64 = self.feature_extractor_stage1(input2_tesnor)
        feature_2_32 = self.feature_extractor_stage2(feature_2_64)
        
        if mask1 is not None and mask2 is not None:
            # print(f"before interpolate mask2 = {mask2.shape}")
            mask_1_64 = F.interpolate(mask1, size=feature_1_64.shape[-2:], mode="nearest")
            mask_1_32 = F.interpolate(mask1, size=feature_1_32.shape[-2:], mode="nearest")
            mask_2_64 = F.interpolate(mask2, size=feature_2_64.shape[-2:], mode="nearest")
            mask_2_32 = F.interpolate(mask2, size=feature_2_32.shape[-2:], mode="nearest")
            # print(f"after interpolate mask2_64 = {mask_2_64.shape}")
            # print(feature_1_64.device)
            # print(mask_1_64.device)
            # feature_1_64 = feature_1_64 * mask_1_64
            # feature_1_32 = feature_1_32 * mask_1_32
            # feature_2_64 = feature_2_64 * mask_2_64
            # feature_2_32 = feature_2_32 * mask_2_32
        else:
            mask_1_64 = None
            mask_1_32 = None
            mask_2_64 = None
            mask_2_32 = None
        ######### stage 1
        # correlation_32 = self.CCL(feature_1_32, feature_2_32)
        correlation_32 = self.Multi_scale_CCL(feature_1_32, feature_2_32, kernel=[3,5,7], mask_1 = mask_1_32, mask_2 = mask_2_32)
        # print(f"correlation_32.shape = {correlation_32.shape}")
        # print(f"correlation_32 max = {correlation_32.max().item()}, min = {correlation_32.min().item()}")
        temp_1 = self.regressNet1_part1(correlation_32)
        temp_1 = temp_1.view(temp_1.size()[0], -1)
        offset_1 = self.regressNet1_part2(temp_1)
        H_motion_1 = offset_1.reshape(-1, 4, 2)
        # print(f"H_motion_1 = {H_motion_1.max().item()}, min = {H_motion_1.min().item()}")

        src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
        if torch.cuda.is_available():
            src_p = src_p.cuda()
        src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
        dst_p = src_p + H_motion_1
        H = torch_DLT.tensor_DLT(src_p/8, dst_p/8)

        M_tensor = torch.tensor([[img_w/8 / 2.0, 0., img_w/8 / 2.0],
                      [0., img_h/8 / 2.0, img_h/8 / 2.0],
                      [0., 0., 1.]])

        if torch.cuda.is_available():
            M_tensor = M_tensor.cuda()

        M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        M_tensor_inv = safe_inverse(M_tensor)
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)
        H_mat = torch.matmul(torch.matmul(M_tile_inv, H), M_tile)

        warp_feature_2_64 = torch_homo_transform.transformer(feature_2_64, H_mat, (int(img_h/8), int(img_w/8)))

        ######### stage 2
        # correlation_64 = self.CCL(feature_1_64, warp_feature_2_64)
        correlation_64 = self.Multi_scale_CCL(feature_1_64, warp_feature_2_64,kernel=[3,5,7], mask_1 = mask_1_64, mask_2 = mask_2_64)
        temp_2 = self.regressNet2_part1(correlation_64)
        temp_2 = temp_2.view(temp_2.size()[0], -1)
        offset_2 = self.regressNet2_part2(temp_2)


        return offset_1, offset_2


    def extract_patches(self, x, kernel=3, stride=1):
        if kernel != 1:
            padding = kernel // 2
            x = nn.ZeroPad2d(padding)(x)
        x = x.permute(0, 2, 3, 1)
        all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
        return all_patches


    def CCL(self, feature_1, feature_2):
        bs, c, h, w = feature_1.size()

        norm_feature_1 = F.normalize(feature_1, p=2, dim=1)
        norm_feature_2 = F.normalize(feature_2, p=2, dim=1)
        # print(f"norm_feature_2.size() = {norm_feature_2.size()}")

        patches = self.extract_patches(norm_feature_2)
        if torch.cuda.is_available():
            patches = patches.cuda()

        matching_filters  = patches.reshape((patches.size()[0], -1, patches.size()[3], patches.size()[4], patches.size()[5]))

        match_vol = []
        for i in range(bs):
            single_match = F.conv2d(norm_feature_1[i].unsqueeze(0), matching_filters[i], padding=1)
            # print(f"single_match.shape = {single_match.shape}")
            match_vol.append(single_match)
        # print(f"pre-cat match_vol.shape = {match_vol.size()}")
        match_vol = torch.cat(match_vol, 0)
        # print(f"post-cat match_vol.shape = {match_vol.size()}")
        # print(f"match_vol.size() = {match_vol .size()}")

        # scale softmax
        softmax_scale = 10
        match_vol = F.softmax(match_vol*softmax_scale,1)

        channel = match_vol.size()[1]

        h_one = torch.linspace(0, h-1, h)
        one1w = torch.ones(1, w)
        if torch.cuda.is_available():
            h_one = h_one.cuda()
            one1w = one1w.cuda()
        h_one = torch.matmul(h_one.unsqueeze(1), one1w)
        h_one = h_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)

        w_one = torch.linspace(0, w-1, w)
        oneh1 = torch.ones(h, 1)
        if torch.cuda.is_available():
            w_one = w_one.cuda()
            oneh1 = oneh1.cuda()
        w_one = torch.matmul(oneh1, w_one.unsqueeze(0))
        w_one = w_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)

        c_one = torch.linspace(0, channel-1, channel)
        if torch.cuda.is_available():
            c_one = c_one.cuda()
        c_one = c_one.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(bs, -1, h, w)

        flow_h = match_vol*(c_one//w - h_one)
        flow_h = torch.sum(flow_h, dim=1, keepdim=True)
        flow_w = match_vol*(c_one%w - w_one)
        flow_w = torch.sum(flow_w, dim=1, keepdim=True)

        feature_flow = torch.cat([flow_w, flow_h], 1)
        #print(flow.size())

        return feature_flow
    def Multi_scale_CCL(self, feature_1, feature_2, kernel, mask_1 = None, mask_2 = None):
        bs, c, h, w = feature_1.size()

        norm_feature_1 = F.normalize(feature_1, p=2, dim=1)
        norm_feature_2 = F.normalize(feature_2, p=2, dim=1)
        #print(norm_feature_2.size())
        scaled_patches = []
        for ks in kernel:
            patches = self.extract_patches(norm_feature_2, ks)
            if torch.cuda.is_available():
                patches = patches.cuda()
            scaled_patches.append(patches)

        matching_filters = []
        for patches in scaled_patches:
            matching_filters.append(patches.reshape((patches.size()[0], -1, patches.size()[3], patches.size()[4], patches.size()[5]))) 

        match_vol = [[] for _ in kernel]
        for i in range(bs):
            for idx, (filters, ks) in enumerate(zip(matching_filters, kernel)):
                padding = ks // 2
                # print(f"norm_feature_1.shape = {norm_feature_1.shape}, filter.shape = {filters.shape}")
                # print(f"filter max = {filters[i].max().item()}, filter min = {filters[i].min().item()}, filter mean = {filters[i].mean().item()}, filter std = {filters[i].std().item()}")
                single_match = F.conv2d(norm_feature_1[i].unsqueeze(0), filters[i], padding=padding)
                match_vol[idx].append(single_match)
                # print(f"single_match.shape = {single_match.shape}")
        #print(match_vol .size())
        softmax_scale = 10
        # scale softmax
        feature_flow = []
        for mv in match_vol:
            mv = torch.cat(mv, 0)
            # print(f"mv.shape = {mv.shape}")
            # print("mv pre-softmax: mean {:.3f}, std {:.3f}, max {:.3f}, min {:.3f}".format(mv.mean().item(), mv.std().item(), mv.max().item(), mv.min().item()))
            if mask_1 is not None and mask_2 is not None:
                # print(f"mask.shape = {mask_2.shape}")
                mask2_exp = mask_2.expand_as(mv)  # expand 成 [bs, channel, h, w]
                mv = mv.masked_fill(mask2_exp == 0, -1e-9)
            mv = F.softmax(mv*softmax_scale,1)
            # print("mv post-softmax: maxprob {:.3f}, meanmax {:.3f}".format(mv.max(dim=1)[0].mean().item(), mv.max().mean().item()))
            channel = mv.size(1)

            h_one = torch.linspace(0, h-1, h)
            one1w = torch.ones(1, w)
            if torch.cuda.is_available():
                h_one = h_one.cuda()
                one1w = one1w.cuda()
            h_one = torch.matmul(h_one.unsqueeze(1), one1w)#太強拉 巧妙利用unsqueeze(1)讓原本h_one.shape = (h)變成(h,1)使其可以跟one1w.shape = (1,w)相乘
            h_one = h_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)#將座標擴展成與multi_match_vol的tensor一樣的格式

            w_one = torch.linspace(0, w-1, w)
            oneh1 = torch.ones(h, 1)
            if torch.cuda.is_available():
                w_one = w_one.cuda()
                oneh1 = oneh1.cuda()
            w_one = torch.matmul(oneh1, w_one.unsqueeze(0))#與上面同理 只是變成w_one.shape = (1,w) 所以可以(h,1)乘於(1,w)
            w_one = w_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)

            c_one = torch.linspace(0, channel-1, channel)
            if torch.cuda.is_available():
                c_one = c_one.cuda()
            c_one = c_one.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(bs, -1, h, w)

            flow_h = mv*(c_one//w - h_one)
            flow_h = torch.sum(flow_h, dim=1, keepdim=True)
            flow_w = mv*(c_one%w - w_one)
            flow_w = torch.sum(flow_w, dim=1, keepdim=True)

            feature_flow.append(torch.cat([flow_w, flow_h], 1)) 
            # print("hello")
        #print(flow.size())
        w_i = torch.tensor([0.5, 0.3, 0.2], device=feature_flow[0].device)
        total_flow = sum(wi * ff for wi, ff in zip(w_i, feature_flow))

        # total_flow = sum(feature_flow)
        return total_flow