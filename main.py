import argparse
import glob
import os
import shutil
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from Warp.Codes.dataset import SceneDataset
from Warp.Codes.network import Network as Wnet, build_output_model as wout_model
from Warp.Codes.cross_and_swin import ResNet50Encoder
from Composition.Codes.network import Network as Cnet, build_model as cout_model
from torchvision.models.swin_transformer import swin_t, Swin_T_Weights
# last_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
# MODEL_DIR = os.path.join(last_path, 'model')
WARP_CKPT_PATH = r'D:\UDIS2\Warp\stiched_model\epoch400_model.pth'
COMPOSE_CKPT_PATH = r'D:\UDIS2\Composition\model\epoch050_model.pth'
SIM_CKPT_PATH = r"D:\UDIS2\sim_model_para\epoch600_model.pth"

def find_best_pairs_greedy(sim_matrix, threshold):
    """
    Greedy 選 pair，batch 固定為 1。
    sim_matrix: Tensor of shape (1, N, N)
    threshold: float, 相似度門檻
    Returns:
        pairs_tensor: Tensor of shape (num_pairs, 2)
    """
    sim = sim_matrix.clone()
    B, N, _ = sim.shape
    assert B == 1, "This version only supports batch=1"

    sim = sim[0].triu(diagonal=1)  # 只取上三角避免重複
    used = torch.zeros(N, dtype=torch.bool, device=sim.device)
    pairs = []

    while True:
        val, idx = torch.max(sim.view(-1), dim=0)
        max_val = val.item()
        if max_val <= threshold:
            break

        i = idx.item() // N
        j = idx.item() % N

        if used[i] or used[j]:
            sim[i, j] = -1e9
            continue

        pairs.append((i, j))
        used[i] = True
        used[j] = True

        # 無效化 row/col
        sim[i, :] = -1e9
        sim[:, i] = -1e9
        sim[j, :] = -1e9
        sim[:, j] = -1e9

    return pairs
def clear_temp_dirs(processed_dir):
    temp_dirs = ['warp1', 'warp2', 'mask1', 'mask2', 'ave_fusion', 'learned_mask1', 'learned_mask2', 'final_mask']
    for d in temp_dirs:
        dir_path = os.path.join(processed_dir, d)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)  # 刪除資料夾
            # print("已刪除資料夾!")
        os.makedirs(dir_path)  # 重建空資料夾
def scaled_resize_pad(image):
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            image = np.transpose(image, (2, 0, 1))  # HWC → CHW
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image) #從numpy 轉成 tensor
        image = image.unsqueeze(0)  # [1,C,H,W] 為了符合後續interpolate的操作 tensor shape必須有batch 維度
    
    _,_,h,w = image.shape
    scale = 512/max(h,w)
    new_h = int(h*scale)
    new_w = int(w*scale)
    
    pad_h = 512 - new_h
    pad_w = 512 - new_w

    # 上下左右平均補，不讓影像偏移
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    if image.shape[1] == 3:
        resized_image = F.interpolate(image, size=(new_h, new_w), mode="bilinear")
        padded_image = F.pad(resized_image, (pad_left, pad_right, pad_top, pad_bottom), value=-1.0)
    if image.shape[1] == 1:
        resized_image = F.interpolate(image, size=(new_h, new_w), mode="nearest")
        padded_image = F.pad(resized_image, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    padded_image = padded_image.squeeze(0)
    return padded_image, scale

def load_images_as_batch(image_dir):
    # print(f"image_dir = {image_dir}")
    image_paths = sorted([
        os.path.join(image_dir, f) 
        for f in os.listdir(image_dir) 
        if f.lower().endswith(('.jpg', '.png'))
    ])

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {image_dir}")

    images = []
    print(image_paths)
    for img_path in image_paths:
        img = cv2.imread(img_path)#dtype = numpy.ndarray shape = (h,w,3)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        img = img.astype(np.float32)
        img = (img / 127.5) - 1.0
        img,_= scaled_resize_pad(img)
        images.append(img)

    batch_tensor = torch.stack(images).unsqueeze(0)  # [N, C, H, W]
    return batch_tensor
def load_masks_as_batch(mask_dir):
    # print(f"mask_dir = {mask_dir}")
    mask_paths = sorted([
        os.path.join(mask_dir, f) 
        for f in os.listdir(mask_dir) 
        if f.lower().endswith(('.jpg', '.png'))
    ])

    if len(mask_paths) == 0:
        raise FileNotFoundError(f"No masks found in {mask_dir}")

    masks = []
    # print(mask_paths)
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 單通道
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")
        # mask = cv2.resize(mask, (width, height))  # resize
        mask = mask.astype(np.float32) / 255.0    # normalize to [0,1]
        mask, _ = scaled_resize_pad(mask)
        masks.append(mask)

    batch_tensor = torch.stack(masks)  # [N, 1, H, W]
    return batch_tensor
def crop_non_black_region(img, encoder, mask):
    """
    從只有黑區背景的拼接結果中裁出非黑色矩形區域
    img: numpy 影像 (H, W, 3)
    """

    # 找出非黑區的座標
    mask = np.squeeze(mask, axis=0)
    print(f"img.shape = {img.shape}, mask.shape = {mask.shape}")
    # mask = np.any(mask != 0, axis=-1)
    print(mask)
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        print("⚠️ 沒有非黑區域，無法裁切。")
        return img

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    print(f"y_min = {y_min}, x_min = {x_min}")
    print(f"y_max = {y_max}, x_max = {x_max}")
    cropped = img[y_min:y_max+1, x_min:x_max+1]
    print(f"img.shape = {img.shape}")
    return cropped
def compute_similarity_matrix(images, encoder, mask):
    images = images.to('cuda')
    B,N,C,H,W = images.shape
    if N < 2:
        print("只剩一張圖，結束！")
        return None
    images = images.view(B*N,C,H,W)
    print(f"images.shape = {images.shape}")
    
    with torch.no_grad():
        features = encoder(images,mask)
        print(f"features.shape = {features.shape}")
    
#=================cosin_similarity========================
    global_sim = features @ features.T
    global_sim.fill_diagonal_(0)    
    print(f"global_sim = {global_sim}")
    idx_i, idx_j = torch.triu_indices(N, N, offset=1)
    sims = global_sim[idx_i, idx_j]
    sorted_idx = torch.argsort(sims, descending=True)
    # print(sorted_idx)
    used = set()
    pairs = []

    for k in sorted_idx:
        sim_value = sims[k].item()
        if sim_value < 0.85:     
            continue
        i = idx_i[k].item()
        j = idx_j[k].item()

        if i in used or j in used:
            continue

        pairs.append((i, j))
        used.add(i)
        used.add(j)

        # 若全部配好就停止
        if len(used) == N:
            break
    return pairs
class UDIS2_multi_stitching:
    def __init__(self, embed_dim, warp_model_path, comp_model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # 1. 初始化模型
        self.feature_encoder = ResNet50Encoder(embed_dim=embed_dim).to(self.device)
        self.warp_net = self._load_warp_model(warp_model_path)
        self.comp_net = self._load_comp_model(comp_model_path)

        self.feature_encoder.eval() # 預設推理模式
        self.warp_net.eval()
        self.comp_net.eval()

    def _load_warp_model(self, path):
        # 這裡實例化 UDIS2 的 Warp 網路結構並載入權重
        model = Wnet().to(self.device)
        model.load_state_dict(torch.load(path)['model'])
        print('load warp model from {}!'.format(path))
        return model 

    def _load_comp_model(self, path):
        # 這裡實例化 UDIS2 的 Composition 網路結構
        model = Cnet().to(self.device)
        model.load_state_dict(torch.load(path)['model'])
        print('load composition model from {}!'.format(path))
        return model
def process_scene(processed_dir, batch_value, cross_attention_module, encoder, wnet, cnet, scene_num, threshold=0.7):
    # 初始化：清空半處理資料夾
    print(f"batch_value.shape = {batch_value.shape}")
    first_round = True
    round_idx = 0
    masks = None
    while True:
        round_idx += 1
        threshold = 0.3 + 0.1 * (round_idx - 1)
        threshold = min(threshold, 0.95)
        path_processed = os.path.join(processed_dir, scene_num)
        print(path_processed)
        print(f"第{round_idx}輪處理! Threshold = {threshold}")
        if first_round:
            with torch.no_grad():
                pairs = compute_similarity_matrix(batch_value, encoder, mask=None)  # shape: (N, N)
            first_round =False
        else:
            batch_value = load_images_as_batch(data_path)
            masks = load_masks_as_batch(mask_path)
            print(f"masks.shape = {masks.shape}")
            with torch.no_grad():
                pairs = compute_similarity_matrix(batch_value, encoder, mask=masks )  # shape: (N, N)
        print(f"pairs:{pairs}")
        if pairs == None:
            print(f"處理完畢，無更多配對")
            break
        # sim_scores = (sim_scores + 1.0) / 2.0  
        # pairs = find_best_pairs_greedy(sim_scores, threshold)
        # print(pairs)
        clear_temp_dirs(path_processed)
        print(f"資料夾清理完畢!")
        batch_value = batch_value.squeeze(0)
        for idx, (idx1, idx2) in enumerate(pairs):
#==========================================Warp================================================================
            if masks != None:
                feature_mask1 = masks[idx1].unsqueeze(0)   # [1, 1, H, W]
                feature_mask2 = masks[idx2].unsqueeze(0)   # [1, 1, H, W]    
                if torch.cuda.is_available():
                    feature_mask1 = feature_mask1.cuda()
                    feature_mask2 = feature_mask2.cuda()
            else:
                feature_mask1 = None
                feature_mask2 = None
                
            path_warp1 = os.path.join(path_processed, 'warp1')
            path_warp2 = os.path.join(path_processed, 'warp2')
            path_mask1 = os.path.join(path_processed, 'mask1')
            path_mask2 = os.path.join(path_processed, 'mask2')
            path_learned_mask1 = os.path.join(path_processed, 'learned_mask1')
            path_learned_mask2 = os.path.join(path_processed, 'learned_mask2')
            path_final_mask = os.path.join(path_processed, 'final_mask')
            inpu1_tesnor = batch_value[idx1].unsqueeze(0) 
            inpu2_tesnor = batch_value[idx2].unsqueeze(0)
            if torch.cuda.is_available():
                inpu1_tesnor = inpu1_tesnor.cuda()
                inpu2_tesnor = inpu2_tesnor.cuda()
            with torch.no_grad():
                batch_out = wout_model(wnet, inpu1_tesnor, inpu2_tesnor, feature_mask1, feature_mask2)
            final_warp1 = batch_out['final_warp1']
            final_warp1_mask = batch_out['final_warp1_mask']
            final_warp2 = batch_out['final_warp2']
            final_warp2_mask = batch_out['final_warp2_mask']

            final_warp1 = ((final_warp1[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
            final_warp2 = ((final_warp2[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
            final_warp1_mask = final_warp1_mask[0].cpu().detach().numpy().transpose(1,2,0)
            final_warp2_mask = final_warp2_mask[0].cpu().detach().numpy().transpose(1,2,0)

            filename = str(idx+1).zfill(6) + ".jpg"
            # print(os.path.join(path_warp1, filename))
            cv2.imwrite(os.path.join(path_warp1, filename), final_warp1)
            cv2.imwrite(os.path.join(path_warp2, filename), final_warp2)
            cv2.imwrite(os.path.join(path_mask1, filename), final_warp1_mask * 255)
            cv2.imwrite(os.path.join(path_mask2, filename), final_warp2_mask * 255)
#===============================================================================================================            
#=================================================Composition===================================================
            warp1 = cv2.imread(os.path.join(path_warp1, filename))
            warp1 = warp1.astype(dtype=np.float32)
            warp1 = (warp1 / 127.5) - 1.0
            warp1 = np.transpose(warp1, [2, 0, 1])

            # load image2
            warp2 = cv2.imread(os.path.join(path_warp2, filename))
            warp2 = warp2.astype(dtype=np.float32)
            warp2 = (warp2 / 127.5) - 1.0
            warp2 = np.transpose(warp2, [2, 0, 1])

            # load mask1
            mask1 = cv2.imread(os.path.join(path_mask1, filename))
            mask1 = mask1.astype(dtype=np.float32)
            mask1 = np.expand_dims(mask1[:,:,0], 2) / 255
            mask1 = np.transpose(mask1, [2, 0, 1])

            # load mask2
            mask2 = cv2.imread(os.path.join(path_mask2, filename))
            mask2 = mask2.astype(dtype=np.float32)
            mask2 = np.expand_dims(mask2[:,:,0], 2) / 255
            mask2 = np.transpose(mask2, [2, 0, 1])

            # convert to tensor
            warp1_tensor = torch.tensor(warp1).unsqueeze(0)
            warp2_tensor = torch.tensor(warp2).unsqueeze(0)
            mask1_tensor = torch.tensor(mask1).unsqueeze(0)
            mask2_tensor = torch.tensor(mask2).unsqueeze(0)

            if torch.cuda.is_available():
                warp1_tensor = warp1_tensor.cuda()
                warp2_tensor = warp2_tensor.cuda()
                mask1_tensor = mask1_tensor.cuda()
                mask2_tensor = mask2_tensor.cuda()
                
            with torch.no_grad():
                batch_out = cout_model(cnet, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)

            stitched_image = batch_out['stitched_image']
            learned_mask1 = batch_out['learned_mask1']
            learned_mask2 = batch_out['learned_mask2']

            stitched_image = ((stitched_image[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
            learned_mask1 = (learned_mask1[0]*255).cpu().detach().numpy()
            learned_mask2 = (learned_mask2[0]*255).cpu().detach().numpy()    

            mask1_bool = learned_mask1 > 0
            mask2_bool = learned_mask2 > 0
            
            geo_mask = (~mask1_bool) & (~mask2_bool)
            gray = stitched_image.mean(axis=2)
            content_mask = gray < 3

            final_mask = (geo_mask | content_mask).astype(np.float32)*255
            # final_mask = geo_mask.astype(np.float32)*255

            # final_mask = np.logical_and(learned_mask1 == 0, learned_mask2 == 0).astype(np.float32)*255
            final_mask = 255-final_mask
            # stitched_image = crop_non_black_region(stitched_image,final_mask)

            learned_mask1 = learned_mask1.transpose(1,2,0)
            learned_mask2 = learned_mask2.transpose(1,2,0)
            final_mask = final_mask.transpose(1,2,0)

            path = str(idx+1).zfill(6) + ".jpg"
            ave_path = os.path.join(path_processed, 'ave_fusion')
            ave_path = os.path.join(ave_path, path)
            path_learned_mask1 = os.path.join(path_learned_mask1, path)
            path_learned_mask2 = os.path.join(path_learned_mask2, path)
            path_final_mask = os.path.join(path_final_mask, path)
            print(f"composition_save_path : {ave_path}")
            cv2.imwrite(ave_path, stitched_image)
            cv2.imwrite(path_learned_mask1, learned_mask1)
            cv2.imwrite(path_learned_mask2, learned_mask2)
            cv2.imwrite(path_final_mask, final_mask)
        
        data_path = os.path.join(path_processed, 'ave_fusion')
        mask_path = os.path.join(path_processed, 'final_mask')
        torch.cuda.empty_cache()
        # break
    

def prepare_folders(semi_processed_path, scene_path):
    os.makedirs(semi_processed_path, exist_ok=True)
    sub_scenes = [d for d in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, d))]
    sub_dirs = [
        'warp1', 'warp2', 'mask1', 'mask2', 
        'ave_fusion', 'learned_mask1', 'learned_mask2', 'final_mask'
    ]
    for item in sub_scenes:
        # 建立場景根目錄 (例如: semi_processed/scene_001)
        processed_subdir = os.path.join(semi_processed_path, item)
        
        for sub_dir in sub_dirs:
            target_path = os.path.join(processed_subdir, sub_dir)
            # exist_ok=True 代表資料夾若已存在，不會噴錯，也不需要額外寫 if check
            os.makedirs(target_path, exist_ok=True)
            
    print(f"成功初始化 {len(sub_scenes)} 個場景的儲存結構。")


def test(args):
    test_data = SceneDataset(data_path=args.test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)

    # define the network
    wnet = Wnet()#build_model(args.model_name)
    cnet = Cnet()
    encoder = ResNet50Encoder()
    abc = abc 

    if torch.cuda.is_available():
        wnet = wnet.cuda()
        cnet = cnet.cuda()
        encoder.cuda()
    #load the existing models if it exists
    if len(WARP_CKPT_PATH) != 0:
        ckpt = torch.load(WARP_CKPT_PATH)
        wnet.load_state_dict(ckpt['model'])
        print('load warp model from {}!'.format(WARP_CKPT_PATH))
    else:
        print('No checkpoint found!')
    if len(COMPOSE_CKPT_PATH) != 0:
        ckpt = torch.load(COMPOSE_CKPT_PATH)
        cnet.load_state_dict(ckpt['model'])
        print('load composition model from {}!'.format(COMPOSE_CKPT_PATH))
    else:
        print('No checkpoint found!')
    # if len(SIM_CKPT_PATH) != 0:
    #     ckpt = torch.load(SIM_CKPT_PATH)
    #     cross_attention_module.load_state_dict(ckpt['cross_attention_module'])
    #     # encoder.load_state_dict(ckpt['encoder'])
    #     print('load sim model from {}!'.format(SIM_CKPT_PATH))
    # else:
    #     print('No checkpoint found!')
    print("##################start testing#######################")
    root_path = args.root_path
    scene_path = args.test_path
    semi_processed_path = os.path.join(root_path, 'semi-processed')
    if not os.path.exists(semi_processed_path):
        os.makedirs(semi_processed_path, exist_ok=True)

    # 找 scene_path 底下所有資料夾（假設每個資料夾是不同場景）
    for item in os.listdir(scene_path):
        item_path = os.path.join(scene_path, item)
        if os.path.isdir(item_path):
            # 在 processed_dir 裡面建立同名子資料夾
            processed_subdir = os.path.join(semi_processed_root, item)
            if not os.path.exists(processed_subdir):
                os.makedirs(processed_subdir)
            # print(f"建立暫存資料夾：{processed_subdir}")
            path_warp1 = os.path.join(processed_subdir, 'warp1')
            if not os.path.exists(path_warp1):
                os.makedirs(path_warp1)
            path_warp2 = os.path.join(processed_subdir, 'warp2')
            if not os.path.exists(path_warp2):
                os.makedirs(path_warp2)
            path_mask1 = os.path.join(processed_subdir, 'mask1')
            if not os.path.exists(path_mask1):
                os.makedirs(path_mask1)
            path_mask2 = os.path.join(processed_subdir, 'mask2')
            if not os.path.exists(path_mask2):
                os.makedirs(path_mask2)
            ave_fusion = os.path.join(processed_subdir, 'ave_fusion')
            if not os.path.exists(ave_fusion):
                os.makedirs(ave_fusion)
            path_learned_mask1 = os.path.join(processed_subdir, 'learned_mask1')
            if not os.path.exists(path_learned_mask1):
                os.makedirs(path_learned_mask1)
            path_learned_mask2 = os.path.join(processed_subdir, 'learned_mask2')
            if not os.path.exists(path_learned_mask2):
                os.makedirs(path_learned_mask2)
            path_final_mask = os.path.join(processed_subdir, 'final_mask')
            if not os.path.exists(path_final_mask):
                os.makedirs(path_final_mask)


    wnet.eval()
    cnet.eval()
    encoder = encoder.eval()
    for batch_idx, batch_value in enumerate(test_loader):
        scene_idx = batch_idx +1
        threshold = 0.7
        batch_value = batch_value.cuda() 
        print(f"i = {batch_idx} , batach_value.shape = {batch_value.shape}")
        scene_num = 'scene' + str(scene_idx).zfill(4)
        print(f"開始處理場景{scene_num}")
        process_scene(semi_processed_root, 
                    batch_value = batch_value,
                    cross_attention_module = cross_attention_module,
                    encoder = encoder,
                    wnet = wnet,
                    cnet = cnet,
                    scene_num = scene_num,
                    threshold = threshold)
        print(f"場景{scene_idx}處理完畢!")
        # break
        # 取得最相似的兩張圖片 idx1, idx2
        # idx1, idx2 = select_best_pair(sim_scores)
        # inpu1_tesnor = batch_value[idx1].unsqueeze(0) 
        # inpu2_tesnor = batch_value[idx2].unsqueeze(0) 
        # print(f"inpu1_tensor = {inpu1_tesnor.shape}, inpu2_tensor = {inpu2_tesnor.shape}")

        # sim_scores = compute_similarity_matrix(batch_value,cross_attention_module,swin_encoder)  # shape: (N, N)
          # 視情況調整
        # pairs_to_process = []

        # num_images = sim_scores.shape[0]
        # for i in range(num_images):
        #     for j in range(i+1, num_images):
        #         if sim_scores[i, j] > threshold:
        #             pairs_to_process.append((i, j))

        

        # if torch.cuda.is_available():
        #     inpu1_tesnor = inpu1_tesnor.cuda()
        #     inpu2_tesnor = inpu2_tesnor.cuda()

        # with torch.no_grad():
        #     batch_out = build_output_model(wnet, inpu1_tesnor, inpu2_tesnor)

        # final_warp1 = batch_out['final_warp1']
        # final_warp1_mask = batch_out['final_warp1_mask']
        # final_warp2 = batch_out['final_warp2']
        # final_warp2_mask = batch_out['final_warp2_mask']
        # final_mesh1 = batch_out['mesh1']
        # final_mesh2 = batch_out['mesh2']


        # final_warp1 = ((final_warp1[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        # final_warp2 = ((final_warp2[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        # final_warp1_mask = final_warp1_mask[0].cpu().detach().numpy().transpose(1,2,0)
        # final_warp2_mask = final_warp2_mask[0].cpu().detach().numpy().transpose(1,2,0)
        # final_mesh1 = final_mesh1[0].cpu().detach().numpy()
        # final_mesh2 = final_mesh2[0].cpu().detach().numpy()


        # filename = str(i+1).zfill(6) + ".jpg"
        # ave_fusion = final_warp1 * (final_warp1/ (final_warp1+final_warp2+1e-6)) + final_warp2 * (final_warp2/ (final_warp1+final_warp2+1e-6))
        # cv2.imwrite(os.path.join(path_warp1, filename), final_warp1)
        # cv2.imwrite(os.path.join(path_warp2, filename), final_warp2)
        # cv2.imwrite(os.path.join(path_mask1, filename), final_warp1_mask * 255)
        # cv2.imwrite(os.path.join(path_mask2, filename), final_warp2_mask * 255)
        # cv2.imwrite(os.path.join(path_ave_fusion, filename), ave_fusion)

        # path = path_warp1 + str(i+1).zfill(6) + ".jpg"
        # cv2.imwrite(path, final_warp1)
        # path = path_warp2 + str(i+1).zfill(6) + ".jpg"
        # cv2.imwrite(path, final_warp2)
        # path = path_mask1 + str(i+1).zfill(6) + ".jpg"
        # cv2.imwrite(path, final_warp1_mask*255)
        # path = path_mask2 + str(i+1).zfill(6) + ".jpg"
        # cv2.imwrite(path, final_warp2_mask*255)

        # ave_fusion = final_warp1 * (final_warp1/ (final_warp1+final_warp2+1e-6)) + final_warp2 * (final_warp2/ (final_warp1+final_warp2+1e-6))
        # path = path_ave_fusion + str(i+1).zfill(6) + ".jpg"
        # cv2.imwrite(path, ave_fusion)

        # print('i = {}'.format( i+1))

        torch.cuda.empty_cache()




    print("##################end testing#######################")
if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    # parser.add_argument('--test_path', type=str, default='/opt/data/private/nl/Data/UDIS-D/testing/')
    parser.add_argument('--root_path', type=str, default=r"D:\UDIS2")
    parser.add_argument('--test_path', type=str, default=r'D:\UDIS2\sorted_image_testing_set')  
    # parser.add_argument('--cross_attention_dim', type=int, default=256)  
    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)



