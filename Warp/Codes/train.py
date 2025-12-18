import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from network import build_model, Network
from dataset import TrainDataset
import glob
from loss import cal_lp_loss, inter_grid_loss, intra_grid_loss
from cross_and_swin import SimpleCrossAttention, compute_similarity_matrix
from dataset import SceneDataset
from torchvision.models.swin_transformer import swin_t, Swin_T_Weights
import torch.nn.functional as F

last_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
# path to save the summary files
SUMMARY_DIR = os.path.join(last_path, 'summary')
writer = SummaryWriter(log_dir=SUMMARY_DIR)
# path to save the model files
# MODEL_DIR = os.path.join(last_path, 'model')
MODEL_DIR = os.path.join(last_path, 'stiched_model')
Stiched_DIR = os.path.join(last_path, 'stiched_model')
# create folders if it dose not exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)


def find_best_pairs_greedy(sim_matrix, threshold):
    """
    sim_matrix: torch.Tensor (N,N), symmetric, diagonal 0
    greedy: pick largest sim (i,j) where i and j unused, until none > threshold.
    returns list of (i,j)
    """
    sim = sim_matrix.clone()
    N = sim.shape[0]
    pairs = []
    used = torch.zeros(N, dtype=torch.bool, device=sim.device)

    # make sure lower triangle doesn't duplicate
    sim = sim.triu(diagonal=1)

    while True:
        val, idx = torch.max(sim.view(-1), dim=0)
        max_val = val.item()
        if max_val <= threshold:
            break
        idx = idx.item()
        i = idx // N
        j = idx % N
        if used[i] or used[j]:
            # zero out that entry and continue
            sim[i, j] = -1e9
            continue
        pairs.append((i, j))
        used[i] = True
        used[j] = True
        # invalidate row/col
        sim[i, :] = -1e9
        sim[:, i] = -1e9
        sim[j, :] = -1e9
        sim[:, j] = -1e9
    return pairs
def scaled_resize_pad(image):
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            image = np.transpose(image, (2, 0, 1))  # HWC → CHW
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image) #從numpy 轉成 tensor
        image = image.unsqueeze(0)  # [1,C,H,W] 為了符合後續interpolate的操作 tensor shape必須有batch 維度
    image = image.unsqueeze(0)
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
def train(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # define dataset
    # train_data = TrainDataset(data_path=args.train_path)
    train_data = SceneDataset(data_path=args.train_path)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)
    swin_encoder = swin_t(weights=Swin_T_Weights.DEFAULT).features
    swin_encoder = swin_encoder.eval().cuda()
    # define the network
    net = Network()
    cross_attention_module = SimpleCrossAttention(dim=16)
    if torch.cuda.is_available():
        net = net.cuda()
        cross_attention_module = cross_attention_module.cuda()

    # define the optimizer and learning rate
    # optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)  # default as 0.0001
    optimizer = optim.Adam(cross_attention_module.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)  # default as 0.0001
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    #load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)

        net.load_state_dict(checkpoint['model'])
        if 'cross_attention_module' in checkpoint:
            cross_attention_module.load_state_dict(checkpoint['cross_attention_module'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        glob_iter = checkpoint['glob_iter']
        scheduler.last_epoch = start_epoch
        print('load model from {}!'.format(model_path))
    else:
        start_epoch = 0
        glob_iter = 0
        print('training from stratch!')

    for param in net.parameters():
        param.requires_grad = False
    for param in swin_encoder.parameters():
        param.requires_grad = False
    print("##################start training#######################")
    score_print_fre = 50
    for epoch in range(start_epoch, args.max_epoch):

        print("start epoch {}".format(epoch))
        net.train()
        loss_sigma = 0.0
        overlap_loss_sigma = 0.
        nonoverlap_loss_sigma = 0.
        print(epoch, 'lr={:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        # print("hi")
        for batch_idx, batch_value in enumerate(train_loader):
            threshold = 0.7
            print(f"batch_value.shape = {batch_value.shape}")
            batch_value = batch_value[0].cuda() 
            images = []
            for img in batch_value:
                img,_ = scaled_resize_pad(img)
                images.append(img)
            batch_value = torch.stack(images)
            print(f"batch_value.shape = {batch_value.shape}")
            sim_scores = compute_similarity_matrix(batch_value,cross_attention_module,swin_encoder,is_train=True)
            min_val = sim_scores.min()
            max_val = sim_scores.max()
            sim_scores = (sim_scores - min_val) / (max_val - min_val + 1e-8)
            pairs = find_best_pairs_greedy(sim_scores, threshold)
            # print(sim_scores)
            overlap_loss = torch.tensor(0.0, device='cuda')
            nonoverlap_loss = torch.tensor(0.0, device='cuda')
            sim_loss = torch.tensor(0.0, device='cuda')
            for idx, (idx1, idx2) in enumerate(pairs):

                inpu1_tesnor = batch_value[idx1].unsqueeze(0) 
                inpu2_tesnor = batch_value[idx2].unsqueeze(0)

                if torch.cuda.is_available():
                    inpu1_tesnor = inpu1_tesnor.cuda()
                    inpu2_tesnor = inpu2_tesnor.cuda()

                # forward, backward, update weights
                optimizer.zero_grad()
                batch_out = build_model(net, inpu1_tesnor, inpu2_tesnor)
                # result
                output_H = batch_out['output_H']
                output_H_inv = batch_out['output_H_inv']
                warp_mesh = batch_out['warp_mesh']
                warp_mesh_mask = batch_out['warp_mesh_mask']
                mesh1 = batch_out['mesh1']
                mesh2 = batch_out['mesh2']
                overlap = batch_out['overlap']

                # calculate loss for overlapping regions
                overlap_loss += cal_lp_loss(inpu1_tesnor, inpu2_tesnor, output_H, output_H_inv, warp_mesh, warp_mesh_mask)
                # calculate loss for non-overlapping regions
                nonoverlap_loss += 10*inter_grid_loss(overlap, mesh2) + 10*intra_grid_loss(mesh2)
                
                sim_loss += 1 - sim_scores[idx1,idx2]  # 越相似越小
            
            if len(pairs):
                overlap_loss = overlap_loss/len(pairs)
                nonoverlap_loss = nonoverlap_loss/len(pairs)
                sim_loss = sim_loss / len(pairs)
                total_loss = overlap_loss + nonoverlap_loss + sim_loss
                total_loss.backward()
            else:
                continue
            # clip the gradient
            # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            torch.nn.utils.clip_grad_norm_(cross_attention_module.parameters(), max_norm=3, norm_type=2)
            optimizer.step()

            overlap_loss_sigma += overlap_loss.item()
            nonoverlap_loss_sigma += nonoverlap_loss.item()
            loss_sigma += total_loss.item()

            # print(glob_iter)

            # record loss and images in tensorboard
            if batch_idx % score_print_fre == 0 and batch_idx != 0:
                average_loss = loss_sigma / score_print_fre
                average_overlap_loss = overlap_loss_sigma/ score_print_fre
                average_nonoverlap_loss = nonoverlap_loss_sigma/ score_print_fre
                print(f"average_loss = {average_loss}")
                print(f"average_overlap_loss = {average_overlap_loss}")
                print(f"average_nonoverlap_loss = {average_nonoverlap_loss}")
                loss_sigma = 0.0
                overlap_loss_sigma = 0.
                nonoverlap_loss_sigma = 0.

                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}]/[{:0>3}] Total Loss: {:.4f}  Overlap Loss: {:.4f}  Non-overlap Loss: {:.4f} lr={:.8f}".format(epoch + 1, args.max_epoch, batch_idx + 1, len(train_loader),
                                        average_loss, average_overlap_loss, average_nonoverlap_loss, optimizer.state_dict()['param_groups'][0]['lr']))
                # visualization
                writer.add_image("inpu1", (inpu1_tesnor[0]+1.)/2., glob_iter)
                writer.add_image("inpu2", (inpu2_tesnor[0]+1.)/2., glob_iter)
                writer.add_image("warp_H", (output_H[0,0:3,:,:]+1.)/2., glob_iter)
                writer.add_image("warp_mesh", (warp_mesh[0]+1.)/2., glob_iter)
                writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], glob_iter)
                writer.add_scalar('total loss', average_loss, glob_iter)
                writer.add_scalar('overlap loss', average_overlap_loss, glob_iter)
                writer.add_scalar('nonoverlap loss', average_nonoverlap_loss, glob_iter)

            glob_iter += 1


        scheduler.step()
            # save model
            
        if ((epoch+1) % 10 == 0 or (epoch+1)==args.max_epoch):
            
            filename ='epoch' + str(epoch+1).zfill(3) + '_model.pth'
            # model_save_path = os.path.join(MODEL_DIR, filename)
            model_save_path = os.path.join(Stiched_DIR, filename)
            state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(),'cross_attention_module': cross_attention_module.state_dict(), 'epoch': epoch+1, "glob_iter": glob_iter}
            torch.save(state, model_save_path)
    print("##################end training#######################")


if __name__=="__main__":


    print('<==================== setting arguments ===================>\n')

    # create the argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=600)
    parser.add_argument('--train_path', type=str, default=r'D:\UDIS2\sorted_image')

    # parse the arguments
    args = parser.parse_args()
    print(args)

    # train
    train(args)


