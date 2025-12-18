import argparse
import glob
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim as optim
from Warp.Codes.cross_and_swin import  SimpleCrossAttention,ResNet18Encoder
from torchvision.models.swin_transformer import swin_t, Swin_T_Weights
from pair_dataset import PairDataset
import matplotlib.pyplot as plt

class SimilarityTrainer(nn.Module):
    def __init__(self, encoder, cross_attention_module):
        super().__init__()
        self.encoder = encoder
        self.cross_attn = cross_attention_module
    def forward(self, img1, img2, is_train=True):
        feat1, _ = self.encoder(img1)
        feat2, _ = self.encoder(img2)
        feat1 = feat1.unsqueeze(1)
        feat2 = feat2.unsqueeze(1)
        # print(feat1.shape) (B, N, L, C) = (8, 1, 256,256)
        # print(f"feat1.shape={feat1.shape}")
        sim = self.cross_attn(feat1,feat2)
        sim = sim.squeeze()
        return sim
    
def contrastive_loss(sim, labels, pos_margin = 0.7,neg_margin = 0.3):
    pos_loss = labels * F.relu(pos_margin - sim)
    neg_loss = (1-labels) * F.relu(sim - neg_margin)
    loss = (pos_loss + neg_loss).mean()
    return loss
    
def train(args):
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    MODEL_DIR = r"D:\UDIS2\sim_model_para"

    train_data = PairDataset(csv_path=args.csv_path,train_path = args.train_path)
    dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=1, shuffle=True, drop_last=False)
    # swin_encoder = swin_t(weights=Swin_T_Weights.DEFAULT).features.cuda()
    
    cross_attention_module = SimpleCrossAttention(dim=args.cross_attention_dim).cuda()
    encoder = ResNet18Encoder().cuda()
    model = SimilarityTrainer(encoder=encoder,cross_attention_module=cross_attention_module).cuda()

    # criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    for param in encoder.backbone.parameters():
        param.requires_grad = False
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)
        # swin_encoder.load_state_dict(checkpoint['encoder'])  
        # encoder.load_state_dict(checkpoint['encoder'])  
        cross_attention_module.load_state_dict(checkpoint['cross_attention_module'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        glob_iter = checkpoint['glob_iter']
        scheduler.last_epoch = start_epoch
        print('load model from {}!'.format(model_path))
    else:
        start_epoch = 0
        glob_iter = 0
        print('training from stratch!')
    print("start training!")
    epoch_losses = []
    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        total_loss = 0
        for img1, img2, labels in dataloader:
            optimizer.zero_grad()
            img1 = img1.cuda()
            img2 = img2.cuda()
            labels = labels.float().cuda()
            sim = model(img1, img2)
            loss = contrastive_loss(sim, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        # sim = ((sim + 1) / 2).clamp(0,1)
        avg_loss = total_loss/len(dataloader)
        epoch_losses.append(avg_loss*100)
        print(f"labels = {labels}")
        print(f"sim = {sim}")
        print("=========================================================")
        print(f"Epoch {epoch+1}, Loss = {avg_loss:.4f}")
        print("=========================================================")
        scheduler.step()
        if ((epoch+1) % 10 == 0 or (epoch+1)==args.max_epoch):
            filename ='epoch' + str(epoch+1).zfill(3) + '_model.pth'
            model_save_path = os.path.join(MODEL_DIR, filename)
            state = {'optimizer': optimizer.state_dict(),'cross_attention_module': cross_attention_module.state_dict(), 'epoch': epoch+1, "glob_iter": glob_iter}
            torch.save(state, model_save_path)
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(epoch_losses)+1), epoch_losses, marker='o', label='Epoch Avg Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss x100')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(r"D:\UDIS2\epoch_loss.png", dpi=300)
if __name__=="__main__":


    print('<==================== setting arguments ===================>\n')

    # create the argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_epoch', type=int, default=600)
    parser.add_argument('--train_path', type=str, default=r'D:\UDIS2\sorted_image_training_set')
    parser.add_argument('--csv_path', type=str, default=r'D:\UDIS2\pairs.csv')
    parser.add_argument('--cross_attention_dim', type=int, default=256)  
    # parse the arguments
    args = parser.parse_args()
    print(args)

    # train
    train(args)