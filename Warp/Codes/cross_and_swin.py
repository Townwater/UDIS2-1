import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SimpleCrossAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(0.3))
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_q1 = nn.Linear(dim, dim, bias=False)
        self.to_k1 = nn.Linear(dim, dim, bias=False)
        self.to_v1 = nn.Linear(dim, dim, bias=False)

        self.to_q2 = nn.Linear(dim, dim, bias=False)
        self.to_k2 = nn.Linear(dim, dim, bias=False)
        self.to_v2 = nn.Linear(dim, dim, bias=False)
    def forward(self, feat1, feat2):
        B,N,L, C = feat1.shape
        # print(f"feat1.shape = {feat1.shape}")
        # print(f"feat1.mean() = {feat1.mean()}")
        Q1 = self.to_q1(feat1).view(B, N*L, self.heads, C // self.heads).permute(0,2,1,3)  # (B, heads, N*L, C_per_heads)
        K1 = self.to_k1(feat1).view(B, N*L, self.heads, C // self.heads).permute(0,2,1,3)  # (B, heads, N*L, C_per_heads)
        print(f"Q1.shape = {Q1.shape}")
        print(f"K1.shape = {K1.shape}")
        Q2 = self.to_q2(feat2).view(B, N*L, self.heads, C // self.heads).permute(0,2,1,3)  # (B, heads, N*L, C_per_heads)
        K2 = self.to_k2(feat2).view(B, N*L, self.heads, C // self.heads).permute(0,2,1,3)  # (B, heads, N*L, C_per_heads)

        context_12 = torch.matmul(Q1, K2.transpose(-2, -1)) * self.scale / self.tau   # (B, heads, N*L, N*L)
        context_12 = F.softmax(context_12, dim=-1)# (B, heads, N*L, N*L)
        print(f"mat context_12s.shape={context_12.shape}")
        context_21 = torch.matmul(Q2, K1.transpose(-2, -1)) * self.scale / self.tau   # (B, heads, N*L, N*L)
        context_21 = F.softmax(context_21, dim=-1)# (B, heads, N*L, N*L)

        context_12 = context_12.view(B, self.heads, N, L, N, L)  # (B, heads, N, L, N, L)
        context_21 = context_21.view(B, self.heads, N, L, N, L)  # (B, heads, N, L, N, L)
        print(f"view context_12s.shape={context_12.shape}")
        context_12 = context_12.amax(dim=-1)# (B, heads, N, L, N)
        context_21 = context_21.amax(dim=-1)# (B, heads, N, L, N)
        print(f"amax context_12s.shape={context_12.shape}")
        context_12 = context_12.mean(dim=3)# (B, heads, N, N)
        context_21 = context_21.mean(dim=3)# (B, heads, N, N)

        context_12 = context_12.mean(dim=1)# (B, N, N)
        context_21 = context_21.mean(dim=1)# (B, N, N)
        print(f"mean context_12s.shape={context_12.shape}")
        sim_matrix = (context_12 + context_21)/2
        return sim_matrix

class SimpleEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            # 512 → 256
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.GroupNorm(32, 32),   # GN(32 groups, 32 channels)
            nn.ReLU(),

            # 256 → 128
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(32, 64),   # GN(32 groups, 64 channels)
            nn.ReLU(),

            # 128 → 64
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(32, 128),  # GN(32 groups, 128 channels)
            nn.ReLU(),

            # 64 → 32
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(32, 256),  # GN(32 groups, 256 channels)
            nn.ReLU(),

            # 32 → 16
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.GroupNorm(32, 256),  # GN(32 groups, 256 channels)
            nn.ReLU(),
        )

        # 投影成目標維度 (embed_dim)
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x):
        x = self.encoder(x)               # (B, 256, 16, 16)
        x = x.flatten(2).transpose(1, 2)  # (B, L=256, C=256)
        x = self.proj(x)                  # (B, 256, embed_dim)
        x = F.normalize(x, dim=-1)        # L2 normalize
        return x
class ResNet18Encoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        # 載入 ImageNet Pretrained ResNet18
        resnet = models.resnet18(pretrained=True)

        # 移除最後兩層 (avgpool, fc)
        self.backbone = nn.Sequential(
            resnet.conv1,  # (B,64,H/2,W/2)
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # (B,64,H/4,W/4)

            resnet.layer1,  # (B,64,H/4,W/4)
            resnet.layer2,  # (B,128,H/8,W/8)
            resnet.layer3,  # (B,256,H/16,W/16)
            resnet.layer4,  # (B,512,H/32,W/32)
        )

        # token 維度投影 (512 -> embed_dim)
        self.token_proj = nn.Linear(512, embed_dim)

    def forward(self, x):
        """
        x: (B,3,H,W)
        return:
            tokens: (B, L, embed_dim)
            global_feat: (B, embed_dim)
        """

        feat = self.backbone(x)               # (B,512,H/32,W/32)

        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)   # (B,L,C)

        # Token 投影
        tokens = self.token_proj(tokens)           # (B,L,embed_dim)
        tokens = F.normalize(tokens, dim=-1)

        return tokens
class ResNet50Encoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        # 載入 ImageNet Pretrained 模型
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # 移除 avgpool + fc，只保留 CNN 主體
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,   # → (B, 2048, H/32, W/32)
        )

        # 將 2048-D channel 壓到 embed_dim
        self.fc = nn.Linear(2048, embed_dim)

    def forward(self, x, mask):
        # CNN 特徵
        feat = self.backbone(x)              # (B, 2048, H/32, W/32)
        _,_,h,w = feat.shape
        if mask is not None:
            mask = mask.to('cuda')
            mask_feat = F.interpolate(
                mask,   
                size=(h, w),
                mode='nearest'
            )
            feat = (feat * mask_feat).sum(dim=(2,3)) / mask_feat.sum(dim=(2,3))
        else:
            feat = F.adaptive_avg_pool2d(feat, 1)  # (B, 2048, 1, 1)
            feat = feat.view(feat.size(0), -1)     # (B, 2048)

        # 映射到 embedding
        feat = self.fc(feat)                  # (B, embed_dim)

        # Normalize 做 cosine-friendly
        feat = F.normalize(feat, dim=1)

        return feat
    
def compute_similarity_matrix(images, cross_attention_module,encoder):
    images = images.to('cuda')
    B,N,C,H,W = images.shape
    if N < 2:
        print("只剩一張圖，結束！")
        return None
    images = images.view(B*N,C,H,W)
    print(images.shape)
    with torch.no_grad():
        features = encoder(images)
    print(f"features.shape = {features.shape}")
#=================cosin_similarity========================
    global_sim = global_feat @ global_feat.T
    global_sim.fill_diagonal_(0)    
    print(f"global_sim = {global_sim}")
    idx_i, idx_j = torch.triu_indices(N, N, offset=1)
    sims = global_sim[idx_i, idx_j]
    sorted_idx = torch.argsort(sims, descending=True)
    used = set()
    pairs = []

    for k in sorted_idx:
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
    print(pairs)
#=========================================================
    _,L,C = features.shape
    features = features.view(B,N,L,C)
    print(features.shape)
    sim_matrix = torch.zeros(N, N, device=images.device)
    for (i,j) in pairs:
        feat_i = features[0][i].unsqueeze(0).unsqueeze(0)
        feat_j = features[0][j].unsqueeze(0).unsqueeze(0)
        sim = cross_attention_module(feat_i,feat_j)
        print(sim)
        sim_matrix[i][j] = sim
        sim_matrix[j][i] = sim
    return sim_matrix