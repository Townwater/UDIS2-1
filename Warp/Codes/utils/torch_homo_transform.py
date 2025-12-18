import torch
import torch.nn.functional as F

def transformer(input_tensor, theta, out_size):
    """
    input_tensor: [B, C, H, W] 輸入影像張量
    theta: [B, 3, 3] 齊次變換矩陣 (Homography)
    out_size: (out_height, out_width) 輸出大小
    """

    batch_size, channels, height, width = input_tensor.size()
    out_height, out_width = out_size
    # print(f"out_height={out_height}, out_width={out_width}")
    # 1. 建立標準化網格座標 (範圍 [-1, 1])
    # grid_y: (out_height, out_width), grid_x: (out_height, out_width)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, out_height, device=input_tensor.device),
        torch.linspace(-1, 1, out_width, device=input_tensor.device),
        indexing='ij'
    )  # PyTorch 1.10+ 才支援 indexing='ij'

    # 合併成 (out_height * out_width, 3), 同時加齊次座標 1
    ones = torch.ones_like(grid_x).view(-1)  # (out_h*out_w,)
    grid_x_flat = grid_x.contiguous().view(-1)
    grid_y_flat = grid_y.contiguous().view(-1)
    grid_homo = torch.stack([grid_x_flat, grid_y_flat, ones], dim=0)  # (3, out_h*out_w)

    # 2. 對 batch 補維度並做齊次變換矩陣乘法
    # theta: [B, 3, 3]
    # grid_homo = grid_homo.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 3, out_h*out_w)
    grid_homo = grid_homo.unsqueeze(0).expand(batch_size, -1, -1)  # 改成 expand 而非 repeat
    warped_grid = torch.bmm(theta, grid_homo)  # (B, 3, out_h*out_w)

    # 3. 坐標除以第三維齊次分量，轉回 Cartesian 坐標
    warped_x = warped_grid[:, 0, :] / (warped_grid[:, 2, :] + 1e-8)
    warped_y = warped_grid[:, 1, :] / (warped_grid[:, 2, :] + 1e-8)

    # 4. 將座標轉換回 grid_sample 要求的格式
    # grid_sample 要求的座標是 (B, out_h, out_w, 2)，且 x,y 都在 [-1,1]
    warped_x = warped_x.view(batch_size, out_height, out_width)
    warped_y = warped_y.view(batch_size, out_height, out_width)
    # 合成最後 grid: (B, out_h, out_w, 2)，順序是 (x, y)
    grid = torch.stack((warped_x, warped_y), dim=3)

    # 5. 使用 grid_sample 做雙線性插值 (padding_mode 可設 "zeros"、"border"、"reflection")
    output = F.grid_sample(input_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    return output
