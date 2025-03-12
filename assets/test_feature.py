import torch

# 指定 pt 文件路径
pt_file = "/data/jyh/mmicl/assets/sd_features.pt"

# 加载 PyTorch 张量
sd_features = torch.load(pt_file)

# 打印张量 shape
print("sd_features shape:", sd_features.shape)
