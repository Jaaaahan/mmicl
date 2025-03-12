import json
import torch

# 1. 读取 JSON 文件
json_path = "/data/jyh/mmicl/datasets/twitter1517/sd_features/updated_annotations_qwen_twitter1517_understand.json"
with open(json_path, "r") as f:
    data = json.load(f)

# 2. 提取所有样本的 sd_feature
sd_features_list = [
    item["sd_feature"] for item in data["annotations"] if "sd_feature" in item
]

# 3. 转换为 PyTorch 张量
if sd_features_list:  # 确保列表非空
    sd_features = torch.tensor(sd_features_list, dtype=torch.float32)

    # 4. 保存为 .pt 文件
    torch.save(sd_features, "sd_features.pt")
    print(f"sd_features saved! Shape: {sd_features.shape}")
else:
    print("Error: No sd_feature found in the JSON file!")
