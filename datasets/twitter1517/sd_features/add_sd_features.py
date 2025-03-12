import json
import numpy as np

# 加载 npz 文件
npz_data = np.load("/data/jyh/mmicl/datasets/twitter1517/sd_features/demo_features_with_ids.npz")
features = npz_data["features"]  # 形状假设为 (N, D)
sample_ids = npz_data["sample_ids"]  # 形状假设为 (N,)

# 读取 JSON 文件
with open("/data/jyh/mmicl/datasets/twitter1517/qwen_ubderstand/formatted_qwen_twitter1517_understand.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

# 建立 id 到特征的映射
id_to_feature = {int(sample_id): feature.tolist() for sample_id, feature in zip(sample_ids, features)}

# 遍历 JSON 文件，添加 'sd_feature' 字段
for annotation in json_data["annotations"]:
    image_id = annotation["image_id"]
    annotation["sd_feature"] = id_to_feature.get(image_id, None)  # 若无匹配，则为 None

# 保存更新后的 JSON 文件
with open("updated_annotations_qwen_twitter1517_understand.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=4, ensure_ascii=False)

print("JSON 文件已更新并保存为 updated_annotations.json")
