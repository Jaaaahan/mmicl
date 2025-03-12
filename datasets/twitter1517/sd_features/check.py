import json

# 指定 JSON 文件路径
json_file_path = "/data/jyh/mmicl/datasets/twitter1517/sd_features/updated_annotations_test_sd.json"

# 读取 JSON 文件
with open(json_file_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)

# 统计缺少 sd_feature 的样本
missing_features = []

for annotation in json_data.get("annotations", []):
    if "sd_feature" not in annotation or annotation["sd_feature"] is None:
        missing_features.append(annotation["id"])

# 输出检查结果
if missing_features:
    print(f"以下 {len(missing_features)} 个样本缺少 'sd_feature'：")
    print(missing_features)
else:
    print("所有样本均包含 'sd_feature'，检查通过！")
