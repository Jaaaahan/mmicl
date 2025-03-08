import pandas as pd
import json


# 假设你的TXT文件路径为 "/mnt/data/twitter_data.txt" 并且是用制表符('\t')分隔的
file_path = "/data/jyh/mmicl/datasets/twitter1517/test.txt"
# 初始化一个列表来保存所有读取到的字典项
data_list = []

# 打开txt文件并逐行读取
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 假设每一行都是一个有效的JSON字符串
        item_dict = json.loads(line.strip())
        data_list.append(item_dict)

# 将列表转换为DataFrame
df = pd.DataFrame(data_list)


# 筛选 cor_label 为 'n' 并且 information_label 为 0 的样本
filtered_df = df[(df["information_label"] == 1)].copy()

# 映射 multi_label 到新的情感标签，确保在此之前你已经定义了label_mapping字典
filtered_df["label"] = filtered_df["label"]

# 构建 COCO 结构的 JSON 数据
coco_json_updated = {
    "info": {
        "description": "Twitter-15/17 Sentiment Dataset (Semantic Inconsistent)",
        "version": "1.0",
        "year": 2025,
        "contributor": "Custom",
        "date_created": "2025/02/02"
    },
    "images": [],
    "annotations": []
}

# 遍历数据，填充 images 和 annotations
for idx, row in filtered_df.iterrows():
    image_entry = {
        "file_name": row["id"],  # 使用 new_image_id 作为图片检索依据
        "id": int(row["id"].split(".")[0])
    }
    annotation_entry = {
        "image_id": int(row["id"].split(".")[0]),
        "id": int(row["id"].split(".")[0]),
        "text": row["text"],  # 图片对应的文本描述
        "label": int(row["label"])  # 使用映射后的情感标签
    }

    coco_json_updated["images"].append(image_entry)
    coco_json_updated["annotations"].append(annotation_entry)

# 保存 JSON 文件
json_path_updated = "/data/jyh/mmicl/datasets/twitter1517/test_non_sd.json"
with open(json_path_updated, "w") as f:
    json.dump(coco_json_updated, f, indent=4)

# 打印完成信息
print(f"JSON 文件已保存至: {json_path_updated}")