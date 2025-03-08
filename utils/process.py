import json
import pandas as pd

# 读取 TXT 数据，假设是逗号分隔的 CSV 格式
file_path = "/data/jyh/multimodal-icl/datasets/twitter1517/test.txt"  # 修改为你的 txt 文件路径
df = pd.read_csv(file_path, delimiter=',')  # 根据实际情况修改分隔符

# 筛选 cor_label 为 'n' 的样本
filtered_df = df[df["information_label"] == "1"].copy()

# 映射 label 到新的情感标签
label_mapping = {1: 2, -1: 0, 0: 1}  # 1 → positive (2), -1 → negative (0), 0 → neutral (1)
filtered_df["label"] = filtered_df["label"].map(label_mapping)

# 重新构建 COCO 结构的 JSON 数据
coco_json = {
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
        "file_name": row["id"],  # 作为图片检索依据
        "id": int(row["id"])
    }
    annotation_entry = {
        "image_id": int(row["id"]),
        "id": int(row["id"]),
        "text": row["text"],  # 作为图片的描述
        "label": int(row["label"])  # 使用 label 作为最终情感标签
    }

    coco_json["images"].append(image_entry)
    coco_json["annotations"].append(annotation_entry)

# 保存 JSON 文件
json_path = "/data/jyh/multimodal-icl/datasets/twitter1517/twitter1517_test.json"
with open(json_path, "w") as f:
    json.dump(coco_json, f, indent=4)

# 返回 JSON 文件路径
print(f"✅ JSON 文件已生成：{json_path}")
