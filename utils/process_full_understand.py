import json
import re

# 读取第一个文件 qwen_fully_understood_entries.json
with open("/data/jyh/mmicl/datasets/twitter1517/qwen_ubderstand/sd_label_1.json", "r") as f:
    qwen_data = [json.loads(line) for line in f]  # 逐行读取 JSONL 格式数据

# 读取第二个文件 twitter1517_coco_updated.json
with open("/data/jyh/mmicl/datasets/twitter1517/qwen_ubderstand/formatted_qwen_twitter1517_understand.json", "r") as f:
    target_data = json.load(f)

# 初始化新的 JSON 结构
output_data = {
    "info": target_data["info"],  # 直接复制 info 部分
    "images": [],
    "annotations": []
}

# 处理 images 和 annotations
for entry in qwen_data:
    # 从 image_path 提取文件名
    file_name = entry["image_path"].split("/")[-1]  # 获取 "2088.jpg"
    
    # 从文件名提取 id（提取数字部分）
    image_id = int(re.search(r"\d+", file_name).group())  # 2088
    
    # 解析 qwen_output 里的 Combination 字段
    combination_label = entry["qwen_output"][0].split("Combination:")[-1].strip()
    
    # 添加到 images
    output_data["images"].append({
        "file_name": file_name,
        "id": image_id
    })
    
    # 添加到 annotations
    output_data["annotations"].append({
        "image_id": image_id,
        "id": image_id,
        "text": entry["txt"],
        "label": combination_label
    })

# 生成新的 JSON 文件
output_file = "/data/jyh/mmicl/datasets/twitter1517/formatted_qwen_twitter1517_understand_sd.json"
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print(f"数据已成功格式化并保存至 {output_file}")
