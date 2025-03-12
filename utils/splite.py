import json
import os

# 读取JSON文件并按行解析
def split_json_by_sd_label(input_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    sd_label_data = {}
    
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                sd_label = data.get("sd_label", "unknown")
                
                if sd_label not in sd_label_data:
                    sd_label_data[sd_label] = []
                
                sd_label_data[sd_label].append(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    
    # 保存拆分后的数据到不同文件
    for label, items in sd_label_data.items():
        output_file = os.path.join(output_dir, f"sd_label_{label}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(items)} items to {output_file}")

# 示例调用
input_json_file = "/data/jyh/mmicl/datasets/twitter1517/qwen_fully_understood_entries.json"  # 替换为实际的 JSON 文件路径
output_directory = "output_json"  # 输出目录
split_json_by_sd_label(input_json_file, output_directory)
