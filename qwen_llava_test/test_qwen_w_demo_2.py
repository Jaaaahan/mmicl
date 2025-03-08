import json
from qwen_llava_test.qwenextend import QWen2_5VLInferExtended

def main():
    # JSON 文件路径，请替换为实际文件路径
    json_file_path = '/data/jyh/multimodal-icl/retriver_results/retrieval_results_shot_1_with_similarity.json'
    
    # 结果保存路径
    output_json_path = '/data/jyh/multimodal-icl/llava_predict_results/llava_result_shot_1_version_2.json'

    # 读取JSON数据
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    results = []

    # 初始化模型
    model = QWen2_5VLInferExtended()
    model.initialize()  # 确保这里指向正确的模型路径
    
    # 遍历所有条目
    for item in data:
        query = item['query']
        demonstrations = [{
            'image': img,
            'text': txt,
            'label': lbl
        } for img, txt, lbl in zip(item['retrieval']['images'], item['retrieval']['texts'].split('\n\n\n'), item['retrieval']['raw_labels'])]

        # 进行推理
        result = model.infer_with_demonstrations(query=query, demonstrations=demonstrations)
        
        # 将结果添加到列表中
        results.append({
            "query_id": query["id"],
            "predicted_sentiment": result,
            "actual_label": query["label"]
        })

    # 将所有结果保存为JSON格式
    with open(output_json_path, 'w', encoding='utf-8') as output_file:
        json.dump(results, output_file, ensure_ascii=False, indent=4)

    print(f"All results have been saved to {output_json_path}")

if __name__ == "__main__":
    main()