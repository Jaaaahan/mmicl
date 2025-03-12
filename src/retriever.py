import os
import json
import yaml
import torch
import tqdm
from torch.utils.data import DataLoader
from types import SimpleNamespace
from datasets_eval import DATASETS
from accelerate import Accelerator
from utils import custom_collate_fn, save_to_tmp_json
from sampling import RandomSampler, BalancedSampler, NoneSampler, RicesSampler
from ordering import ReverseOrdering, RandomOrdering, SimilarityOrdering

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
torch.set_default_dtype(torch.float32)

class Retriever:
    def __init__(self, config):
        """
        初始化 Retriever 类，config 为字典参数，包含如下键：
            batch_size, dataset, test_dataset, num_shots,
            sampling, ordering, limit_batches, paths, seed, num_runs
        """
        self.batch_size = config.get("batch_size", 2)
        self.dataset_name = config.get("dataset", "twitter1517")
        self.test_dataset_name = config.get("test_dataset", "twitter1517_test")
        self.num_shots = config.get("num_shots", 4)
        self.sampling = config.get("sampling", "rice")
        self.ordering_type = config.get("ordering", "leave")
        self.limit_batches = config.get("limit_batches", None)
        self.seed = config.get("seed", 42)
        self.num_runs = config.get("num_runs", 1)
        self.paths_file = config.get("paths", "/data/jyh/mmicl/configs/template.yaml")
        self.cached_features_path = config.get("cached_features_path","/data/jyh/mmicl/cached_features")
        self.cached_sd_features_path = config.get("cached_sd_features_path","/data/jyh/mmicl/cached_features")
        
        # 读取 YAML 配置文件
        with open(self.paths_file, "r") as f:
            self.paths = yaml.safe_load(f)
        os.makedirs(self.paths["cache"], exist_ok=True)
        
        self.accelerator = Accelerator()
        
        # 载入数据集及支持集
        ds_info, ds_args = DATASETS[self.dataset_name], self.paths[self.dataset_name]
        test_ds_args = self.paths[self.test_dataset_name]
        self.dataset = ds_info["dataset"](**test_ds_args["args"])
        self.support_set = ds_info["dataset"](**ds_args["args"])
        # 选择采样策略
        if self.sampling == "random":
            sampler = RandomSampler(self.dataset, self.support_set, num_shots=self.num_shots)
            self.loader = DataLoader(sampler, batch_size=self.batch_size, num_workers=4, collate_fn=custom_collate_fn)
        elif self.sampling == "balanced":
            sampler = BalancedSampler(self.dataset, self.support_set, num_shots=self.num_shots)
            self.loader = DataLoader(sampler, batch_size=self.batch_size, num_workers=4, collate_fn=custom_collate_fn)
        elif self.sampling == "none":
            sampler = NoneSampler(self.dataset)
            self.loader = DataLoader(sampler, batch_size=self.batch_size, num_workers=4, collate_fn=custom_collate_fn)
        elif self.sampling == "rice":
            # 使用内部 DataLoader 先遍历数据集
            internal_loader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4, collate_fn=custom_collate_fn)
            self.loader = RicesSampler(
                internal_loader,
                self.support_set,
                return_similarity=True,
                keys=["text", "image"],
                num_shots=self.num_shots,
                device="cuda",
                cached_features_path = self.cached_features_path,
                cached_sd_features_path = self.cached_sd_features_path
            )
        else:
            raise ValueError("不支持的采样策略")

        # 选择 ordering 策略
        if self.ordering_type == "leave":
            self.ordering = lambda examples, queries: examples
        elif self.ordering_type == "random":
            self.ordering = RandomOrdering("label")
        elif self.ordering_type == "reverse":
            self.ordering = ReverseOrdering("label")
        elif self.ordering_type == "similarity":
            self.ordering = SimilarityOrdering("text", device=self.accelerator.device)
        else:
            raise ValueError("不支持的 ordering 类型")

    def retrieve(self):
        """执行检索过程，并返回检索结果列表，同时保存结果到 JSON 文件"""
        retrieval_results = []
    
        for i, batch in enumerate(tqdm.tqdm(self.loader, total=len(self.loader))):
            examples, queries = batch["examples"], batch["query"]
            ordered_examples = self.ordering(examples, queries)
            
            for idx, query in enumerate(queries):
                # 如果不足，则使用最后一组示例
                ex_group = ordered_examples[idx] if idx < len(ordered_examples) else ordered_examples[-1]
                retireval_image = [os.path.basename(img_path) for img_path in ex_group["image"]]
                retrieval_results.append({
                    "query": {
                        "id": query["id"],
                        "text": query["text"],
                        "image": os.path.basename(query["image"]),
                        "label": query["label"],
                    },
                    "retrieval": {
                        "images": retireval_image,
                        "texts": ex_group["text"],
                        "raw_labels": ex_group["label"],
                        "similarity": ex_group["similarity"]
                    }
                })
            if self.limit_batches and i >= self.limit_batches - 1:
                break
            
        # Define the output directory where you want to save the results
        output_dir = "/data/jyh/mmicl"  # Change this to your preferred directory
    
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        # Specify the output file path
        output_json_path = os.path.join(output_dir, "/data/jyh/mmicl/retriver_results/add_sd_features/retrieval_results_non_sd_shot_1.json")
    
        # Save the results to a JSON file
        with open(output_json_path, "w") as json_file:
            json.dump(retrieval_results, json_file, indent=4)
    
        print(f"Retrieval results saved to {output_json_path}")
    
        # End the training process
        self.accelerator.end_training()
    
        return retrieval_results


# 示例调用
if __name__ == "__main__":
    # 配置参数可根据实际情况修改
    config = {
        "batch_size": 2,
        "dataset": "twitter1517",
        "test_dataset": "twitter1517_test",
        "num_shots": 1,
        "sampling": "rice",
        "ordering": "leave",
        "limit_batches": None,
        "paths": "/data/jyh/mmicl/configs/template.yaml",
        "seed": 42,
        "num_runs": 1,
        "cached_features_path":"/data/jyh/mmicl/assets/cached_features_full_understand",
        "cached_sd_features_path":"/data/jyh/mmicl/assets/sd_features.pt"
    }
    retriever = Retriever(config)
    results = retriever.retrieve()
