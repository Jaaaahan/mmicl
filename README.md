# MMICL: 多模态上下文学习框架

<div align="center">

![MMICL Logo](https://img.shields.io/badge/MMICL-多模态上下文学习-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

## 📖 项目概述

MMICL (Multi-Modal In-Context Learning) 是一个用于多模态上下文学习的框架，专注于图像-文本任务的检索和推理。该框架支持多种大型语言模型（如Qwen、LLaVA、IDEFICS等）进行多模态推理，并实现了不同的示例检索策略和排序方法，以提高模型在少样本学习场景下的性能。

### 主要特点

- **多模态检索**：支持基于语义相似度的多模态示例检索
- **灵活的采样策略**：实现了多种采样方法（随机、平衡、RICE等）
- **多样的排序策略**：支持多种示例排序方法（随机、反转、相似度等）
- **多模型支持**：集成了多种大型语言模型（Qwen、LLaVA、IDEFICS等）
- **数据集支持**：主要针对Twitter1517数据集进行优化，同时支持其他多模态数据集

## 🚀 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/Jaaaahan/mmicl.git
cd mmicl

# 安装依赖
pip install -r requirements.txt
```

### 配置文件

在使用前，请先修改`configs/template.yaml`文件，设置数据集路径：

```yaml
cache: "run/cache/"
logs: "run/logs/"

twitter1517:  # Twitter1517 数据集路径
  args:
    annotations_path: "/path/to/annotations.json"
    image_path: "/path/to/images"

twitter1517_test:  # Twitter1517 测试集路径
  args:
    annotations_path: "/path/to/test_annotations.json"
    image_path: "/path/to/images"
```

## 💡 核心功能

### 检索器 (Retriever)

检索器负责从支持集中检索与查询样本相似的示例，支持多种采样和排序策略：

```python
from src.retriever import Retriever

# 配置检索器
config = {
    "batch_size": 2,
    "dataset": "twitter1517",
    "test_dataset": "twitter1517_test",
    "num_shots": 4,
    "sampling": "rice",  # 可选: random, balanced, none, rice
    "ordering": "similarity",  # 可选: leave, random, reverse, similarity
    "paths": "configs/template.yaml"
}

# 初始化检索器
retriever = Retriever(config)

# 执行检索
results = retriever.retrieve()
```

### 模型推理

支持多种大型语言模型进行多模态推理：

```python
# 使用 Qwen 模型
from qwen_llava_test.qwen import QWen2_5VLInfer

qwen_model = QWen2_5VLInfer()
qwen_model.initialize(model_id="Qwen/Qwen2.5-VL-7B-Instruct")
result = qwen_model.infer()

# 使用 LLaVA 模型
from qwen_llava_test.llava import LlavaInfer

llava_model = LlavaInfer(images=images, texts=texts, prompt=prompt)
llava_model.initialize(model_id="llava-1.5-7b-hf")
result = llava_model.infer()
```

## 📊 数据集

### Twitter1517

项目主要针对Twitter1517数据集进行优化，该数据集包含推文文本和图像，用于研究多模态语义不一致性检测。

```python
from src.datasets_eval.twitter1517 import Twitter1517Dataset

# 加载数据集
dataset = Twitter1517Dataset(
    annotations_path="/path/to/annotations.json",
    image_path="/path/to/images"
)
```

## 🧩 项目结构

```
├── assets/                # 特征缓存和模型资源
├── configs/               # 配置文件
├── datasets/              # 数据集文件
│   └── twitter1517/       # Twitter1517 数据集
├── qwen_llava_test/       # Qwen 和 LLaVA 模型实现
├── retriver_results/      # 检索结果保存目录
├── src/                   # 源代码
│   ├── datasets_eval/     # 数据集评估模块
│   ├── models/            # 模型实现
│   ├── retriever.py       # 检索器实现
│   ├── sampling.py        # 采样策略
│   ├── ordering.py        # 排序策略
│   └── utils.py           # 工具函数
└── utils/                 # 辅助工具脚本
```

## 📄 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 👥 贡献

欢迎贡献代码、报告问题或提出改进建议！请随时提交 Pull Request 或创建 Issue。

## 📧 联系方式

如有任何问题，请联系：jay119059@gamil.com
