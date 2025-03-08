import json
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torch

class Twitter1517Dataset(Dataset):
    def __init__(self, annotations_path, image_path, transform=None, prompt_template=None):
        """
        Twitter-15/17 数据集 (语义不一致支持集)
        :param annotations_path: JSON 标注文件路径
        :param image_path: 图片文件夹路径
        :param transform: 图像预处理
        :param prompt_template: 可选的自定义 prompt 模板，模板中可以使用 {image}、{text} 和 {label} 占位符
        """
        self.annotations_path = annotations_path
        self.image_path = image_path
        self.transform = transform
        self.prompt_template = prompt_template  # 自定义模板

        # 加载 JSON 标注
        with open(self.annotations_path, "r") as f:
            self.data = json.load(f)

        self.image_dict = {img["id"]: img for img in self.data["images"]}
        self.annotations = self.data["annotations"]
        self.image_ids = list(self.image_dict.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # 获取图像信息
        image_info = self.image_dict[image_id]
        image_file = os.path.join(self.image_path, image_info["file_name"])
        #image = Image.open(image_file).convert("RGB")

        # 获取文本描述 & 标签
        annotation = next(ann for ann in self.annotations if ann["image_id"] == image_id)
        text = annotation["text"]
        #label = torch.tensor(annotation["label"], dtype=torch.long)
        label = annotation["label"]

        if self.transform:
            image = self.transform(image)

        return {
            "id": image_id,
            "image": image_file,
            "text": text,
            "label": label,
        }

    @staticmethod
    def result(sample, prediction):
        """
        生成预测结果
        :param sample: 数据集样本
        :param prediction: 预测的情感类别
        :return: 结果字典
        """
        return {
            "image_id": sample["id"],
            "prediction": prediction,  # 直接返回分类标签
            "label": sample["label"].item(),  # Ground truth
        }

    def score(self, result_path):
        """
        计算分类准确率
        :param result_path: 结果 JSON 文件路径
        :return: 准确率
        """
        with open(result_path, "r") as f:
            results = json.load(f)

        correct = sum(1 for r in results if r["prediction"] == r["label"])
        accuracy = correct / len(results)
        print(f"Classification Accuracy: {accuracy:.4f}")
        return accuracy * 100

    @staticmethod
    def default_template():
        """
        定义默认的 prompt 模板
        模板中 {image}、{text} 和 {label} 占位符会分别被图像、文本描述和标签替换
        """
        return (
            "Image: {image}\n"
            "Text: {text}\n"
            "Label options: positive (2), neutral (1), negative (0).the sentiment label is:{label} \n"
        )

    def prompt(self, image, text, label, hide_label=False, **kwargs):
        """
        生成单个实例的 prompt。
        :param image: 图像数据（可以是图像文件路径或者图像的其他表示形式）
        :param text: 文本描述
        :param label: 标签数据，默认展示，但可隐藏
        :param hide_label: 是否隐藏标签，默认 False
        :param kwargs: 允许传入额外参数用于模板扩展
        :return: 格式化后的 prompt 字符串
        """
        label_str = "" if hide_label else str(label)
        template = self.prompt_template if self.prompt_template is not None else Twitter1517Dataset.default_template()
        prompt_text = template.format(image=image, text=text, label=label_str, **kwargs)
        return prompt_text

    def prompt_group(self, images, texts, labels, hide_label=True, separator="\n\n", **kwargs):
        """
        针对支持集组（包含多个实例）生成 prompt，
        为组内每个实例生成单独 prompt，然后用 separator 拼接。
        :param images: 图像列表
        :param texts: 文本描述列表
        :param labels: 标签列表
        :param hide_label: 默认隐藏真实标签（仅作为示例）
        :param separator: 分隔符
        :param kwargs: 额外参数传入 prompt 模板
        :return: 拼接后的支持集 prompt 字符串
        """
        prompts = []
        for img, txt, lab in zip(images, texts, labels):
            p = self.prompt(img, txt, lab, hide_label=hide_label, **kwargs)
            prompts.append(p)
        return separator.join(prompts)
