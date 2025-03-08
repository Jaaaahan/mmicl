# Code partially borrowed from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/eval/eval_datasets.py # noqa: E501

import json
import os

from torch.utils.data import Dataset
from PIL import Image
from .vqa_metric import compute_vqa_accuracy
from tqdm import tqdm


class VQADataset(Dataset):
    def __init__(
        self,
        image_dir_path,
        question_path,
        annotations_path,
        is_train,
        dataset_name,
        filter=None,
    ):
        self.question_path = question_path
        self.annotations_path = annotations_path

        self.questions = json.load(open(question_path, "r"))["questions"]
        if annotations_path is not None:
            self.answers = json.load(open(annotations_path, "r"))["annotations"]
        else:
            self.answers = None
        self.image_dir_path = image_dir_path
        self.is_train = is_train
        self.dataset_name = dataset_name
        if self.dataset_name in {"vqav2", "okvqa"}:
            self.img_coco_split = self.image_dir_path.strip("/").split("/")[-1]
            assert self.img_coco_split in {"train2014", "val2014", "test2015"}

        if filter is not None:
            self.questions = [
                q
                for q in tqdm(self.questions, desc="filtering q")
                if q["question_id"] in filter
            ]
            if annotations_path is not None:
                self.answers = [
                    a
                    for a in tqdm(self.answers, desc="filtering a")
                    if a["question_id"] in filter
                ]

    def __len__(self):
        return len(self.questions)

    def get_img_path(self, question):
        if self.dataset_name in {"vqav2", "okvqa"}:
            return os.path.join(
                self.image_dir_path,
                f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg"
                if self.is_train
                else f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg",
            )
        elif self.dataset_name == "vizwiz":
            return os.path.join(self.image_dir_path, question["image_id"])
        elif self.dataset_name == "textvqa":
            return os.path.join(self.image_dir_path, f"{question['image_id']}.jpg")
        else:
            raise Exception(f"Unknown VQA dataset {self.dataset_name}")

    def __getitem__(self, idx):
        question = self.questions[idx]
        img_path = self.get_img_path(question)
        image = Image.open(img_path)
        image.load()
        results = {
            "image": image,
            "question": question["question"],
            "id": question["question_id"],
            "label": self.answers[idx]["answers"][0]["answer"],
            "image_id": question["image_id"],
            # todo only taking first answer, needs verification
        }
        return results

    @staticmethod
    def result(sample, answer):
        return {
            "question_id": sample["id"],
            "answer": answer,
        }

    def score(self, result_path):
        return compute_vqa_accuracy(
            result_path, self.question_path, self.annotations_path
        )

    @staticmethod
    def prompt(image, question, label, hide_label=False, **kwargs):
        label = "" if hide_label else f"{label} \n"
        image = ["Image: ", image] if image else []
        question = f"Question: {question}" if question else ""
        return image + [f"{question} Answer: {label}"]



class VQA_OCR(VQADataset):
    def __init__(
        self,
        image_dir_path,
        question_path,
        annotations_path,
        ocr_path,
        is_train,
        dataset_name,
        filter=None,
    ):
        super().__init__(
            image_dir_path,
            question_path,
            annotations_path,
            is_train,
            dataset_name,
            filter,
        )
        self.ocr_path = ocr_path
        self.ocr = json.load(open(ocr_path, "r"))["data"]

        self.ocr = {d["image_id"]: " ".join(d["ocr_tokens"]) for d in self.ocr}

    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        res["ocr"] = self.ocr[res["image_id"]]
        return res

    @staticmethod
    def prompt(image, question, ocr, label, hide_label=False, **kwargs):
        label = "" if hide_label else f"{label} \n"
        ocr = f" OCR: '{ocr}'" if ocr else ""
        image = ["Image: ", image] if image else []
        question = f"Question: {question}" if question else ""
        return image + [f" {ocr} {question} Answer: {label}"]
