from datasets import load_dataset
from .utils import find_label_in_text, compute_accuracy, save_load_from_disk
import torch
import json
import os


class ScienceQA(torch.utils.data.Dataset):
    def __init__(self, split: str = "train", **kwargs):
        super().__init__(**kwargs)
        dataset_name = "lmms-lab/ScienceQA-IMG"

        if "SAVE_DATASET" in os.environ:
            self.dataset = save_load_from_disk(
                dataset_name, os.environ["SAVE_DATASET"], split
            )
        else:
            # If SAVE_DATASET environment variable is not present, just load the dataset
            self.dataset = load_dataset(dataset_name, split=self.split)

    def __getitem__(self, index):
        index = int(index)
        sample = self.dataset[index]
        sample["id"] = index
        sample["label"] = sample["choices"][sample["answer"]]
        return sample

    def __len__(self):
        return len(self.dataset)

    def result(self, sample, answer):
        labels = [chr(ord("A") + i) for i in range(len(sample["choices"]))]
        answer_id, a = find_label_in_text(answer, labels)

        return {
            "id": sample["id"],
            "answer_id": answer_id,
        }

    def score(self, result_path: str):
        with open(result_path, "r") as f:
            results = json.load(f)

        # Prepare lists of ground truths and predicted classes
        ground_truths = [self[result["id"]]["answer"] for result in results]
        predicted_classes = [result["answer_id"] for result in results]

        # Use the compute_accuracy function
        return compute_accuracy(ground_truths, predicted_classes)

    @staticmethod
    def prompt(image, question, choices, hint, answer, hide_label=False, **kwargs):
        context = [item for item in (image, hint) if item not in (None, "")]
        if context:
            context = ["Context: "] + context + ["\n"]

        options = " ".join(
            [f"({chr(ord('A') + i)}) {c} " for i, c in enumerate(choices)]
        )
        label = "" if hide_label else f"{chr(ord('A') + answer)} \n"
        return (
            [f"\nQuestion: {question}\nOptions: {options}\n"]
            + context  # image is in here
            + [f" Answer: {label}"]
        )
