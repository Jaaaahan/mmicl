from typing import List
import json
import torch
import Levenshtein
import os
from datasets import load_dataset, load_from_disk


def find_label_in_text(text: str, labels: List[str]) -> tuple[int, str]:
    lower_text = text.lower()
    for i, label in enumerate(labels):
        if label.lower() in lower_text:
            return i, label
    return -1, ""


def compute_accuracy(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must be of the same length")

    correct_matches = sum(el1 == el2 for el1, el2 in zip(list1, list2))
    accuracy = correct_matches / len(list1)
    return accuracy * 100


def find_closest_string(target, strings):
    """
    Finds closest string in a list to the target string based on Levenshtein distance.

    :param target: Target string to compare with.
    :param strings: List of strings to search in.
    :return: The closest string from the list.
    """
    # Calculating Levenshtein distance for each string in the list
    distances = [Levenshtein.distance(target, s) for s in strings]

    # Finding the index of the smallest distance
    min_index = distances.index(min(distances))

    return strings[min_index]


def save_load_from_disk(dataset_name, path, split):
    dataset_path = os.path.join(path, dataset_name)
    # Check if the dataset exists at the specified path
    if os.path.exists(dataset_path):
        # Load the dataset from disk
        dataset = load_from_disk(dataset_path)[split]
    else:
        # If dataset is not found on disk, load it and save it
        dataset = load_dataset(dataset_name)
        dataset.save_to_disk(dataset_path)
        dataset = dataset[split]
    return dataset


class EvalDataset(torch.utils.data.Dataset):
    @staticmethod
    def result(sample, class_name):
        raise NotImplementedError


class ClassificationDataset(EvalDataset):
    def __init__(self):
        super().__init__()
        self.name_to_id = {label: idx for idx, label in enumerate(self.labels)}
        self.id_to_name = dict(enumerate(self.labels))

    def exact_match_result(self, sample, class_name):
        try:
            class_id = self.name_to_id[class_name]
        except KeyError:
            class_id = -1

        return {
            "id": sample["id"],
            "class_id": class_id,
        }

    def levenshtein_result(self, sample, class_name):
        class_name = find_closest_string(class_name, self.labels)
        return self.exact_match_result(sample, class_name)

    def get_labels(self):
        return self.labels

    def score(self, result_path: str):
        """
        Calculate and return the accuracy score based on the results from a JSON file,
        using the compute_accuracy function.
        """
        with open(result_path, "r") as f:
            results = json.load(f)

        # Prepare lists of ground truths and predicted classes
        ground_truths = [self[result["id"]]["class_id"] for result in results]
        predicted_classes = [result["class_id"] for result in results]

        # Use the compute_accuracy function
        return compute_accuracy(ground_truths, predicted_classes)
