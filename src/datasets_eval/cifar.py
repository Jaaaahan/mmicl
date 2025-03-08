from datasets import load_dataset

# sys.path.append(__file__)
from .utils import ClassificationDataset, save_load_from_disk
import os
import random


CIFAR100_LABELS = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "cra",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]


class Cifar100(ClassificationDataset):
    labels = CIFAR100_LABELS

    def __init__(self, split="train"):
        super().__init__()
        # self.dataset = load_dataset("cifar100", split=split)
        if "SAVE_DATASET" in os.environ:
            self.dataset = save_load_from_disk(
                "cifar100", os.environ["SAVE_DATASET"], split
            )
        else:
            self.dataset = load_dataset("cifar100", split=split)
        self.dataset.set_format(columns=["img", "fine_label"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        idx = int(idx)
        class_id = self.dataset[idx]["fine_label"]
        return {
            "id": idx,
            "image": self.dataset[idx]["img"],
            "label": self.id_to_name[class_id],
            "class_id": class_id,
        }

    def result(self, sample, class_name):
        return self.levenshtein_result(sample, class_name)

    @staticmethod
    def prompt(image, label, hide_label=False, **kwargs):
        label = "" if hide_label else f"{label} \n"
        image = ["Image: ", image] if image else []
        return image + [f" This is a picture of a {label}"]


CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class Cifar10(Cifar100):
    labels = CIFAR10_LABELS

    def __init__(self, split="train"):
        super().__init__()
        # self.dataset = load_dataset("cifar100", split=split)
        if "SAVE_DATASET" in os.environ:
            self.dataset = save_load_from_disk(
                "cifar10", os.environ["SAVE_DATASET"], split
            )
        else:
            self.dataset = load_dataset("cifar10", split=split)
        self.dataset.set_format(columns=["img", "label"])

    def __getitem__(self, idx):
        idx = int(idx)
        class_id = self.dataset[idx]["label"]
        return {
            "id": idx,
            "image": self.dataset[idx]["img"],
            "label": self.id_to_name[class_id],
            "class_id": class_id,
        }


class Cifar10_random(Cifar10):
    labels = random.sample(CIFAR10_LABELS, len(CIFAR10_LABELS))
