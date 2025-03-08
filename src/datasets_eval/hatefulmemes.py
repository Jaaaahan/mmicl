import json
import os
from PIL import Image
from .utils import ClassificationDataset


class HatefulMemes(ClassificationDataset):
    labels = ["no", "yes"]

    def __init__(self, image_dir_path, annotations_path):
        super().__init__()
        self.image_dir_path = image_dir_path
        with open(annotations_path, "r") as f:
            self.annotations = [json.loads(line) for line in f]

        # id to index
        self.id2idx = {
            int(annotation["id"]): idx
            for idx, annotation in enumerate(self.annotations)
        }

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_dir_path, annotation["img"].split("/")[-1])
        image = Image.open(img_path)
        image.load()
        return {
            "id": int(idx),
            "image": image,
            "ocr": annotation["text"],
            "label": self.id_to_name[annotation["label"]],
            "class_id": annotation["label"],
        }

    @staticmethod
    def prompt(image, ocr: str, label: str, hide_label: bool = False, **kwargs):
        label = "" if hide_label else f"{label} \n"
        image = ["Image: ", image] if image else []
        ocr = f"OCR: '{ocr}'" if ocr else ""
        return image + [f" {ocr} Is this meme hateful? {label}"]

    def result(self, sample, label):
        return self.exact_match_result(sample, label)
