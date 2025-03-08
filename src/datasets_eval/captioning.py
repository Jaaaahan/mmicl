import json
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from collections import defaultdict
from tqdm import tqdm


class Coco(Dataset):
    def __init__(self, annotations_path, image_path):
        self.coco = COCO(annotations_path)
        self.annotations_path = annotations_path
        self.image_path = image_path
        self.image_ids = self.coco.getImgIds()

    def __len__(self):
        # warning: we iterate over images therefore we do net see all captions
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Get the image information
        image_info = self.coco.loadImgs(image_id)[0]
        image = Image.open(self.image_path + image_info["file_name"])
        image.load()

        # Get the captions for the image
        ann_ids = self.coco.getAnnIds(imgIds=image_id)

        # Randomly select one caption
        # caption = self.coco.loadAnns(random.choice(ann_ids))[0]["caption"]

        # Always select first caption
        caption = self.coco.loadAnns(ann_ids[0])[0]["caption"]

        return {
            "id": self.image_ids[idx],
            "image": image,
            "label": caption,
        }

    @staticmethod
    def result(sample, caption):
        return {
            "image_id": sample["id"],
            "caption": caption,
        }

    def score(self, result_path):
        # create coco object and coco_result object
        coco = COCO(self.annotations_path)
        coco_result = coco.loadRes(result_path)

        # create coco_eval object by taking coco and coco_result
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.params["image_id"] = coco_result.getImgIds()
        coco_eval.evaluate()

        return coco_eval.eval["CIDEr"] * 100

    @staticmethod
    def prompt(image, label, hide_label=False, **kwargs):
        label = "" if hide_label else f"{label} \n"
        image = ["Image: ", image] if image else []
        return image + [f" Caption: {label}"]


class Flickr(Coco):
    def __init__(self, annotations_path, image_path, split="train"):
        self.annotations_path = annotations_path
        self.image_path = image_path
        self.split = split

        images = json.load(open(annotations_path, "r"))["images"]
        annotations = json.load(open(annotations_path, "r"))["annotations"]

        self.image_ids = {im["image_id"]: im["split"] for im in images}

        self.image_to_captions = defaultdict(list)
        for ann in tqdm(annotations, desc=f"loading flickr {split}", leave=False):
            # for each image id, find split and add caption
            if self.image_ids[ann["image_id"]] == split:
                self.image_to_captions[ann["image_id"]].append(ann["caption"])

        self.image_ids = list(self.image_to_captions.keys())

    def __getitem__(self, idx):
        image_id = list(self.image_ids)[idx]

        # Get the image information
        image = Image.open(self.image_path + f"{image_id}.jpg")
        image.load()

        # Get the captions for the image
        captions = self.image_to_captions[image_id]

        # Randomly select one caption
        # caption = random.choice(captions)

        # Always select first caption
        caption = captions[0]

        return {
            "id": image_id,
            "image": image,
            "label": caption,
        }
