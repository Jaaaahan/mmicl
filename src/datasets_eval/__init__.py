from .captioning import Coco, Flickr
from .hatefulmemes import HatefulMemes
from .imagenet import ImageNet
#from .mmmu import MMMU
from .vqa import VQADataset, VQA_OCR
from .rendered_sst2 import RenderedSST2
from .cifar import Cifar100, Cifar10_random, Cifar10
from .scienceqa import ScienceQA
from .twitter1517 import Twitter1517Dataset  # 🚀 新增导入

DATASETS = {
    "coco": {
        "dataset": Coco,
        "rices": "image",
    },
    "imagenet": {
        "dataset": ImageNet,
        "rices": "image",
    },
    "vqa": {
        "dataset": VQADataset,
        "rices": "image_question",
    },

    "flickr": {
        "dataset": Flickr,
        "rices": "image",
    },
    "okvqa": {
        "dataset": VQADataset,
        "rices": "image_question",
    },
    "hateful_memes": {
        "dataset": HatefulMemes,
        "rices": "image_ocr",
    },
    "rendered_sst2": {
        "dataset": RenderedSST2,
        "rices": "image",
    },
    "textvqa": {
        "dataset": VQADataset,
        "rices": "image_question",
    },
    "cifar100": {
        "dataset": Cifar100,
        "rices": "image",
    },
    "cifar10": {
        "dataset": Cifar10,
        "rices": "image",
    },
    "vizwiz": {
        "dataset": VQADataset,
        "rices": "image_question",
    },
    "scienceqa": {
        "dataset": ScienceQA,
        "rices": "image_question",
    },
    "twitter1517": {  # 🚀 新增 Twitter15/17
        "dataset": Twitter1517Dataset,
        "rices": "image",  # 使用 image 作为 rices 关键字
    },
}
