import torchvision.datasets as ds
from .utils import ClassificationDataset


class RenderedSST2(ds.RenderedSST2, ClassificationDataset):
    labels = ["negative", "positive"]

    def __init__(
        self, root, split, transform=None, target_transform=None, download=False
    ):
        ds.RenderedSST2.__init__(
            self,
            root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        ClassificationDataset.__init__(self)

        self.split = split

    def __len__(self):
        return super(RenderedSST2, self).__len__()

    def __getitem__(self, idx):
        image, label = super(RenderedSST2, self).__getitem__(idx)

        return {
            "id": int(idx),
            "image": image,
            "label": self.id_to_name[label],
            "class_id": label,
        }

    def result(self, sample, class_name):
        return self.exact_match_result(sample, class_name)

    @staticmethod
    def prompt(image, label, hide_label=False, **kwargs):
        label = "" if hide_label else f"{label} \n"
        image = ["Image: ", image] if image else []
        return image + [f" Sentiment: {label}"]
