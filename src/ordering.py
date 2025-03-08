from collections import defaultdict
import torch
from transformers import CLIPProcessor, CLIPModel


class Ordering:
    def __init__(self, key):
        self.key = key

    def _inner(self, examples, query):
        pass

    @staticmethod
    def _order(examples, ids_order):
        # for each key of examples, order the values according to ids_order
        ordered_examples = {}
        for key, values in examples.items():
            ordered_examples[key] = [values[i] for i in ids_order]
        return ordered_examples

    def __call__(self, examples, query):
        return [self._inner(e, q) for e, q in zip(examples, query)]


class RandomOrdering(Ordering):
    def __init__(self, key):
        super().__init__(key)

    def _inner(self, examples, query):
        return self._order(examples, torch.randperm(len(examples[self.key])))


class ReverseOrdering(Ordering):
    def __init__(self, key):
        super().__init__(key)

    def _inner(self, examples, query):
        return self._order(examples, torch.arange(len(examples[self.key])).flip(0))


class GroupOrdering(Ordering):
    def __init__(self, key, order):
        super().__init__(key)
        self.order = order

    def _inner(self, examples, query):
        grouped_examples = defaultdict(list)
        for i, e in enumerate(examples[self.key]):
            grouped_examples[e].append(i)

        # assert that every key of grouped_examples is in self.order
        assert set(grouped_examples.keys()) <= set(self.order)

        ids_order = []
        for group in self.order:
            ids_order.extend(grouped_examples.get(group, []))
        return self._order(examples, ids_order)


class AlternateOrdering(Ordering):
    def _inner(self, examples, query):
        grouped_examples = defaultdict(list)
        for i, e in enumerate(examples[self.key]):
            grouped_examples[e].append(i)

        # while dict is not empty
        ids_order = []
        while grouped_examples:
            # for each key of grouped_examples, pop the first value
            for key in list(grouped_examples.keys()):
                ids_order.append(grouped_examples[key].pop(0))
                if not grouped_examples[key]:
                    del grouped_examples[key]

        return self._order(examples, ids_order[::-1])


class SimilarityOrdering(Ordering):
    def __init__(self, keys, reverse=False, device="cuda"):
        super().__init__(keys)

        checkpoint = "openai/clip-vit-large-patch14"
        self.model = CLIPModel.from_pretrained(checkpoint).to(device)
        self.processor = CLIPProcessor.from_pretrained(checkpoint)

        self.model.eval()
        self.device = device

        self.reverse = reverse

        self.keys = [keys] if isinstance(keys, str) else keys

    def _image_encoder(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        return self.model.get_image_features(**inputs)

    def _text_encoder(self, texts):
        inputs = self.processor(text=texts, padding=True, return_tensors="pt").to(
            self.device
        )
        return self.model.get_text_features(**inputs)

    def _keys_encoder(self, samples):
        features = []

        for key in self.keys:
            # check if samples[key] is a str of a list of str
            if isinstance(
                samples[key][0] if isinstance(samples[key], list) else samples[key], str
            ):

                features.append(self._text_encoder(samples[key]).unsqueeze(0))
            else:
                features.append(self._image_encoder(samples[key]).unsqueeze(0))

        return torch.cat(features, dim=0).mean(dim=0)

    def _inner(self, examples, query):
        # encode all examples
        encoded_examples = self._keys_encoder(examples)
        encoded_query = self._keys_encoder(query)

        # compute similarity
        similarities = torch.cosine_similarity(encoded_examples, encoded_query, dim=-1)

        return self._order(
            examples, torch.argsort(similarities, descending=self.reverse, dim=-1)
        )
