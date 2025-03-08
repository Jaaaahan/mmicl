from torch.utils.data import Dataset
import random
from tqdm import tqdm
import torch
from utils import custom_collate_fn
import os
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoModel, AutoProcessor
from collections import defaultdict
from torch.utils.data import Subset
import einops


class NoneSampler(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            "examples": {k: [] for k in self.dataset[idx].keys()},
            "query": self.dataset[idx],
        }


class EmptySampler(Dataset):
    def __init__(self, dataset, support_set, num_shots=1):
        self.dataset = dataset
        self.num_shots = num_shots
        self.support_set = support_set

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # sample num_shots index from dataset
        idxs = random.sample(range(len(self.support_set)), self.num_shots)
        items = [self.support_set[i] for i in idxs]

        # convert list of dicts to dict of lists
        collated_batch = {}
        for key in items[0].keys():
            if key in ["image", "text"]:
                collated_batch[key] = ["" for _ in items]
            else:
                collated_batch[key] = [item[key] for item in items]
        return {
            "examples": collated_batch,
            "query": self.dataset[idx],
        }


class RandomSampler(Dataset):
    def __init__(self, dataset, support_set, num_shots=1):
        self.dataset = dataset
        self.support_set = support_set
        self.num_shots = num_shots

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # sample num_shots index from dataset
        idxs = random.sample(range(len(self.support_set)), self.num_shots)
        # items = [self.support_set[i] for i in idxs]
        with ThreadPoolExecutor(max_workers=self.num_shots) as executor:
            items = list(executor.map(lambda i: self.support_set[i], idxs))

        # convert list of dicts to dict of lists
        collated_batch = {}
        for key in items[0].keys():
            collated_batch[key] = [item[key] for item in items]
        return {
            "examples": collated_batch,
            "query": self.dataset[idx],
        }


class BalancedSampler(Dataset):
    def __init__(self, dataset, support_set, num_shots=1, choice=None):
        assert num_shots % 2 == 0, "num_shots must be even"

        self.dataset = dataset
        self.support_set = support_set
        self.num_shots = num_shots

        # obtain all different labels in oneline
        if isinstance(self.dataset, Subset):
            self.labels = self.dataset.dataset.get_labels()
        else:
            self.labels = self.dataset.get_labels()

        # obtain o dict of list of indices for each label
        self.label2idxs = {}
        for idx, item in enumerate(self.support_set):
            label = item["label"]
            self.label2idxs.setdefault(label, []).append(idx)

        self.choice = self.labels if choice is None else choice
        if choice is None:
            print("WARNING: using all labels for balanced sampling")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # for each label select num_shots // len(labels)
        # indices and add to list random order
        # todo
        assert (
            self.num_shots % len(self.choice) == 0
        ), "num_shots must be divisible by len(choice)"
        assert (
            len(self.choice) <= self.num_shots
        ), "num_shots must be greater than number of choice of labels"
        idxs = []
        for label in self.choice:
            idxs.extend(
                random.sample(
                    self.label2idxs[label], k=self.num_shots // len(self.choice)
                )
            )
        random.shuffle(idxs)

        items = [self.support_set[i] for i in idxs]

        # convert list of dicts to dict of lists
        collated_batch = {}
        for key in items[0].keys():
            collated_batch[key] = [item[key] for item in items]
        return {
            "examples": collated_batch,
            "query": self.dataset[idx],
        }


class RicesSampler:
    def __init__(
        self,
        dataloader,
        support_set,
        num_shots=3,
        cached_features_path=None,
        device="cuda",
        keys=None,
        return_similarity=False,
        reduce="mean",
        reverse=False,
        checkpoint="openai/clip-vit-large-patch14",
        **kwargs,
    ):
        self.num_shots = num_shots
        assert keys is not None, "keys must be specified"
        self.keys = keys
        self.dataloader = dataloader

        self.support_set = support_set
        self.device = device
        self.batch_size = 128

        self.model = AutoModel.from_pretrained(checkpoint).to(device)
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.model.eval()

        try:
            print(
                "Rices text max length:", self.processor.tokenizer.max_model_input_sizes
            )
        except AttributeError:
            pass
        print("Cached features path:", cached_features_path)
        # Precompute features
        cached_features = (
            torch.load(cached_features_path)
            if cached_features_path and os.path.exists(cached_features_path)
            else None
        )
        if cached_features is None:
            self.features = self._precompute_features()
            if cached_features_path:
                torch.save(self.features, cached_features_path)
        else:
            self.features = cached_features

        self.return_similarity = return_similarity
        self.reduce = reduce
        self.reverse = reverse

    def _compute_features(self, batch):
        f = defaultdict(list)
        for key in self.keys:
            data = batch[key]

            if isinstance(data[0], str):
                inputs = self.processor(
                    text=data, padding=True, return_tensors="pt", truncation=True
                ).to(self.device)
                features = self.model.get_text_features(**inputs)
            else:
                inputs = self.processor(images=data, return_tensors="pt").to(
                    self.device
                )
                features = self.model.get_image_features(**inputs)

            features /= features.norm(dim=-1, keepdim=True)
            f[key] = features.detach().cpu()

        return f

    def _precompute_features(self):
        all_features = defaultdict(list)

        # Switch to evaluation mode
        self.model.eval()

        # Set up loader
        loader = torch.utils.data.DataLoader(
            self.support_set,
            batch_size=self.batch_size,
            collate_fn=custom_collate_fn,
            num_workers=6,
        )

        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc="Precomputing features for RICES",
            ):
                features = self._compute_features(batch)
                for key in self.keys:
                    all_features[key].append(features[key])

        for key in self.keys:
            all_features[key] = torch.cat(all_features[key])

        return all_features

    def find(self, batch, num_examples):
        """
        Get the top num_examples most similar examples to the images.
        """
        with torch.no_grad():
            query_feature = self._compute_features(batch)

            similarity = {}
            for key in self.keys:
                if query_feature[key].ndim == 1:
                    query_feature[key] = query_feature[key].unsqueeze(0)

                similarity[key] = (query_feature[key] @ self.features[key].T).squeeze()

                if similarity[key].ndim == 1:
                    similarity[key] = similarity[key].unsqueeze(0)

            # similarity is average of all keys
            similarity = torch.stack(
                [similarity[key] for key in self.keys]
            )  # .mean(dim=0)
            similarity = einops.reduce(
                similarity, "keys batch set -> batch set", self.reduce
            )

            # Get the indices of the 'num_examples' most similar images
            indices = similarity.argsort(dim=-1, descending=True)[:, :num_examples]

            if self.reverse:
                indices = indices.flip(dims=[1])

        # Return with the most similar images last
        # return [[self.dataset[i] for i in reversed(row)] for row in indices]
        with ThreadPoolExecutor(max_workers=num_examples) as executor:
            examples = list(
                executor.map(
                    lambda row: [self.support_set[i] for i in reversed(row)], indices
                )
            )

        if self.return_similarity:
            sim = similarity.sort(dim=-1, descending=False)[0][:, -num_examples:]
            examples = [
                [e | {"similarity": s} for e, s in zip(example, similar)]
                for example, similar in zip(examples, sim.tolist())
            ]

        return examples

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            examples = self.find(batch, self.num_shots)

            examples = [{k: [d[k] for d in e] for k in e[0].keys()} for e in examples]
            queries = [dict(zip(batch, t)) for t in zip(*batch.values())]

            yield {
                "examples": examples,
                "query": queries,
            }


class MMICES(RicesSampler):
    def __init__(
        self,
        dataloader,
        support_set,
        num_shots=3,
        cached_features_path=None,
        device="cuda",
        keys=None,
        return_similarity=False,
        **kwargs,
    ):
        super().__init__(
            dataloader,
            support_set,
            num_shots=num_shots,
            cached_features_path=cached_features_path,
            device=device,
            keys=keys,
            return_similarity=return_similarity,
            **kwargs,
        )

        assert len(self.keys) == 2, "MMICES requires exactly 2 keys"
        self.prefilter = 200

    def find(self, batch, num_examples):
        """
        Get the top num_examples most similar examples to the images.
        """
        with torch.no_grad():
            query_feature = self._compute_features(batch)

            similarity = {}
            for key in self.keys:

                if query_feature[key].ndim == 1:
                    query_feature[key] = query_feature[key].unsqueeze(0)

                similarity[key] = (query_feature[key] @ self.features[key].T).squeeze()

                if similarity[key].ndim == 1:
                    similarity[key] = similarity[key].unsqueeze(0)

            prefilter_indices = similarity[self.keys[0]].argsort(
                dim=-1, descending=True
            )[:, self.prefilter + 1 :]
            similarity = similarity[self.keys[1]]
            for batch_idx, indices in enumerate(prefilter_indices):
                similarity[batch_idx, indices] = -float("inf")

            # Get the indices of the 'num_examples' most similar images
            indices = similarity.argsort(dim=-1, descending=True)[:, :num_examples]

            if self.reverse:
                indices = indices.flip(dims=[1])

        # Return with the most similar images last
        # return [[self.dataset[i] for i in reversed(row)] for row in indices]
        with ThreadPoolExecutor(max_workers=num_examples) as executor:
            examples = list(
                executor.map(
                    lambda row: [self.support_set[i] for i in reversed(row)], indices
                )
            )

        if self.return_similarity:
            sim = similarity.sort(dim=-1, descending=False)[0][:, -num_examples:]
            examples = [
                [e | {"similarity": s} for e, s in zip(example, similar)]
                for example, similar in zip(examples, sim.tolist())
            ]

        return examples


class BalancedRices(RicesSampler):
    def __init__(
        self,
        dataloader,
        support_set,
        num_shots=3,
        cached_features_path=None,
        device="cuda",
        keys=None,
        return_similarity=False,
        **kwargs,
    ):
        keys.remove("rices")
        super().__init__(
            dataloader,
            support_set,
            num_shots=num_shots,
            cached_features_path=cached_features_path,
            device=device,
            keys=keys,
            return_similarity=return_similarity,
            **kwargs,
        )

    def __iter__(self):
        for batch in self.dataloader:
            examples = self.find(batch, self.num_shots * 10)

            examples = [{k: [d[k] for d in e] for k in e[0].keys()} for e in examples]

            res = []
            for e in examples:
                filtered = defaultdict(list)
                counts = defaultdict(int)
                desired_shots_per_key = self.num_shots // len(
                    self.support_set.get_labels()
                )

                for items in reversed(list(zip(*e.values()))):
                    kwargs = dict(zip(e.keys(), items))
                    key = kwargs["label"]

                    if counts[key] < desired_shots_per_key:
                        filtered[key].append(kwargs)
                        counts[key] += 1

                merged = [item for sublist in filtered.values() for item in sublist]
                if len(merged) != self.num_shots:
                    print(f"WARNING: {len(merged)} != {self.num_shots}")
                random.shuffle(merged)
                # convert list of dicts to dict of lists
                merged = merged[: self.num_shots]
                res.append({k: [d[k] for d in merged] for k in merged[0].keys()})

            queries = [dict(zip(batch, t)) for t in zip(*batch.values())]

            yield {
                "examples": res,
                "query": queries,
            }
