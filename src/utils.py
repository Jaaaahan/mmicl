import torch
import random
import tempfile
import json


class Subset(torch.utils.data.Subset):
    """
    A subclass of torch.utils.data.Subset to handle subsets of datasets in PyTorch.
    Provides additional functionalities to work with the subset of data.

    Attributes:
        dataset (Dataset): The whole dataset that the subset is part of.
        indices (list): The indices in the dataset that define the subset.
    """

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        """Retrieve an item by index from the subset."""
        return self.dataset[int(self.indices[idx])]

    def __len__(self):
        """Return the number of items in the subset."""
        return len(self.indices)

    def score(self, result_path):
        """Evaluate the subset with a scoring function defined in the dataset."""
        return self.dataset.score(result_path)

    def result(self, idx, answer):
        """Submit a result for scoring based on dataset specific functionality."""
        return self.dataset.result(idx, answer)

    def get_labels(self):
        """Get the labels for the subset items, assuming labels exist in the dataset."""
        return self.dataset.get_labels()


def custom_collate_fn(batch):
    """
    Collate function for DataLoader that collates a list of dicts into a dict of lists.
    This is useful for processing batches of data where each item of the batch is a
    dictionary.

    Parameters:
        batch (list of dict): The batch to collate.

    Returns:
        dict: A dictionary of lists, where each key has a list of values corresponding
        to the batch data.
    """
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch


def set_deterministic(seed=0):
    """
    Set the random seeds for reproducibility in PyTorch and Python's random module.

    Parameters:
        seed (int): The seed value to use for random number generators. Default is 0.
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)


def save_to_tmp_json(results):
    """
    Save the provided results to a temporary JSON file and return the file path.

    Parameters:
        results (any): The data to serialize to JSON.

    Returns:
        str: The file path to the temporary file containing the results.
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        json.dump(results, f)
        return f.name
