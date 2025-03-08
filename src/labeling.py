import random
import copy
import torch


def random_label(examples, words, key="label"):
    """
    Randomly select a ground truth word from the list of words for each label
    :param labels:
    :param words:
    :return:
    """
    if isinstance(words, dict):
        words = list(words.keys())
    for e in examples:
        e[key] = random.choices(words, k=len(e[key]))
    return examples


def random_qa(examples, qa):
    """
    Randomly select a ground truth word from the list of words for each label
    :param labels:
    :param words:
    :return:
    """
    for e in examples:
        e["question"], e["label"] = zip(*random.choices(qa, k=len(e["question"])))
    return examples


def random_image(examples, images):
    """
    Randomly select a ground truth word from the list of words for each label
    :param labels:
    :param words:
    :return:
    """
    for e in examples:
        e["image"] = random.sample(images, k=len(e["image"]))
    return examples


def swap_label(examples, mappings):
    """
    Swap the label with the corresponding mapping
    :param examples:
    :param mappings:
    :return:
    """
    for e in examples:
        e["label"] = [mappings[label] for label in e["label"]]
    return examples


def inverse_mapping(labels):
    assert len(labels) == 2
    return {labels[0]: labels[1], labels[1]: labels[0]}


def random_mapping(labels):
    shuffled = copy.deepcopy(labels)
    random.shuffle(shuffled)
    return {labels[i]: shuffled[i] for i in range(len(labels))}


def auto_mapping(labels):
    if len(labels) == 2:
        return inverse_mapping(labels)
    return random_mapping(labels)


def black_images(batch):
    """
    Set the images to black
    :param examples:
    :return:
    """
    for examples in batch:
        for e in examples:
            e["image"] = torch.zeros_like(e["image"])
    return examples


def no_ocr(examples):
    for e in examples:
        e["ocr"] = ["" for _ in e["ocr"]]
    return examples


def no_question(examples):
    for e in examples:
        e["question"] = ["" for _ in e["question"]]
    return examples


def no_image(examples):
    for e in examples:
        e["image"] = ["" for _ in e["image"]]
    return examples


def no_image_question(examples):
    for e in examples:
        e["image"] = ["" for _ in e["image"]]
        e["question"] = ["" for _ in e["question"]]
    return examples


def random_question(examples, questions):
    for e in examples:
        e["question"] = random.choices(questions, k=len(e["question"]))
    return examples
