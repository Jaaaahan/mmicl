from typing import Dict

import numpy as np
from torch.nn import Module

from collections import defaultdict


def weighted_majority_vote(answers):
    # Create a defaultdict to store the cumulative weights for each answer
    answer_weights = defaultdict(int)

    # Iterate through the answers and their weights
    for answer, weight in answers:
        answer_weights[answer] += weight

    # Find the answer(s) with the highest cumulative weight(s)
    max_weight = max(answer_weights.values())
    majority_answers = [
        answer for answer, weight in answer_weights.items() if weight == max_weight
    ]

    return majority_answers[0]


class MajorityVoting:
    def __init__(
        self,
    ):
        self.model = Module()
        self.model.config = {}

    def generate(self, prompts, config: Dict, verbose: bool = False):
        res = []
        for i, e in enumerate(prompts):
            labels = e["label"]
            similarities = e["similarity"]

            # exp similarity
            similarities = [np.exp(sim) for sim in similarities]

            majority = weighted_majority_vote(zip(labels, similarities))
            res.append(majority)

            if i == 0 and verbose:
                # if verbose:
                print(
                    f"Label: {labels}, Similarity: {similarities}, Majority: {majority}"
                )

        return res

    def prompt(self, examples, queries, task):
        return {"prompts": examples}

    def extract_answer(self, text):
        return text
