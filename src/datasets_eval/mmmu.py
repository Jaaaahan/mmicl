# code adapted by https://github.com/MMMU-Benchmark/MMMU/blob/main/eval
import subprocess
import sys
import os
from datasets import load_dataset, concatenate_datasets, load_from_disk
from torch.utils.data import Dataset
import re
import json
import ast
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(__file__) + "/../mmmu/eval/utils")
from eval_utils import (  # noqa
    parse_multi_choice_response,
    parse_open_response,
)
from data_utils import (  # noqa
    CAT_SHORT2LONG,
    process_single_sample,
    construct_prompt,
    get_multi_choice_info,
)


CONFIG = {
    "task_instructions": "",
    "multi_choice_example_format": "{}\n\n{}\n\nAnswer with the option's letter from the given choices directly.",  # noqa
    "short_ans_example_format": "{}\n\nAnswer with a short sentence or phrase.",
}


class MMMU(Dataset):
    def __init__(self, split="validation"):
        print(f"Loading MMMU dataset for split {split}")

        self.dataset_name = "MMMU/MMMU"
        self.split = split

        if "SAVE_DATASET" in os.environ:
            dataset_path = os.path.join(
                os.environ["SAVE_DATASET"], self.dataset_name, split
            )
            print(f"dataset_path: {dataset_path}")
            # Check if the dataset exists at the specified path
            if os.path.exists(dataset_path):
                # Load the dataset from disk
                dataset = load_from_disk(dataset_path)
            else:
                # If dataset is not found on disk, load it and save it
                dataset = self.load_full()
                dataset.save_to_disk(dataset_path)
                dataset = dataset
        else:
            dataset = self.load_full()

        self.dataset = dataset
        print(f"Loaded MMMU dataset for split {split}")

        self.config = CONFIG

    def load_full(self):
        def load_subject_dataset(subject):
            return load_dataset(self.dataset_name, subject, split=self.split)

            # Using ThreadPoolExecutor to load datasets in parallel

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Dictionary to keep track of future and subject
            future_to_subject = {
                executor.submit(load_subject_dataset, subject): subject
                for subject in CAT_SHORT2LONG.values()
            }

            sub_dataset_list = []
            # Use tqdm to display progress
            for future in tqdm(
                as_completed(future_to_subject), total=len(CAT_SHORT2LONG)
            ):
                subject = future_to_subject[future]
                try:
                    sub_dataset = future.result()
                    sub_dataset_list.append(sub_dataset)
                except Exception as e:
                    print(f"Error loading dataset for subject {subject}: {e}")

            # merge all dataset
        return concatenate_datasets(sub_dataset_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        idx = int(idx)
        sample = self.dataset[idx]
        sample = process_single_sample(sample)
        sample = construct_prompt(sample, self.config)

        # print(sample)

        # from sample 'final_input_prompt' split where image
        # and generate list of text and image interleaved
        prompt = sample["final_input_prompt"]
        # regex to replace <image 1> with <image> and <image 2> with <image>
        prompt = re.sub("<image [0-9]>", "<image>", prompt)
        # split on <image>
        prompt = prompt.split("<image>")

        image = (
            sample["image"] if isinstance(sample["image"], list) else [sample["image"]]
        )
        # add interleaved image and text
        prompt_interleaved = []
        for i in range(len(prompt)):
            prompt_interleaved.append(prompt[i])
            if i < len(image):
                prompt_interleaved.append(image[i])

        result = {
            "id": sample["id"],
            "prompt": prompt_interleaved,
            "label": sample["gt_content"],
            "prompt_label": sample["gt_content"],
            "question_type": sample["question_type"],
            "question": sample["question"],
            "image": image[0],
            # Initialize optional keys with None or empty structures
            "options": None,
            "index2ans": None,
        }

        if sample["question_type"] == "multiple-choice":
            # Update keys specific to multiple-choice questions
            result.update(
                {
                    "options": sample["options"],
                    "index2ans": sample["index2ans"],
                    "prompt_label": sample["correct_choice"],
                    # Overwrite label for multiple-choice
                }
            )

        return result

    @staticmethod
    def result(sample, response):
        # remove initial and trailing spaces

        response = response.strip()
        if sample["question_type"] == "multiple-choice":
            all_choices = get_multi_choice_info(sample["options"])[1]
            pred_ans = parse_multi_choice_response(
                response, all_choices, sample["index2ans"]
            )
        else:  # open question
            pred_ans = response

        return {sample["id"]: pred_ans}

    def score(self, result_path):
        with open(result_path, "r") as f:
            results = json.load(f)

        # merge all dicts
        merged_results = {}
        for d in results:
            merged_results.update(d)

        with open(result_path, "w") as f:
            json.dump(merged_results, f)

        command = [
            "python",
            os.path.dirname(__file__) + "/../mmmu/eval/main_eval_only.py",
            "--output_path",
            result_path,
            "--answer_path",
            os.path.dirname(__file__) + "/../mmmu/eval/answer_dict_val.json",
        ]
        print(" ".join(command))
        res = subprocess.run(command, capture_output=True, text=True)

        matches = re.search(r"\{.*\}", res.stdout)
        extracted_dict = ast.literal_eval(matches[0])
        print(extracted_dict)
        return extracted_dict["Overall"]["acc"] * 100

    @staticmethod
    def prompt(prompt: list, prompt_label: str, hide_label: bool = False, **kwargs):
        label = "" if hide_label else f"{prompt_label} \n"

        return ["\n### Question ###"] + prompt + [f" {label}"]


class SubjectMMMU(MMMU):
    def __init__(self, split, subject):
        # run for each subject
        self.dataset = load_dataset("MMMU/MMMU", subject, split=split)
        self.config = CONFIG
