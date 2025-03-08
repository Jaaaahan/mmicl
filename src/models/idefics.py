from typing import Dict
from transformers import IdeficsForVisionText2Text, AutoProcessor, AutoModelForCausalLM
import torch


class IdeficsWrapper:
    def __init__(self, instruct: bool, checkpoint: str = None):
        self.checkpoint = checkpoint

        if self.checkpoint:
            self.processor = AutoProcessor.from_pretrained(self.checkpoint)
            self.model = IdeficsForVisionText2Text.from_pretrained(
                self.checkpoint, torch_dtype=torch.bfloat16, device_map="auto"
            )
            print(f"device_map: {self.model.hf_device_map}")
            self.model = torch.compile(
                self.model, mode="reduce-overhead", fullgraph=True
            )

        self.instruct = instruct

        self.user = ["User: Given the following examples: \n"]
        self.assistant = ["\nAssistant: "]

    @torch.no_grad()
    def generate(self, prompts, config: Dict, verbose: bool = False):
        inputs = self.processor(
            prompts, return_tensors="pt", add_end_of_utterance_token=False
        ).to(self.model.device)

        bad_words_ids = self.processor.tokenizer(
            ["<image>", "<fake_token_around_image>"], add_special_tokens=False
        ).input_ids
        # todo newline token shorter output in not instruct but < appears ad end of
        #  output, must in this case remove it
        exit_condition = self.processor.tokenizer(
            "<end_of_utterance>",
            "\n",
            "<",
            add_special_tokens=False
        ).input_ids

        generated_ids = self.model.generate(
            **inputs, bad_words_ids=bad_words_ids, eos_token_id=exit_condition, **config
        )

        if verbose:
            print("Generated text:")
            for i, g in enumerate(
                self.processor.batch_decode(generated_ids, skip_special_tokens=False)
            ):
                print(f"{i}:\n{g}\n")
                break

        # take only new tokens
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return generated_text

    @torch.no_grad()
    def prompt(self, examples, queries, prompt_function):
        res = []
        for e, q in zip(examples, queries):
            prompt = []
            for items in zip(*e.values()):
                # Create a temporary dictionary to hold the current set of items
                kwargs = dict(zip(e.keys(), items))
                prompt += prompt_function(**kwargs)

            if self.instruct:
                raise NotImplementedError
            else:
                prompt += prompt_function(**q, hide_label=True)

            res.append(prompt)
        return {"prompts": res}

    def image_output_prompt(self, image, label, hide_label=False, **kwargs):
        label = "" if hide_label else f"{label} \n"
        return [image, f" Output: {label}"]

    def text_output_prompt(self, text, label, hide_label=False, **kwargs):
        label = "" if hide_label else f"{label} \n"
        return [f"{text.strip()} Output: {label}"]

    def extract_answer(self, text):
        if self.instruct:
            extracted = text.split("<end_of_utterance>")[0]
        else:
            extracted = text.split("\n")[0].replace("<", "")
        return extracted.strip()
