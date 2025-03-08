from typing import List, Dict, Union

from PIL import Image
import torch
from einops import repeat
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
import open_clip

from open_flamingo.src.flamingo import Flamingo
from open_flamingo.src.flamingo_lm import FlamingoLMMixin
from open_flamingo.src.utils import extend_instance
from open_flamingo.src.factory import _infer_decoder_layers_attr_name


# from open_flamingo.src.factory import create_model_and_transforms
from open_flamingo.eval.utils import unwrap_model, get_autocast, get_cast_dtype
from transformers.modeling_outputs import CausalLMOutputWithPast
import os
from huggingface_hub import hf_hub_download
import wandb


class OpenFlamingoWrapper:
    """OpenFlamingo model evaluation.

    Attributes:
      model (nn.Module): Underlying Torch model.
      tokenizer (transformers.PreTrainedTokenizer): Tokenizer for model.
      device: Index of GPU to use, or the string "CPU"
    """

    def __init__(
        self,
        clip_vision_encoder_path,
        clip_vision_encoder_pretrained,
        lang_encoder_path,
        tokenizer_path,
        cross_attn_every_n_layers,
        checkpoint,
        **kwargs,
    ):
        self.device = "cuda"  # todo accelerator
        (
            self.model,
            self.image_processor,
            self.tokenizer,
        ) = create_model_and_transforms(
            clip_vision_encoder_path=clip_vision_encoder_path,
            clip_vision_encoder_pretrained=clip_vision_encoder_pretrained,
            lang_encoder_path=lang_encoder_path,
            tokenizer_path=tokenizer_path,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            cache_dir=os.path.join(os.environ["HF_HOME"], "hub")
            if "HF_HOME" in os.environ
            else None,
        )

        checkpoint_path = hf_hub_download(checkpoint, "checkpoint.pt")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint, strict=False)
        # self.model.load_state_dict(torch.load(checkpoint_path), strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer.padding_side = "left"

        self.lm_name = lang_encoder_path.split("/")[-1]

        # autocast bf16 if gpu support else f16
        precision = "amp_bf16"  # if torch.cuda.is_bf16_supported() else "amp_f16"
        print(f"Using {precision} precision")
        self.autocast = get_autocast(precision)
        self.cast_dtype = get_cast_dtype(precision)

        self.tasks = {
            "vqa": self.get_vqa_prompt,
            "caption": self.get_caption_prompt,
            "imagenet": self.get_imagenet_prompt,
            "hateful_memes": self.get_hateful_memes_prompt,
            "sentiment": self.get_sst2_prompt,
        }

    def _prepare_images(self, batch: List[List[Image.Image]]) -> torch.Tensor:
        """
        Convert images to tensors, reshape them, and stack them.
        Args:
            batch: A list of lists of images.
        Returns:
            preprocessed images (tensors) or None
                shape (B, T_img, F, C, H, W)
                None if no images in batch
        """
        images_per_example = max(len(x) for x in batch)
        batch_images = None
        for iexample, example in enumerate(batch):
            for iimage, image in enumerate(example):
                preprocessed = self.image_processor(image)
                if batch_images is None:
                    batch_images = torch.zeros(
                        (len(batch), images_per_example, 1) + preprocessed.shape,
                        dtype=preprocessed.dtype,
                    )
                batch_images[iexample, iimage, 0] = preprocessed
        if batch_images is not None:
            batch_images = batch_images.to(
                self.device, dtype=self.cast_dtype, non_blocking=True
            )
        return batch_images

    def _prepare_text(
        self,
        batch: List[List[str]],
        padding="longest",
        truncation=True,
        max_length=2000,
    ):
        """
        Tokenize the text and stack them.
        Args:
            batch: A list of lists of strings.
        Returns:
            input_ids (tensor)
                shape (B, T_txt)
            attention_mask (tensor)
                shape (B, T_txt)
        """
        encodings = self.tokenizer(
            batch,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
            max_length=max_length,
        )
        input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
        input_ids = input_ids.to(self.device, dtype=self.cast_dtype, non_blocking=True)
        attention_mask = attention_mask.to(
            self.device, dtype=self.cast_dtype, non_blocking=True
        )
        return input_ids, attention_mask.bool()

    def generate(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        config: Dict,
        verbose: bool = False,
    ) -> List[str]:
        """
        Get generation outputs.
        """
        # if batch images none, provide empty images
        if len(batch_images) == 0:
            batch_images = [[Image.new("RGB", (1, 1))]]

        batch_images = self._prepare_images(batch_images)
        input_ids, attention_mask = self._prepare_text(batch_text)

        # Explicitly set pad_token_id in the config
        config["pad_token_id"] = self.tokenizer.eos_token_id

        self.tokenizer(
            ["<|endofchunk|>", "\n", "Question"], return_tensors="pt"
        ).input_ids.flatten().tolist() + [self.tokenizer.eos_token_id]
        # exit_condition = self.tokenizer(["<|endofchunk|>"], return_tensors="pt").input_ids.flatten().tolist() # noqa
        # self.tokenizer.eos_token_id = exit_condition
        # print(self.tokenizer.eos_token_id)

        with torch.inference_mode():
            with self.autocast():
                outputs = unwrap_model(self.model).generate(
                    batch_images,
                    input_ids,
                    attention_mask,
                    # eos_token_id=exit_condition,
                    **config,
                )

        if verbose:
            print("Generated text:")
            for i, g in enumerate(
                self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
            ):
                print(f"{i}:\n{g}\n")
                try:
                    a = wandb.Table(columns=["generated"], data=[[g]])
                    wandb.log({"generated": a})
                except wandb.errors.Error:
                    pass
                break

        # Extract only the new gnerated tokens
        outputs = outputs[:, len(input_ids[0]) :]

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def get_rank_classifications(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        all_class_names: List[str],
        use_cache: bool,
        normalize_length: bool,
    ):
        """
        Returns a (B, |all_class_names|) tensor containing the logprobs for each class
        name.
        """
        batch_images = self._prepare_images(batch_images)
        ctx_input_ids, ctx_attention_mask = self._prepare_text(batch_text)

        # Cache the context
        if use_cache:
            # reserve the last token in the context for the main forward pass
            self.cache_media(
                input_ids=ctx_input_ids,
                vision_x=batch_images,
            )
            precomputed = self.__call__(
                vision_x=None,
                lang_x=ctx_input_ids,
                attention_mask=ctx_attention_mask,
                clear_conditioned_layers=False,
                use_cache=True,
            )
            precomputed_logits = precomputed.logits
            precomputed_pkvs = precomputed.past_key_values
        else:
            precomputed_pkvs = None

        # Loop through class names and get log-likelihoods
        # Note: if all classnames are one token, this code is redundant, since we could
        # get all logits after one pass. However, if there are multi-token classnames,
        # we need to loop through each classname separately.
        overall_probs = []
        for class_name in all_class_names:
            # Tokenize only the class name
            classname_tokens = self.tokenizer(
                class_name, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(self.device)
            assert classname_tokens.ndim == 2
            classname_tokens = repeat(
                classname_tokens, "b s -> (repeat b) s", repeat=len(batch_text)
            )
            num_tokens_in_classname = classname_tokens.shape[1]

            # Concatenate the class name tokens
            if not use_cache:
                _lang_x = torch.cat([ctx_input_ids, classname_tokens], dim=1)
                _attention_mask = torch.cat(
                    [
                        ctx_attention_mask,
                        torch.ones_like(classname_tokens).bool(),
                    ],
                    dim=1,
                )
                _vision_x = batch_images
            else:
                _lang_x = classname_tokens
                _attention_mask = None
                _vision_x = None

            # Call forward to get the logits
            outputs = self.__call__(
                vision_x=_vision_x,
                lang_x=_lang_x,
                attention_mask=_attention_mask,
                clear_conditioned_layers=(not use_cache),
                past_key_values=precomputed_pkvs,
            )

            # Get the logits of the classname
            # logits shape is either (B, num_tokens_in_classname, vocab_len) with
            # use_cache
            # or (B, len(_lang_x), vocab_len) without use_cache
            # remember that the logits at index t on dim 1 correspond to predictions
            # for the t+1st token
            logits = outputs.logits
            if use_cache:
                logits = torch.cat([precomputed_logits, logits], dim=1)

            logprobs = torch.log_softmax(logits, dim=-1)
            gen_probs = logprobs[
                :, -num_tokens_in_classname - 1 : -1, :
            ]  # (B, num_tokens_in_classname, vocab_len)
            gen_probs = torch.gather(
                gen_probs, 2, classname_tokens[:, :, None]
            ).squeeze(-1)

            # Aggregate over tokens in the classname
            if normalize_length:
                class_prob = torch.mean(gen_probs, dim=1)
            else:
                class_prob = torch.sum(gen_probs, dim=1)
            overall_probs.append(class_prob)  # (B, 1)

        self.uncache_media()
        overall_probs = torch.vstack(overall_probs).T.cpu()  # shape (B, num_classes)
        return overall_probs

    def __call__(
        self,
        lang_x: torch.Tensor,
        vision_x: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: torch.Tensor = None,
        clear_conditioned_layers: bool = False,
        use_cache: bool = False,
    ):
        """
        Calls the forward function of the model.
        Special logic to handle the case if past_key_values is not None:
            then lang_x is assumed to contain the tokens to be generated
            *excluding* the tokens already in past_key_values.
            We then repeatedly call forward, updating the past_key_values.
        """
        # standard forward pass
        if past_key_values is None:
            with torch.inference_mode():
                with self.autocast():
                    outputs = self.model(
                        vision_x=vision_x,
                        lang_x=lang_x,
                        attention_mask=attention_mask,
                        clear_conditioned_layers=clear_conditioned_layers,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                    )
            return outputs

        # loop to handle updating past_key_values
        logits = []
        for token_idx in range(lang_x.shape[1]):
            _lang_x = lang_x[:, token_idx].reshape((-1, 1))
            if attention_mask is not None:
                _attention_mask = attention_mask[:, token_idx].reshape((-1, 1))
            else:
                _attention_mask = None

            with torch.inference_mode():
                with self.autocast():
                    outputs = self.model(
                        vision_x=vision_x,
                        lang_x=_lang_x,
                        attention_mask=_attention_mask,
                        clear_conditioned_layers=False,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

            past_key_values = outputs.past_key_values
            logits.append(outputs.logits)

        logits = torch.cat(logits, dim=1)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
        )

    def encode_vision_x(self, image_tensor: torch.Tensor):
        unwrap_model(self.model)._encode_vision_x(image_tensor.to(self.device))

    def uncache_media(self):
        unwrap_model(self.model).uncache_media()

    def cache_media(self, input_ids, vision_x):
        unwrap_model(self.model).cache_media(input_ids=input_ids, vision_x=vision_x)

    def get_vqa_prompt(self, sample) -> str:
        question, answer = sample["question"], sample["label"]
        return f"<image>Question:{question} Short answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"  # noqa

    def get_caption_prompt(self, sample) -> str:
        caption = sample["label"]
        return f"<image>Output:{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"  # noqa

    def get_imagenet_prompt(self, sample) -> str:
        label = sample["label"]
        return f"<image>Output:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"  # noqa

    def get_hateful_memes_prompt(self, sample) -> str:
        text, label = sample["image"], sample["label"]
        return f"<image>is an image with: '{text}' written on it. Is it hateful? Answer:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"  # noqa

    def get_sst2_prompt(self, sample) -> str:
        text, label = sample["text"], sample["label"]
        return f"Review: {text} Sentiment: {label + '<|endofchunk|>' if label is not None else ''}"  # noqa

    def prompt(
        self,
        examples: Dict[str, Union[List[int], List[Image.Image], List[str]]],
        queries: Dict[str, Union[int, Image.Image, str]],
        # task: str,
        prompt_func,
    ):
        prompts = []
        images = []
        examples, queries = examples.copy(), queries.copy()
        for example, query in zip(examples, queries):
            try:
                images.append(example["image"] + [query["image"]])
            except KeyError:
                pass
            # convert dict of lists to list of dicts
            example = [
                {k: v[i] for k, v in example.items()}
                for i in range(len(example["label"]))
            ]
            query["label"] = None
            one = []
            for e in example:
                # one.append(self.tasks[task](e) + "\n")
                e["image"] = "<image>" if e["image"] else ""
                one.append(
                    "".join(prompt_func(**e)).rstrip("\n") + " <|endofchunk|> \n"
                )
            # one.append(self.tasks[task](query))
            query["image"] = "<image>" if query["image"] else ""
            one.append("".join(prompt_func(**query, hide_label=True)))

            prompts.append("".join(one))

            # remove all "" from images
            images = [[x for x in y if x] for y in images]
        return {"batch_text": prompts, "batch_images": images}

    def extract_answer(self, text):
        return (
            text.split("<|endofchunk|>")[0]
            .split("<|endoftext|>")[0]
            .split("\n")[0]
            .strip()
            .split("Caption:")[0]
            .split("Question:")[0]
            .split("Image:")[0]
            .strip()
        )


def create_model_and_transforms(
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    lang_encoder_path: str,
    tokenizer_path: str,
    cross_attn_every_n_layers: int = 1,
    use_local_files: bool = False,
    decoder_layers_attr_name: str = None,
    freeze_lm_embeddings: bool = False,
    cache_dir: Optional[str] = None,
    **flamingo_kwargs,
):
    """
    Initialize a Flamingo model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        clip_vision_encoder_path (str): path to pretrained clip model (e.g. "ViT-B-32")
        clip_vision_encoder_pretrained (str): name of pretraining dataset for clip model
         (e.g. "laion2b_s32b_b79k")
        lang_encoder_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        cross_attn_every_n_layers (int, optional): determines how often to add a
        cross-attention layer. Defaults to 1.
        use_local_files (bool, optional): whether to use local files. Defaults to False.
        decoder_layers_attr_name (str, optional): name of the decoder layers attribute.
        Defaults to None.
        freeze_lm_embeddings (bool, optional): whether to freeze LM input embeddings
        when configuring Perceiver.
        cache_dir (str, optional): path to cache directory for downloading OpenClip/HF
        weights.
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path,
        pretrained=clip_vision_encoder_pretrained,
        cache_dir=cache_dir,
        device="cuda",
    )
    # set the vision encoder to output the visual features
    vision_encoder.visual.output_tokens = True

    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    # add Flamingo special tokens to the tokenizer
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if text_tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    print("loading lang encoder")
    lang_encoder = AutoModelForCausalLM.from_pretrained(
        lang_encoder_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
        # these two lines are custom added to load model in cluster without fail
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        init_device="meta",
    )
    lang_encoder.to("cuda")

    print("loaded lang encoder")

    # hacks for MPT-1B, which doesn't have a get_input_embeddings method
    if "mpt-1b-redpajama-200b" in lang_encoder_path:

        class EmbeddingFnMixin:
            def get_input_embeddings(self):
                return self.transformer.wte

            def set_input_embeddings(self, new_embeddings):
                self.transformer.wte = new_embeddings

        extend_instance(lang_encoder, EmbeddingFnMixin)

    # convert LM to FlamingoLM
    extend_instance(lang_encoder, FlamingoLMMixin)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))

    model = Flamingo(
        vision_encoder,
        lang_encoder,
        text_tokenizer.encode("<|endofchunk|>")[-1],
        text_tokenizer.encode("<image>")[-1],
        vis_dim=open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"][
            "width"
        ],
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        **flamingo_kwargs,
    )

    # Freeze all parameters
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
    model.perceiver.requires_grad_(True)
    model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
    if not freeze_lm_embeddings:
        model.lang_encoder.get_input_embeddings().requires_grad_(True)
        # TODO: investigate also training the output embeddings when untied

    print(
        f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"  # noqa
    )

    return model, image_processor, text_tokenizer
