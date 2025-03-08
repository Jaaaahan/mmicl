from .majority_voting import MajorityVoting
from .idefics import IdeficsWrapper
from .open_flamingo import OpenFlamingoWrapper

MODELS = {
    "majority": {
        "model": MajorityVoting,
        "args": {},
    },
    "idefics_debug": {
        "model": IdeficsWrapper,
        "args": {
            "instruct": False,
            "checkpoint": "HuggingFaceM4/tiny-random-idefics",
        },
    },
    "idefics_9b_instruct": {
        "model": IdeficsWrapper,
        "args": {
            "instruct": True,
            "checkpoint": "HuggingFaceM4/idefics-9b-instruct",
        },
    },
    "idefics_9b_base": {
        "model": IdeficsWrapper,
        "args": {
            "instruct": False,
            "checkpoint": "HuggingFaceM4/idefics-9b",
        },
    },
    "idefics_80b_base": {
        "model": IdeficsWrapper,
        "args": {
            "instruct": False,
            "checkpoint": "HuggingFaceM4/idefics-80b",
        },
    },
    "open_flamingo_9b": {
        "model": OpenFlamingoWrapper,
        "args": {
            "clip_vision_encoder_path": "ViT-L-14",
            "clip_vision_encoder_pretrained": "openai",
            "lang_encoder_path": "anas-awadalla/mpt-7b",
            "tokenizer_path": "anas-awadalla/mpt-7b",
            "cross_attn_every_n_layers": 4,
            "checkpoint": "openflamingo/OpenFlamingo-9B-vitl-mpt7b",
        },
    },
    "open_flamingo_3b": {
        "model": OpenFlamingoWrapper,
        "args": {
            "clip_vision_encoder_path": "ViT-L-14",
            "clip_vision_encoder_pretrained": "openai",
            "lang_encoder_path": "anas-awadalla/mpt-1b-redpajama-200b",
            "tokenizer_path": "anas-awadalla/mpt-1b-redpajama-200b",
            "cross_attn_every_n_layers": 1,
            "checkpoint": "openflamingo/OpenFlamingo-3B-vitl-mpt1b",
        },
    },
}
