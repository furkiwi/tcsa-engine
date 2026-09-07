"""Paper constants, official Krea 2 templates, and target-module aliases."""

from __future__ import annotations

import re

# Official Qwen3-VL-4B tap used by Krea 2 (hidden_states indices, no offset).
SELECT_LAYERS: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
N_LAYERS = len(SELECT_LAYERS)
HIDDEN = 2560
DIT_WIDTH = 6144
MAX_LENGTH = 512
PREFIX_IDX = 34
SUFFIX_IDX = 5

PROMPT_TEMPLATE_ENCODE_PREFIX = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
PROMPT_TEMPLATE_ENCODE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

# Measured txtfusion.projector (1×12) from open Krea 2 weights.
# Used only when the RAW file is absent, so projector geometry stays on-manifold.
PUBLISHED_PROJECTOR = (
    -0.05,
    -0.16,
    0.37,
    0.50,
    0.71,
    0.39,
    0.40,
    -1.44,
    -0.51,
    -0.89,
    -0.61,
    0.11,
)

GATE_COSINE = 0.4
GATE_LAMBDA1 = 0.35
GATE_SPEC = 0.15
GATE_SOFT = 0.1
DEFAULT_RANK = 4
DEFAULT_ALPHA = 1.0
ALS_SWEEPS = 4

STYLE_PRESETS: dict[str, list[str]] = {
    "watercolor": [
        "watercolor",
        "watercolor painting",
        "traditional watercolor illustration",
        "hand-painted watercolor",
    ],
    "oil painting": [
        "oil painting",
        "classical oil on canvas",
        "impasto oil painting",
        "baroque oil portrait style",
    ],
    "pencil sketch": [
        "pencil sketch",
        "graphite drawing",
        "detailed pencil illustration",
        "charcoal pencil study",
    ],
    "anime style": [
        "anime style",
        "cel-shaded anime illustration",
        "modern anime key visual",
        "japanese animation still",
    ],
    "ink wash painting": [
        "ink wash painting",
        "sumi-e",
        "chinese ink wash",
        "literati ink landscape style",
    ],
    "cinematic photography": [
        "cinematic photography",
        "anamorphic cinematic still",
        "film-still cinematic lighting",
        "widescreen cinematic photograph",
    ],
}

NULLS: dict[str, list[str]] = {
    "adjective": ["delicious", "expensive", "ancient"],
    "attribute": ["red", "large", "left-handed"],
    "nonce": ["xyzzorp"],
}

EXTRACTION = [
    "a tiger walking through a jungle",
    "a stone castle on a cliff",
    "a woman standing by a window",
    "a spaceship above a desert",
    "a bowl of fruit on a table",
    "a motorcycle parked at dusk",
    "a jungle path after rain",
    "a quiet harbor at sunrise",
    "a fox in snow",
    "a violin on a wooden chair",
    "a lighthouse in fog",
    "a boy with a red umbrella",
    "a mountain temple",
    "a glass greenhouse",
    "a steaming teacup",
    "a black cat on a rooftop",
    "a sailboat on a lake",
    "an old library corridor",
    "a field of sunflowers",
    "a ceramic vase with peonies",
]

EVALUATION = [
    "a dog sitting in a garden",
    "a portrait of an old man",
    "a train crossing a bridge",
    "a wide landscape at golden hour",
    "a robot in a workshop",
    "a plate of food on linen",
    "a library interior with tall shelves",
    "a night market street",
]

# (id, diffusers_key, native_key, in_features, out_features)
# Shapes verified against Krea2TextProjection / TextFusionTransformer.
TARGETS = [
    ("projector", "text_fusion.projector", "txtfusion.projector", N_LAYERS, 1),
    ("linear_1", "txt_in.linear_1", "txtmlp.1", HIDDEN, DIT_WIDTH),
    ("linear_2", "txt_in.linear_2", "txtmlp.3", DIT_WIDTH, DIT_WIDTH),
]

RAW_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "projector": (
        "txtfusion.projector.weight",
        "text_fusion.projector.weight",
        "diffusion_model.txtfusion.projector.weight",
        "transformer.text_fusion.projector.weight",
    ),
    "linear_1": (
        "txtmlp.1.weight",
        "txt_in.linear_1.weight",
        "diffusion_model.txtmlp.1.weight",
        "transformer.txt_in.linear_1.weight",
    ),
    "linear_1_bias": (
        "txtmlp.1.bias",
        "txt_in.linear_1.bias",
        "diffusion_model.txtmlp.1.bias",
        "transformer.txt_in.linear_1.bias",
    ),
    "linear_2": (
        "txtmlp.3.weight",
        "txt_in.linear_2.weight",
        "diffusion_model.txtmlp.3.weight",
        "transformer.txt_in.linear_2.weight",
    ),
    "linear_2_bias": (
        "txtmlp.3.bias",
        "txt_in.linear_2.bias",
        "diffusion_model.txtmlp.3.bias",
        "transformer.txt_in.linear_2.bias",
    ),
    "txt_norm": (
        "txtmlp.0.scale",
        "txt_in.norm.weight",
        "diffusion_model.txtmlp.0.scale",
        "transformer.txt_in.norm.weight",
    ),
}


def _named_in(phrase: str, name: str) -> bool:
    if phrase == name:
        return True
    return re.search(rf"(^|\s){re.escape(name)}(\s|$)", phrase) is not None


def paraphrases_for(style: str) -> list[str]:
    key = style.strip().lower()
    if key in STYLE_PRESETS:
        return list(STYLE_PRESETS[key])
    for name, paras in STYLE_PRESETS.items():
        if _named_in(key, name) or key in [p.lower() for p in paras]:
            return list(paras)
    return [style.strip()]


def canonical_style_id(phrase: str) -> str | None:
    p = phrase.strip().lower()
    for name, paras in STYLE_PRESETS.items():
        if _named_in(p, name) or p in [x.lower() for x in paras]:
            return name
    return None


def insert_style(content: str, style: str) -> str:
    """Prefix-style insertion (paper §4.2) — do not paraphrase the scene."""
    s, c = style.strip(), content.strip()
    if not s:
        return c
    low = c.lower()
    if low.startswith("an "):
        return f"an {s} {c[3:]}"
    if low.startswith("a "):
        return f"a {s} {c[2:]}"
    return f"{s} {c}"
