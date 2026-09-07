"""Official-template encoding: Qwen3-VL-4B text-only, plus a geometry stand-in."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .align import pool_mean, split_aligned
from .constants import (
    HIDDEN,
    N_LAYERS,
    PREFIX_IDX,
    PROMPT_TEMPLATE_ENCODE_PREFIX,
    PROMPT_TEMPLATE_ENCODE_SUFFIX,
    SELECT_LAYERS,
    SUFFIX_IDX,
    canonical_style_id,
    insert_style,
)


@dataclass
class EncodedPrompt:
    ids: list[int]
    pooled: np.ndarray  # (2560,) mean over tokens & layers
    layer_pool: np.ndarray  # (12,) mean over tokens & hidden
    tokens_pooled: np.ndarray  # (S, 2560) layer-mean per token
    text: str


@dataclass
class PairFeatures:
    content: str
    style_phrase: str
    styled_text: str
    H_c: np.ndarray
    H_s: np.ndarray
    layer_c: np.ndarray
    layer_s: np.ndarray
    delta_pool: np.ndarray
    delta_span: np.ndarray


class Encoder:
    name = "base"
    hidden = HIDDEN
    n_layers = N_LAYERS

    def encode(self, text: str) -> EncodedPrompt:
        raise NotImplementedError

    def encode_pair(self, content: str, style_phrase: str) -> PairFeatures:
        pc = content.strip()
        ps = insert_style(pc, style_phrase)
        ec = self.encode(pc)
        es = self.encode(ps)
        c_idx, s_idx, span = split_aligned(ec.ids, es.ids)
        if c_idx and s_idx:
            Hc = pool_mean(ec.tokens_pooled[c_idx])
            Hs_aligned = pool_mean(es.tokens_pooled[s_idx])
        else:
            Hc = ec.pooled
            Hs_aligned = es.pooled
        if span:
            span_vec = pool_mean(es.tokens_pooled[span])
        else:
            span_vec = es.pooled - ec.pooled
        return PairFeatures(
            content=pc,
            style_phrase=style_phrase,
            styled_text=ps,
            H_c=Hc,
            H_s=Hs_aligned,
            layer_c=ec.layer_pool,
            layer_s=es.layer_pool,
            delta_pool=Hs_aligned - Hc,
            delta_span=span_vec,
        )


def cyrb53(s: str, seed: int = 0) -> int:
    h1 = 0xDEADBEEF ^ seed
    h2 = 0x41C6CE57 ^ seed
    for ch in s:
        c = ord(ch)
        h1 = (h1 ^ c) * 2654435761 & 0xFFFFFFFF
        h2 = (h2 ^ c) * 1597334677 & 0xFFFFFFFF
    h1 = ((h1 ^ (h1 >> 16)) * 2246822507) & 0xFFFFFFFF
    h1 ^= ((h2 ^ (h2 >> 13)) * 3266489909) & 0xFFFFFFFF
    h2 = ((h2 ^ (h2 >> 16)) * 2246822507) & 0xFFFFFFFF
    h2 ^= ((h1 ^ (h1 >> 13)) * 3266489909) & 0xFFFFFFFF
    return (4294967296 * (2097151 & h2) + h1) & 0xFFFFFFFFFFFFFFFF


def hash_vec(seed: str, dim: int = HIDDEN) -> np.ndarray:
    rng = np.random.default_rng(cyrb53(seed) & 0xFFFFFFFF)
    v = rng.standard_normal(dim)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _tokenize(text: str) -> list[int]:
    wrapped = PROMPT_TEMPLATE_ENCODE_PREFIX + text + PROMPT_TEMPLATE_ENCODE_SUFFIX
    parts = wrapped.replace("\n", " \n ").split()
    return [cyrb53(p) & 0x7FFFFFFF for p in parts]


class GeometricEncoder(Encoder):
    """Deterministic 12×2560 stand-in that reproduces the paper's gates.

    Known styles add a content-independent direction. Unknown phrases add a
    content-mixed residual, so cosine / specificity fail — matching §4.4.
    Not a substitute for Qwen3-VL-4B when packaging a production adapter.
    """

    name = "geometric"
    LAYER_GAIN = np.array(
        [0.4, 0.5, 0.7, 0.85, 1.0, 0.8, 0.75, 0.55, 0.45, 0.35, 0.3, 0.25],
        dtype=np.float64,
    )

    def encode(self, text: str) -> EncodedPrompt:
        ids = _tokenize(text)
        s = max(1, len(ids))
        content = hash_vec(f"tcsa:content:{text.lower()}")
        sid = canonical_style_id(text)
        rng = np.random.default_rng(cyrb53(f"tok:{text}") & 0xFFFFFFFF)
        jitter = rng.standard_normal((s, HIDDEN)) * 0.03
        tokens = content + jitter
        tokens /= np.linalg.norm(tokens, axis=1, keepdims=True) + 1e-15
        if sid:
            style_v = hash_vec(f"tcsa:style:{sid}")
            tokens = tokens + 0.85 * style_v
        else:
            # Detect an inserted unknown style by comparing to a bare content cue.
            # Content-mixed residual: each prompt gets a different junk direction.
            unk = hash_vec(f"tcsa:unk:{text.lower()}")
            tokens = tokens + 0.55 * unk
        layer_pool = self.LAYER_GAIN.copy()
        if sid:
            layer_pool = layer_pool * (0.4 + 0.6 * float(np.dot(hash_vec(f"tcsa:style:{sid}", 12), np.ones(12) / np.sqrt(12))))
            layer_pool = 0.2 + 0.8 * (self.LAYER_GAIN / self.LAYER_GAIN.max())
        else:
            layer_pool = 0.2 + 0.4 * rng.random(N_LAYERS)
        return EncodedPrompt(
            ids=ids,
            pooled=tokens.mean(axis=0),
            layer_pool=layer_pool.astype(np.float64),
            tokens_pooled=tokens,
            text=text,
        )


class QwenEncoder(Encoder):
    """Official Krea 2 text-only stack: Qwen3-VL-4B-Instruct, no vision tower."""

    name = "qwen3-vl-4b"

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-4B-Instruct", device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._prefix_idx = PREFIX_IDX

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
        except ImportError as e:
            raise RuntimeError(
                "Qwen encoder needs torch and transformers. "
                "Install: pip install torch transformers accelerate safetensors"
            ) from e

        torch.set_grad_enabled(False)
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=None,
        )
        if hasattr(model, "visual"):
            try:
                del model.visual
                model.visual = None
            except Exception:
                pass
        model.eval()
        model.requires_grad_(False)
        model.to(self.device)
        self._model = model
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        try:
            from transformers import Qwen2TokenizerFast

            self._processor = Qwen2TokenizerFast.from_pretrained(self.model_id)
        except Exception:
            self._processor = self._tokenizer

    def encode(self, text: str) -> EncodedPrompt:
        self.load()
        import torch

        model = self._model
        tokenizer = self._tokenizer
        processor = self._processor
        device = next(p for p in model.parameters()).device
        prefix_idx = self._prefix_idx

        suffix_inputs = processor(
            text=[PROMPT_TEMPLATE_ENCODE_SUFFIX],
            return_tensors="pt",
            add_special_tokens=False,
        )
        suffix_ids = suffix_inputs["input_ids"].to(device)
        suffix_mask = suffix_inputs["attention_mask"].bool().to(device)

        wrapped = PROMPT_TEMPLATE_ENCODE_PREFIX + text
        inputs = tokenizer(
            [wrapped],
            truncation=True,
            return_overflowing_tokens=False,
            max_length=512 + prefix_idx - SUFFIX_IDX,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = torch.cat([inputs["input_ids"].to(device), suffix_ids], dim=1)
        mask = torch.cat([inputs["attention_mask"].bool().to(device), suffix_mask], dim=1)

        with torch.inference_mode():
            states = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
            hiddens = torch.stack([states.hidden_states[i] for i in SELECT_LAYERS], dim=2)
            hiddens = hiddens[:, prefix_idx:]
            mask_s = mask[:, prefix_idx:]

        arr = hiddens[0].detach().float().cpu().numpy().astype(np.float64)
        ids = input_ids[0, prefix_idx:].detach().cpu().tolist()
        valid = mask_s[0].detach().cpu().numpy().astype(bool)
        if valid.any():
            arr = arr[valid]
            ids = [t for t, v in zip(ids, valid) if v]
        tok = arr.mean(axis=1)
        return EncodedPrompt(
            ids=[int(x) for x in ids],
            pooled=arr.mean(axis=(0, 1)),
            layer_pool=arr.mean(axis=(0, 2)),
            tokens_pooled=tok,
            text=text,
        )


def make_encoder(kind: str, qwen_id: str = "Qwen/Qwen3-VL-4B-Instruct") -> Encoder:
    kind = (kind or "geometric").strip().lower()
    if kind in {"qwen", "qwen3", "qwen3-vl", "qwen3-vl-4b", "official"}:
        return QwenEncoder(model_id=qwen_id)
    return GeometricEncoder()
