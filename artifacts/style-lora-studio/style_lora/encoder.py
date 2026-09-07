"""Text encoders: demo hash embeddings, optional official Qwen3-VL 12-layer tap."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .defaults import DEMO_DIM, KNOWN_STYLES, KREA_TAP_LAYERS
from .prompts import aligned_content_span, tokenize


def _seed64(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % (2**63)


def token_vector(token: str, dim: int) -> np.ndarray:
    rng = np.random.default_rng(_seed64("tok:" + token))
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v) + 1e-8)
    return v / n


def style_direction(phrase: str, dim: int) -> np.ndarray:
    head = tokenize(phrase)[0] if tokenize(phrase) else phrase.lower()
    return token_vector("STYLE::" + head, dim)


def is_known_style(phrase: str) -> bool:
    low = phrase.lower()
    return any(k in low for k in KNOWN_STYLES)


def canonical_style_key(phrase: str) -> str:
    low = phrase.lower()
    return next((k for k in sorted(KNOWN_STYLES, key=len, reverse=True) if k in low), phrase)


@dataclass
class EncodeResult:
    H_c: np.ndarray
    H_s: np.ndarray
    Delta: np.ndarray
    pairs: list[dict]
    dim: int
    backend: str
    notes: list[str]
    Delta_span: np.ndarray | None = None
    H_layers_c: np.ndarray | None = None
    H_layers_s: np.ndarray | None = None
    tap_layers: list[int] = field(default_factory=list)
    pooling: str = "mean"


class DemoEncoder:
    """Pedagogical encoder. Known styles share one offset; unknown stay content-tied."""

    def __init__(self, dim: int = DEMO_DIM):
        self.dim = dim

    def _style_boost(self, phrase: str, styled: str) -> np.ndarray:
        if is_known_style(phrase):
            extra = tokenize(phrase)
            boost = 0.85 * token_vector("STYLE::" + canonical_style_key(phrase), self.dim)
            if extra:
                boost = boost + 0.10 * np.mean(
                    [token_vector("span:" + t, self.dim) for t in extra], axis=0
                )
            return boost.astype(np.float32)
        return (
            0.55 * token_vector("UNK::" + phrase + "::" + styled, self.dim)
            + 0.04 * style_direction(phrase, self.dim)
        ).astype(np.float32)

    def encode_pairs(self, pairs: list[dict], phrases: list[str]) -> EncodeResult:
        H_c, H_s, spans = [], [], []
        notes = [
            "demo encoder: hash embeddings, not Qwen3-VL",
            "known style phrases get a shared offset; unknown phrases stay content-tied",
        ]
        annotated = []
        for p in pairs:
            shared, extra = aligned_content_span(p["content"], p["styled"])
            toks_c = tokenize(p["content"]) or ["empty"]
            toks_shared = shared or toks_c
            hc = np.mean([token_vector(t, self.dim) for t in toks_c], axis=0)
            hs = np.mean([token_vector(t, self.dim) for t in toks_shared], axis=0)
            phrase = p.get("phrase") or (phrases[0] if phrases else "")
            boost = self._style_boost(phrase, p["styled"])
            hs = hs + boost
            res = 0.06 * token_vector("RES::" + p["content"], self.dim)
            hc = (0.55 * hc + res).astype(np.float32)
            hs = (0.55 * hs + res).astype(np.float32)
            if extra:
                span = np.mean([token_vector("span:" + t, self.dim) for t in extra], axis=0)
                dspan = (boost + 0.12 * span).astype(np.float32)
            else:
                dspan = boost.astype(np.float32)
            q = dict(p)
            q["shared_tokens"] = shared
            q["style_tokens"] = extra
            H_c.append(hc)
            H_s.append(hs)
            spans.append(dspan)
            annotated.append(q)
        H_c = np.stack(H_c, axis=0)
        H_s = np.stack(H_s, axis=0)
        return EncodeResult(
            H_c=H_c,
            H_s=H_s,
            Delta=H_s - H_c,
            Delta_span=np.stack(spans, axis=0),
            pairs=annotated,
            dim=self.dim,
            backend="demo",
            notes=notes,
            pooling="mean",
        )


class RealEncoder:
    """Official Krea 2 text stack: Qwen3-VL-4B, text-only, 12-layer tap.

    Paper §2.3: hidden states at indices 2, 5, 8, …, 35, shape (B, 12, S, 2560).
    Layer axis is stored and only collapsed for the cosine / PCA gates.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        max_length: int = 256,
        instruction_template: str = "",
        pooling: str = "mean",
    ):
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.instruction_template = instruction_template.strip()
        self.pooling = pooling
        self._model = None
        self._tokenizer = None
        self._torch = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "真實編碼器需要安裝 torch 與 transformers：pip install torch transformers"
            ) from e
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        self._model.to(self.device)
        self._model.eval()
        self._torch = torch

    def _wrap(self, text: str) -> str:
        if self.instruction_template:
            return self.instruction_template.format(prompt=text)
        tok = self._tokenizer
        if tok is not None and hasattr(tok, "apply_chat_template"):
            try:
                return tok.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                return text
        return text

    def _pool_seq(self, hidden: np.ndarray) -> np.ndarray:
        if hidden.ndim != 2 or hidden.shape[0] == 0:
            return hidden.reshape(-1)[: hidden.size].astype(np.float32)
        if self.pooling == "last":
            return hidden[-1].astype(np.float32)
        return hidden.mean(axis=0).astype(np.float32)

    def encode_texts(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return (n, d) pooled over layers+tokens, and (n, 12, d) layer-wise pooled tokens."""
        self._load()
        torch = self._torch
        pooled, layers = [], []
        n_hs = None
        with torch.no_grad():
            for text in texts:
                wrapped = self._wrap(text)
                batch = self._tokenizer(
                    wrapped,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                )
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = self._model(**batch, output_hidden_states=True)
                hs = out.hidden_states
                n_hs = len(hs)
                stack = []
                for idx in KREA_TAP_LAYERS:
                    layer = hs[idx] if idx < len(hs) else hs[-1]
                    stack.append(layer[0].float().cpu().numpy())
                stack = np.stack(stack, axis=0)  # (12, S, d)
                layer_pool = np.stack([self._pool_seq(stack[i]) for i in range(stack.shape[0])], axis=0)
                layers.append(layer_pool)
                pooled.append(layer_pool.mean(axis=0).astype(np.float32))
        notes_tap = f"tap={KREA_TAP_LAYERS} hidden_states={n_hs}"
        self._last_tap_note = notes_tap
        return np.stack(pooled, axis=0), np.stack(layers, axis=0)

    def encode_pairs(self, pairs: list[dict], phrases: list[str]) -> EncodeResult:
        H_c, L_c = self.encode_texts([p["content"] for p in pairs])
        H_s, L_s = self.encode_texts([p["styled"] for p in pairs])
        annotated = []
        spans = []
        for i, p in enumerate(pairs):
            q = dict(p)
            shared, extra = aligned_content_span(p["content"], p["styled"])
            q["shared_tokens"] = shared
            q["style_tokens"] = extra
            annotated.append(q)
            if extra:
                spans.append(np.mean([token_vector("span:" + t, H_s.shape[1]) for t in extra], axis=0))
            else:
                spans.append((H_s[i] - H_c[i]).astype(np.float32))
        notes = [
            f"encoder={self.model_path}",
            f"pooling={self.pooling} over 12-layer tap {KREA_TAP_LAYERS}",
            getattr(self, "_last_tap_note", "tap=krea-12"),
            "layer axis stored; collapsed by mean for cosine / PCA gates",
            "instruction template must match Krea inference — override te_template if needed",
        ]
        return EncodeResult(
            H_c=H_c,
            H_s=H_s,
            Delta=H_s - H_c,
            Delta_span=np.stack(spans, axis=0).astype(np.float32),
            H_layers_c=L_c,
            H_layers_s=L_s,
            tap_layers=list(KREA_TAP_LAYERS),
            pairs=annotated,
            dim=int(H_c.shape[1]),
            backend="qwen3vl-12tap",
            notes=notes,
            pooling=self.pooling,
        )


def get_encoder(
    mode: str,
    te_path: str | None = None,
    instruction_template: str = "",
) -> DemoEncoder | RealEncoder:
    if mode == "real":
        if not te_path:
            raise RuntimeError("真實模式需要 text encoder 路徑（Qwen3-VL-4B）")
        return RealEncoder(te_path, instruction_template=instruction_template)
    return DemoEncoder()
