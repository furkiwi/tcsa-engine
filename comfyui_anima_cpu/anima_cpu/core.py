"""Model-independent profile math and safe, atomic storage."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT = "tcsa-anima-prefix-v2"
MAX_TOKENS = 512


def pad_context(value):
    if value.ndim != 3 or value.shape[0] != 1 or not 0 < value.shape[1] <= MAX_TOKENS:
        raise ValueError("Expected one prompt with 1–512 T5 tokens.")
    if not torch.isfinite(value).all():
        raise ValueError("Non-finite conditioning values.")
    return torch.nn.functional.pad(value, (0, 0, 0, MAX_TOKENS - value.shape[1]))


def prefix_length(bare_ids, tagged_ids):
    length = len(tagged_ids) - len(bare_ids)
    if length <= 0 or tagged_ids[length:] != bare_ids:
        raise ValueError("T5 token boundary changed. Use plain, unweighted English scene text. "
                         "The experiment requires the artist to be a separate prefix.")
    return length


def transplant(base, prefix, strength):
    if not math.isfinite(strength) or not 0 <= strength <= 1.5:
        raise ValueError("Strength must be finite and between 0 and 1.5.")
    baseline = pad_context(base)
    if strength == 0:
        return baseline.clone()
    if prefix.ndim != 2 or prefix.shape[1] != base.shape[-1] or prefix.shape[0] < 1:
        raise ValueError("Profile dimensions do not match this encoder/adapter.")
    candidate = pad_context(torch.cat((prefix.to(base)[None], base), dim=1))
    # Interpolate full, equally padded contexts. Zero is exactly the baseline.
    return torch.lerp(baseline, candidate, strength)


def profile_path(directory, name):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise ValueError("Profile name must contain 1–64 ASCII letters, digits, - or _.")
    return Path(directory) / (name + ".safetensors")


def save_profile(path, prefix, probe, metadata, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    # Serialize competing writers, including across ComfyUI processes.
    with lock.open("x"):
        pass
    temp_path = None
    try:
        if path.exists() and not overwrite:
            raise FileExistsError("Profile exists. Choose another name or enable overwrite.")
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temp:
            temp_path = Path(temp.name)
        save_file({"prefix": prefix.detach().cpu().float().contiguous().clone(),
                   "encoder_probe": probe.detach().cpu().float().contiguous().clone()},
                  str(temp_path), metadata={"format": FORMAT, "info": json.dumps(metadata, ensure_ascii=False)})
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def load_profile(path):
    with safe_open(str(path), framework="pt", device="cpu") as file:
        meta = file.metadata() or {}
        if meta.get("format") != FORMAT:
            raise ValueError("Unsupported profile. Extract a new v2 profile; old .npy files are unsupported.")
        info = json.loads(meta["info"])
        for key in ("prefix", "encoder_probe"):
            shape = file.get_slice(key).get_shape()
            if len(shape) != 2 or not 0 < shape[0] <= MAX_TOKENS or shape[1] != 1024:
                raise ValueError("Invalid Anima profile tensor shape.")
        prefix = file.get_tensor("prefix")
        probe = file.get_tensor("encoder_probe")
    if not torch.isfinite(prefix).all() or not torch.isfinite(probe).all():
        raise ValueError("Invalid profile values.")
    return prefix, probe, info


def file_digest(path):
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(file.read()).hexdigest()
