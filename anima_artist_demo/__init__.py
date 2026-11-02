"""CPU-only experiment for the stock ComfyUI Anima workflow."""
from pathlib import Path
import os
import tempfile
import re
import numpy as np
import torch
import folder_paths


def profile_path(name):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name):
        raise ValueError("Use 1-64 letters, digits, underscores or hyphens for profile.")
    directory = Path(folder_paths.get_output_directory()) / "anima_artist_profiles"
    directory.mkdir(exist_ok=True)
    return directory / (name + ".npy")


@torch.inference_mode()
def encode(model, clip, prompt):
    cond = clip.encode_from_tokens(clip.tokenize(prompt), return_pooled=True, return_dict=True)
    ctx = cond.pop("cond")
    ids = cond.pop("t5xxl_ids", None)
    net = model.model.diffusion_model
    if ids is None or not hasattr(net, "llm_adapter"):
        raise ValueError("Connect stock Anima MODEL and Anima CLIP.")
    param = next(net.llm_adapter.parameters())
    if param.device.type != "cpu":
        raise ValueError("Start ComfyUI with --cpu --force-fp32.")
    weights = cond.pop("t5xxl_weights")
    out = net.preprocess_text_embeds(ctx.to(param), ids[None].to(param.device),
                                     weights.reshape(1, -1, 1).to(param))
    return ids.tolist(), out[:, :ids.numel()].float(), cond


BASE = {"model": ("MODEL",), "clip": ("CLIP",), "profile": ("STRING", {"default": "artist_trial"})}


class Extract:
    CATEGORY = "Anima Artist Demo"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {**BASE, "artist": ("STRING", {"default": "@nnn yryr"}),
                             "scenes": ("STRING", {"multiline": True, "default": "a fox in snow\na quiet harbor\na woman holding flowers"})}}

    def run(self, model, clip, profile, artist, scenes):
        artist = artist.strip()
        if not artist.startswith("@") or len(artist) < 2:
            raise ValueError("Enter @ followed by the artist name.")
        vectors = []
        ref = None
        for scene in filter(None, map(str.strip, scenes.splitlines())):
            bare, _, _ = encode(model, clip, scene)
            tagged, states, _ = encode(model, clip, artist + ", " + scene)
            n = len(tagged) - len(bare)
            if n <= 0 or tagged[n:] != bare or (ref is not None and tagged[:n] != ref):
                raise ValueError("Unstable T5 prefix boundary. Use simple unweighted scene descriptions.")
            ref = tagged[:n]
            vectors.append(states[0, :n].numpy())
        if not vectors:
            raise ValueError("Provide at least one scene.")
        path = profile_path(profile)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            try:
                np.save(tmp, np.mean(vectors, axis=0), allow_pickle=False)
                tmp.close()
                os.replace(tmp.name, path)
            finally:
                Path(tmp.name).unlink(missing_ok=True)
        return {"ui": {"text": [str(path)]}, "result": (profile,)}


class Apply:
    CATEGORY = Extract.CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("CONDITIONING",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {**BASE, "scene": ("STRING", {"multiline": True, "default": "a robot reading a book"}),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05})}}

    @classmethod
    def IS_CHANGED(cls, model, clip, profile, scene, strength):
        p = profile_path(profile)
        return p.stat().st_mtime_ns if strength and p.exists() else 0

    def run(self, model, clip, profile, scene, strength):
        _, ctx, meta = encode(model, clip, scene)
        if strength:
            arr = np.load(profile_path(profile), allow_pickle=False)
            if arr.ndim != 2 or arr.shape[1] != ctx.shape[-1] or not np.isfinite(arr).all():
                raise ValueError("Invalid Anima artist profile.")
            ctx = torch.cat([torch.from_numpy(arr).to(ctx)[None] * strength, ctx], dim=1)
        ctx = torch.nn.functional.pad(ctx, (0, 0, 0, max(0, 512 - ctx.shape[1])))
        return ([[ctx, meta]],)


NODE_CLASS_MAPPINGS = {"AnimaArtistExtractCPU": Extract, "AnimaArtistApplyCPU": Apply}
