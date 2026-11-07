import json
from pathlib import Path

import torch
import folder_paths

from .backend import encode, encoder_probe, load_adapter
from .core import (file_digest, load_profile, pad_context, prefix_length,
                   profile_path, save_profile, transplant)

CATEGORY = "TCSA/Anima CPU"
SCENES = "a fox in snow\na woman holding flowers\na quiet harbor at sunrise"
MODES = ["native_baseline", "cpu_roundtrip", "native_artist", "extracted", "zero_strength"]


def output_directory():
    return Path(folder_paths.get_output_directory()) / "tcsa_anima"


def checkpoint_path(name):
    for category in ("diffusion_models", "unet_gguf", "unet"):
        if category in folder_paths.folder_names_and_paths:
            path = folder_paths.get_full_path(category, name)
            if path:
                return Path(path)
    raise FileNotFoundError(name)


class AdapterLoader:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("TCSA_ANIMA_ADAPTER",)

    @classmethod
    def INPUT_TYPES(cls):
        names = set()
        for category in ("diffusion_models", "unet_gguf", "unet"):
            if category in folder_paths.folder_names_and_paths:
                names.update(folder_paths.get_filename_list(category))
        return {"required": {"checkpoint": (sorted(n for n in names if n.lower().endswith((".gguf", ".safetensors"))),)}}

    @classmethod
    def IS_CHANGED(cls, checkpoint):
        stat = checkpoint_path(checkpoint).stat()
        return (stat.st_size, stat.st_mtime_ns)

    def run(self, checkpoint):
        return (load_adapter(checkpoint_path(checkpoint)),)


class Extract:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("TCSA_ANIMA_PROFILE", "STRING")
    RETURN_NAMES = ("profile", "report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "adapter": ("TCSA_ANIMA_ADAPTER",), "clip": ("CLIP",),
            "artist": ("STRING", {"default": "@your_artist", "multiline": True,
                                  "tooltip": "Comma-separated artist prompts, e.g. (@artist_a:0.8), (@artist_b:0.9). Use \\( and \\) for literal parentheses."}),
            "scenes": ("STRING", {"multiline": True, "default": SCENES}),
            "name": ("STRING", {"default": "artist_trial"}),
            "overwrite": ("BOOLEAN", {"default": False})}}

    @torch.inference_mode()
    def run(self, adapter, clip, artist, scenes, name, overwrite=False):
        from comfy.utils import ProgressBar
        path = profile_path(output_directory(), name)
        if path.exists() and not overwrite:
            raise FileExistsError("Profile exists. Use Load Profile, another name, or overwrite=true.")
        artist = artist.strip()
        if not artist.lstrip("( ").startswith("@") or "@your_artist" in artist or len(artist) < 2:
            raise ValueError("Enter artist tags, e.g. (@artist_a:0.8), (@artist_b:0.9). Replace @your_artist.")
        texts = list(dict.fromkeys(line.strip() for line in scenes.splitlines() if line.strip()))
        if not 2 <= len(texts) <= 32:
            raise ValueError("Provide 2–32 distinct extraction scenes, one per line.")
        progress = ProgressBar(len(texts))
        total = None
        reference_ids = None
        for index, scene in enumerate(texts):
            if "@" in scene:
                raise ValueError("Scene lines must not contain artist tags.")
            bare_ids, _, _ = encode(adapter, clip, scene)
            tagged_ids, context, _ = encode(adapter, clip, artist + ", " + scene, allow_weighted=True)
            length = prefix_length(bare_ids, tagged_ids)
            if reference_ids is not None and tagged_ids[:length] != reference_ids:
                raise ValueError("Artist prefix tokenization differs across scenes.")
            reference_ids = tagged_ids[:length]
            prefix = context[0, :length]
            total = prefix.clone() if total is None else total + prefix
            progress.update_absolute(index + 1)
        prefix = total / len(texts)
        probe = encoder_probe(clip)
        info = {"artist": artist, "adapter_sha256": adapter["sha256"],
                "checkpoint": adapter["checkpoint"], "scenes": texts,
                "prefix_ids": reference_ids, "method": "mean-prefix-transplant"}
        save_profile(path, prefix, probe, info, overwrite)
        report = json.dumps({"saved": str(path), "scenes": len(texts),
                             "prefix_tokens": len(reference_ids)}, ensure_ascii=False, indent=2)
        return {"ui": {"text": [report]}, "result": ((prefix, probe, info), report)}


class LoadProfile:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("TCSA_ANIMA_PROFILE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"name": ("STRING", {"default": "artist_trial"})}}

    @classmethod
    def IS_CHANGED(cls, name):
        path = profile_path(output_directory(), name)
        return file_digest(path) if path.exists() else "missing"

    def run(self, name):
        return (load_profile(profile_path(output_directory(), name)),)


class Compare:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "adapter": ("TCSA_ANIMA_ADAPTER",), "clip": ("CLIP",),
            "profile": ("TCSA_ANIMA_PROFILE",),
            "scene": ("STRING", {"multiline": True, "default": "a robot reading a book"}),
            "mode": (MODES, {"default": "native_baseline"}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05})}}

    @torch.inference_mode()
    def run(self, adapter, clip, profile, scene, mode, strength):
        prefix, probe, info = profile
        scene = scene.strip()
        if not scene or "@" in scene:
            raise ValueError("Enter a new scene without artist tags.")
        if scene in info["scenes"]:
            raise ValueError("Use a held-out scene, different from extraction scenes.")
        if info["adapter_sha256"] != adapter["sha256"]:
            raise ValueError("Adapter weights differ from the extraction checkpoint. Re-extract.")
        actual_probe = encoder_probe(clip)
        if probe.shape != actual_probe.shape or not torch.allclose(probe, actual_probe, rtol=1e-4, atol=1e-5):
            raise ValueError("Text encoder output differs from extraction. Use the same CPU CLIP.")
        bare_ids, base, raw = encode(adapter, clip, scene)
        tagged_ids, teacher, raw_teacher = encode(adapter, clip, info["artist"] + ", " + scene, allow_weighted=True)
        length = prefix_length(bare_ids, tagged_ids)
        if tagged_ids[:length] != info["prefix_ids"]:
            raise ValueError("Artist token boundary differs on this scene.")
        candidate = transplant(base, prefix, strength)
        cpu_base = pad_context(base)
        zero = transplant(base, prefix, 0)
        teacher = pad_context(teacher)
        native = raw_teacher if mode == "native_artist" else raw
        if mode in ("native_baseline", "native_artist"):
            positive = [[native["cond"], {k: v for k, v in native.items() if k != "cond"}]]
        else:
            contexts = {"cpu_roundtrip": cpu_base, "extracted": candidate, "zero_strength": zero}
            if mode not in contexts:
                raise ValueError("Unknown comparison mode.")
            # Already projected. No T5 IDs or Qwen masks may reach Anima a second time.
            positive = [[contexts[mode], {}]]
        relative = torch.linalg.vector_norm(candidate - teacher) / torch.linalg.vector_norm(teacher).clamp_min(1e-8)
        report = json.dumps({"mode": mode, "artist": info["artist"], "scene": scene,
                             "checkpoint": info["checkpoint"], "strength": strength,
                             "zero_max_abs_error": (zero - cpu_base).abs().max().item(),
                             "candidate_teacher_relative_l2": relative.item(),
                             "note": "Feature distance is diagnostic only; inspect fixed-seed images."},
                            ensure_ascii=False, indent=2)
        return {"ui": {"text": [report]}, "result": (positive, report)}


class TrainLoRA:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lora_path", "report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "adapter": ("TCSA_ANIMA_ADAPTER",), "clip": ("CLIP",),
            "profile": ("TCSA_ANIMA_PROFILE",),
            "train_scenes": ("STRING", {"multiline": True, "default":
                "a furry girl in a snowy forest\na furry girl reading by a window\na fox walking in a garden\n"
                "a woman drinking tea\na cat sitting on a rooftop\na boy holding an umbrella\n"
                "a stone bridge over a river\na seaside village at sunset"}),
            "validation_scenes": ("STRING", {"multiline": True, "default":
                "a furry girl playing a violin\na lighthouse surrounded by waves"}),
            "rank": ("INT", {"default": 32, "min": 1, "max": 128}),
            "steps": ("INT", {"default": 300, "min": 1, "max": 10000}),
            "learning_rate": ("FLOAT", {"default": 0.001, "min": 0.000001, "max": 0.1, "step": 0.0001}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647}),
            "name": ("STRING", {"default": "kiwi_mix_lora"}),
            "overwrite": ("BOOLEAN", {"default": False})}}

    def run(self, adapter, clip, profile, train_scenes, validation_scenes, rank, steps,
            learning_rate, seed, name, overwrite=False):
        from .lora_train import train_profile
        from comfy.utils import ProgressBar
        from comfy.model_management import throw_exception_if_processing_interrupted
        import logging

        progress_bar = ProgressBar(steps)

        def progress(stage, step, total):
            throw_exception_if_processing_interrupted()
            if stage == 'cache' or step % 20 == 0 or step == total:
                logging.info("TCSA LoRA %s %d/%d", stage, step, total)
            if stage == 'train':
                progress_bar.update_absolute(step, total)

        path = profile_path(Path(folder_paths.get_output_directory()) / 'tcsa_anima_lora', name)
        report = train_profile(adapter, clip, profile, train_scenes, validation_scenes, rank,
                               steps, learning_rate, seed, path, overwrite, progress)
        text = json.dumps(report, ensure_ascii=False, indent=2)
        return {"ui": {"text": [text]}, "result": (str(path), text)}


class InspectLoRA:
    CATEGORY = CATEGORY
    FUNCTION = 'run'
    RETURN_TYPES = ('CONDITIONING', 'CONDITIONING', 'STRING')
    RETURN_NAMES = ('baseline', 'lora_applied', 'report')
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'adapter': ('TCSA_ANIMA_ADAPTER',), 'clip': ('CLIP',),
            'lora_name': (folder_paths.get_filename_list('loras'),),
            'scene': ('STRING', {'multiline': True, 'default': 'a furry girl reading a book'}),
            'strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 3.0, 'step': 0.1})}}

    @classmethod
    def IS_CHANGED(cls, adapter, clip, lora_name, scene, strength):
        path = folder_paths.get_full_path('loras', lora_name)
        return file_digest(path) if path else 'missing'

    def run(self, adapter, clip, lora_name, scene, strength):
        from .lora_inspect import inspect_lora
        path = folder_paths.get_full_path('loras', lora_name)
        if path is None:
            raise FileNotFoundError(lora_name)
        baseline, applied, report = inspect_lora(adapter, clip, path, scene, strength)
        text = json.dumps(report, ensure_ascii=False, indent=2)
        print('TCSA LoRA Check\n' + text, flush=True)
        return {'ui': {'text': [text]}, 'result': ([[baseline, {}]], [[applied, {}]], text)}


NODE_CLASS_MAPPINGS = {"TCSAAnimaAdapterCPU": AdapterLoader, "TCSAAnimaExtract": Extract,
                       "TCSAAnimaLoadProfile": LoadProfile, "TCSAAnimaCompare": Compare,
                       "TCSAAnimaTrainLoRACPU": TrainLoRA, "TCSAAnimaInspectLoRA": InspectLoRA}
NODE_DISPLAY_NAME_MAPPINGS = {"TCSAAnimaAdapterCPU": "TCSA Anima Adapter (CPU)",
                              "TCSAAnimaExtract": "TCSA Extract Artist",
                              "TCSAAnimaLoadProfile": "TCSA Load Profile",
                              "TCSAAnimaCompare": "TCSA Compare Artist",
                              "TCSAAnimaTrainLoRACPU": "TCSA Train LoRA (CPU)",
                              "TCSAAnimaInspectLoRA": "TCSA LoRA Check (CPU)"}
