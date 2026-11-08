"""Small, explicitly experimental single-step denoising nodes."""
import json
from pathlib import Path
import time

import torch
import folder_paths

from .core import profile_path, file_digest
from .denoise_core import atomic_json, load_cache, save_cache

CATEGORY = 'TCSA/Anima Denoise Experiment'


def directory():
    return Path(folder_paths.get_output_directory()) / 'tcsa_anima_denoise'


def reporter(path):
    from comfy.model_management import throw_exception_if_processing_interrupted
    def progress(value):
        throw_exception_if_processing_interrupted()
        print('TCSA Denoise ' + json.dumps(value, ensure_ascii=False), flush=True)
        atomic_json(path, {'status': 'running', **value})
    return progress


class CacheTeacher:
    CATEGORY = CATEGORY
    FUNCTION = 'run'
    OUTPUT_NODE = True
    RETURN_TYPES = ('STRING', 'LATENT', 'STRING')
    RETURN_NAMES = ('cache_name', 'teacher_latent', 'report')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': ('MODEL',), 'adapter': ('TCSA_ANIMA_ADAPTER',), 'clip': ('CLIP',),
            'profile': ('TCSA_ANIMA_PROFILE',),
            'scenes': ('STRING', {'multiline': True, 'default': 'a furry girl reading a book'}),
            'width': ('INT', {'default': 256, 'min': 128, 'max': 512, 'step': 16}),
            'height': ('INT', {'default': 384, 'min': 128, 'max': 768, 'step': 16}),
            'seed': ('INT', {'default': 42, 'min': 0, 'max': 2147483646}),
            'name': ('STRING', {'default': 'kiwi_denoise_cache_01'})}}

    def run(self, model, adapter, clip, profile, scenes, width, height, seed, name):
        from .denoise_runtime import build_cache
        path = profile_path(directory(), name)
        report_path = path.with_suffix('.json')
        if path.exists():
            raise FileExistsError('Cache exists. Use TCSA Denoise Train with this cache name, or choose a new name.')
        started = time.perf_counter()
        progress = reporter(report_path)
        try:
            progress({'stage': 'encoding'})
            rows, info, latent = build_cache(model, adapter, clip, profile, scenes, width, height, seed, progress)
            info['cache_seconds'] = time.perf_counter() - started
            save_cache(path, rows, info)
            report = dict(info, status='cached', cache_path=str(path), samples=len(rows),
                          teacher_preview='last scene, validation seed = seed + 1')
            atomic_json(report_path, report)
        except Exception as error:
            atomic_json(report_path, {'status': 'failed', 'stage': 'teacher_cache',
                                     'error': str(error), 'seconds': time.perf_counter() - started})
            raise RuntimeError(f'Teacher cache failed: {error}\nReport: {report_path}') from error
        text = json.dumps(report, ensure_ascii=False, indent=2)
        print('TCSA Denoise Cache\n' + text, flush=True)
        return {'ui': {'text': [text]}, 'result': (name, {'samples': latent}, text)}


class TrainDenoise:
    CATEGORY = CATEGORY
    FUNCTION = 'run'
    OUTPUT_NODE = True
    RETURN_TYPES = ('STRING', 'STRING')
    RETURN_NAMES = ('lora_path', 'report')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': ('MODEL',), 'adapter': ('TCSA_ANIMA_ADAPTER',),
            'cache_name': ('STRING', {'default': 'kiwi_denoise_cache_01'}),
            'mode': (['probe_one_update', 'train_short'],),
            'updates': ('INT', {'default': 10, 'min': 1, 'max': 100}),
            'rank': ('INT', {'default': 32, 'min': 1, 'max': 64}),
            'learning_rate': ('FLOAT', {'default': 0.0001, 'min': 0.000001, 'max': 0.01, 'step': 0.00001}),
            'seed': ('INT', {'default': 42, 'min': 0, 'max': 2147483647}),
            'name': ('STRING', {'default': 'kiwi_denoise_probe_01'})}}

    @classmethod
    def IS_CHANGED(cls, model, adapter, cache_name, **kwargs):
        path = profile_path(directory(), cache_name)
        return file_digest(path) if path.exists() else 'missing'

    def run(self, model, adapter, cache_name, mode, updates, rank, learning_rate, seed, name):
        from .denoise_runtime import train_cache
        from .lora_train import export_lora
        if mode not in ('probe_one_update', 'train_short'):
            raise ValueError('Unknown experiment mode.')
        path = profile_path(directory(), cache_name)
        output = profile_path(directory(), name)
        report_path = output.with_suffix('.json')
        if output == path:
            raise ValueError('Run name must differ from cache name.')
        if output.exists() or report_path.exists():
            raise FileExistsError('Run name exists. Choose a new name to preserve previous diagnostics.')
        started = time.perf_counter()
        progress = reporter(report_path)
        saved = ''
        try:
            rows, info = load_cache(path)
            count = 1 if mode == 'probe_one_update' else updates
            progress({'stage': 'checking_model', 'mode': mode, 'updates': count})
            best, result = train_cache(model, adapter, rows, info, count, rank, learning_rate, seed, progress)
            report = {**info, **result, 'mode': mode, 'rank': rank, 'steps': count,
                      'learning_rate': learning_rate, 'training_seed': seed,
                      'cache_path': str(path), 'report_path': str(report_path),
                      'total_seconds': time.perf_counter() - started,
                      'teacher': 'successful_extracted_profile_strength_1',
                      'status': 'probe_passed' if mode == 'probe_one_update' else 'no_validation_improvement'}
            if mode == 'train_short' and best is not None:
                export_lora(output, *best, report)
                saved = str(output)
                report['status'] = 'exported_experimental_lora'
            report['lora_path'] = saved
            atomic_json(report_path, report)
        except Exception as error:
            failure = {'status': 'failed', 'mode': mode, 'error': str(error),
                       'total_seconds': time.perf_counter() - started,
                       'lora_exported': False, 'report_path': str(report_path),
                       'hint': 'Do not increase steps after OOM, detached graph, zero gradient, or parity failure.'}
            atomic_json(report_path, failure)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print('TCSA Denoise Failed\n' + json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
            raise RuntimeError(f'Single-step probe failed: {error}\nReport: {report_path}') from error
        text = json.dumps(report, ensure_ascii=False, indent=2)
        print('TCSA Denoise Result\n' + text, flush=True)
        return {'ui': {'text': [text]}, 'result': (saved, text)}


NODE_CLASS_MAPPINGS = {'TCSAAnimaCacheDenoiseTeacher': CacheTeacher,
                       'TCSAAnimaTrainDenoise': TrainDenoise}
NODE_DISPLAY_NAME_MAPPINGS = {'TCSAAnimaCacheDenoiseTeacher': 'TCSA Denoise Teacher Cache',
                              'TCSAAnimaTrainDenoise': 'TCSA Denoise Train (Experimental)'}
