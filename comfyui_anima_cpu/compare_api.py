"""Run five serial comparisons using a known-working ComfyUI API workflow."""
import argparse
import copy
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

MODES = ['native_baseline', 'cpu_roundtrip', 'native_artist', 'extracted', 'zero_strength']


def request(base, route, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base.rstrip('/') + route, data=data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def prepare(graph, mode, seed):
    graph = copy.deepcopy(graph)
    compare = [key for key, node in graph.items() if node.get('class_type') == 'TCSAAnimaCompare']
    if len(compare) != 1:
        raise ValueError('Workflow must contain exactly one TCSAAnimaCompare node.')
    if any(node.get('class_type') == 'TCSAAnimaExtract' for node in graph.values()):
        raise ValueError('Use TCSAAnimaLoadProfile for batch comparisons; extract separately first.')
    samplers = [node for node in graph.values() if node.get('class_type') in ('KSampler', 'KSamplerAdvanced')]
    if len(samplers) != 1 or samplers[0]['inputs'].get('positive') != [compare[0], 0]:
        raise ValueError('Use one KSampler or KSamplerAdvanced with positive connected directly to Compare.')
    sampler = samplers[0]
    sampler['inputs']['seed' if sampler['class_type'] == 'KSampler' else 'noise_seed'] = seed
    graph[compare[0]]['inputs']['mode'] = mode
    saves = [node for node in graph.values() if node.get('class_type') == 'SaveImage']
    if not saves:
        raise ValueError('Connect VAE Decode to a SaveImage node.')
    for node in saves:
        node['inputs']['filename_prefix'] = 'TCSA/' + mode
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workflow', type=Path, help='Exported API-format workflow JSON')
    parser.add_argument('--url', default='http://127.0.0.1:8188')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--out', type=Path, default=Path('tcsa_comparison'))
    parser.add_argument('--timeout', type=int, default=7200, help='Seconds per image')
    args = parser.parse_args()
    graph = json.loads(args.workflow.read_text(encoding='utf-8'))
    if 'nodes' in graph:
        parser.error('Export API format, not the normal editor workflow.')
    prepared = [(mode, prepare(graph, mode, args.seed)) for mode in MODES]
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {'seed': args.seed, 'runs': []}
    for mode, workflow in prepared:
        response = request(args.url, '/prompt', {'prompt': workflow})
        if response.get('node_errors'):
            raise RuntimeError(response['node_errors'])
        prompt_id = response['prompt_id']
        run = {'mode': mode, 'prompt_id': prompt_id, 'workflow': workflow, 'status': 'queued'}
        manifest['runs'].append(run)
        manifest_path = args.out / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(mode, prompt_id, flush=True)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            history = request(args.url, '/history/' + prompt_id).get(prompt_id)
            if history:
                run['status'] = history.get('status', {})
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
                if run['status'].get('status_str') != 'success':
                    raise RuntimeError(history.get('status'))
                for node_id, output in history.get('outputs', {}).items():
                    for index, entry in enumerate(output.get('images', [])):
                        query = urllib.parse.urlencode({key: entry[key] for key in ('filename', 'subfolder', 'type')})
                        with urllib.request.urlopen(args.url.rstrip('/') + '/view?' + query, timeout=60) as image:
                            (args.out / f'{mode}_{node_id}_{index}.png').write_bytes(image.read())
                break
            time.sleep(2)
        else:
            raise TimeoutError(f'{prompt_id} is still pending. Check ComfyUI before resubmitting.')
    print('Finished. Compare images in', args.out, flush=True)


if __name__ == '__main__':
    main()
