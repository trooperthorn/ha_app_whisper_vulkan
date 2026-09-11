"""App startup, model storage, GPU preflight, and child supervision."""
import asyncio
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import urllib.request

import httpx
from bridge import serve

LOG = logging.getLogger(__name__)


def options(raw):
    result = dict(model='small.en', language='en', backend='vulkan', threads=4, beam_size=5, download_model=True)
    result.update(raw)
    if result['model'] not in ('tiny.en', 'base.en', 'small.en') or result['language'] != 'en':
        raise ValueError('Unsupported model or language')
    if result['backend'] not in ('vulkan', 'cpu'):
        raise ValueError('Unsupported backend')
    for key, limit in (('threads', 16), ('beam_size', 10)):
        if type(result[key]) is not int or not 1 <= result[key] <= limit:
            raise ValueError(f'Invalid {key}')
    if type(result['download_model']) is not bool:
        raise ValueError('Invalid download_model')
    return result


def model_path(config):
    directory = Path('/data/models')
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"ggml-{config['model']}.bin"
    if not target.exists():
        if not config['download_model']:
            raise RuntimeError('Model missing; enable download_model for initial download')
        LOG.info('Downloading model %s; subsequent starts use the cached file', config['model'])
        temporary = target.with_suffix('.part')
        try:
            with urllib.request.urlopen(f'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{target.name}', timeout=60) as source, temporary.open('wb') as dest:
                while chunk := source.read(1024 * 1024):
                    dest.write(chunk)
            if temporary.stat().st_size < 1000000:
                raise RuntimeError('Model download is unexpectedly small')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


def command(config, model):
    args = ['/usr/local/bin/whisper-server', '--model', str(model), '--host', '127.0.0.1', '--port', '8080',
            '--language', 'en', '--threads', str(config['threads']), '--beam-size', str(config['beam_size']), '--no-context']
    if config['backend'] == 'cpu':
        args.append('--no-gpu')
    return args


async def main():
    config = options(json.loads(Path('/data/options.json').read_text()))
    if config['backend'] == 'vulkan':
        nodes = list(Path('/dev/dri').glob('renderD*'))
        if not nodes:
            raise RuntimeError('No GPU render device mapped; choose CPU mode or check video device access')
        diagnostic = subprocess.run(['vulkaninfo', '--summary'], capture_output=True, text=True, timeout=30)
        LOG.info('Vulkan preflight:\n%s', diagnostic.stdout)
        if diagnostic.returncode or 'AMD' not in diagnostic.stdout:
            raise RuntimeError('AMD Vulkan device not detected; refusing silent CPU fallback')
    model = model_path(config)
    LOG.info('Requested backend=%s model=%s; engine logs below establish actual backend use', config['backend'], config['model'])
    # The upstream HTTP server runs unprivileged, with only render-device groups.
    model.chmod(0o644)
    cache = Path('/data/cache')
    cache.mkdir(exist_ok=True)
    os.chown(cache, 65534, 65534)
    groups = sorted({node.stat().st_gid for node in Path('/dev/dri').glob('renderD*')})
    child = await asyncio.create_subprocess_exec(*command(config, model), user=65534, group=65534, extra_groups=groups)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    tasks = []
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            for _ in range(180):
                if child.returncode is not None or stop.is_set():
                    raise RuntimeError('Engine exited before readiness')
                try:
                    response = await client.get('http://127.0.0.1:8080/health', timeout=2)
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1)
            else:
                raise RuntimeError('Engine readiness timed out')
        LOG.info('Ready for Wyoming at port 10300; add this App hostname in the Wyoming integration')
        tasks = [asyncio.create_task(serve(config['model'])), asyncio.create_task(child.wait()), asyncio.create_task(stop.wait())]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        if not stop.is_set():
            for task in done:
                task.result()
            raise RuntimeError('Speech service stopped unexpectedly')
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if child.returncode is None:
            child.terminate()
            try:
                await asyncio.wait_for(child.wait(), 10)
            except TimeoutError:
                child.kill()
                await child.wait()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    asyncio.run(main())
