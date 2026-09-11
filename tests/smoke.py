"""Real CPU engine plus Wyoming test, run inside the built container."""
import asyncio
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import wave

from wyoming.client import AsyncClient
from wyoming.info import Describe, Info
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioStart, AudioChunk, AudioStop


async def check():
    async with AsyncClient.from_uri('tcp://127.0.0.1:10300') as client:
        await client.write_event(Describe().event())
        event = await asyncio.wait_for(client.read_event(), 5)
        assert Info.is_type(event.type)
        await client.write_event(Transcribe(language='en').event())
        await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())
        with wave.open('/tmp/sample.wav', 'rb') as wav:
            assert (wav.getframerate(), wav.getsampwidth(), wav.getnchannels()) == (16000, 2, 1)
            while audio := wav.readframes(1600):
                await client.write_event(AudioChunk(rate=16000, width=2, channels=1, audio=audio).event())
        await client.write_event(AudioStop().event())
        event = await asyncio.wait_for(client.read_event(), 120)
        assert Transcript.is_type(event.type), event
        text = Transcript.from_event(event).text.lower()
        assert 'country' in text, 'Expected sample transcription'


Path('/data').mkdir(exist_ok=True)
Path('/data/options.json').write_text(json.dumps({'model': 'tiny.en', 'backend': 'cpu'}))
urllib.request.urlretrieve('https://raw.githubusercontent.com/ggml-org/whisper.cpp/927cfce34f31707e17f2bff35c349632fb9e2c3a/samples/jfk.wav', '/tmp/sample.wav')
child = subprocess.Popen(['/opt/venv/bin/python', '/app/app.py'])
try:
    import socket
    for _ in range(240):
        assert child.poll() is None, 'App exited before ready'
        try:
            with socket.create_connection(('127.0.0.1', 10300), timeout=1):
                break
        except OSError:
            time.sleep(1)
    else:
        raise RuntimeError('Readiness timeout')
    asyncio.run(check())
    print('Real CPU transcription through Wyoming passed')
finally:
    child.terminate()
    child.wait(timeout=20)
