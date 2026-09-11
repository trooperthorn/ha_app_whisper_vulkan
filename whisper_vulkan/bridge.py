"""Bounded Wyoming audio receiver for the loopback whisper.cpp server."""
import asyncio
import io
import logging
import time
import wave
from functools import partial

import httpx
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioStart, AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

LOG = logging.getLogger(__name__)
MAX_BYTES = 16000 * 2 * 60


def wav_bytes(audio):
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
        wav.writeframes(audio)
    return output.getvalue()


class Handler(AsyncEventHandler):
    def __init__(self, info, lock, client, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.info, self.lock, self.client = info, lock, client
        self.audio = bytearray()
        self.started = False

    async def fail(self, message):
        self.audio.clear()
        self.started = False
        await self.write_event(Event(type='error', data={'text': message, 'code': 'transcription-failed'}))
        return False

    async def handle_event(self, event):
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
        elif Transcribe.is_type(event.type):
            self.audio.clear()
            self.started = False
            if Transcribe.from_event(event).language not in (None, 'en'):
                return await self.fail('This model supports English only')
        elif AudioStart.is_type(event.type):
            fmt = AudioStart.from_event(event)
            if (fmt.rate, fmt.width, fmt.channels) != (16000, 2, 1):
                return await self.fail('Expected 16 kHz, 16-bit mono PCM')
            self.audio.clear()
            self.started = True
        elif AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            if not self.started or (chunk.rate, chunk.width, chunk.channels) != (16000, 2, 1):
                return await self.fail('Invalid audio format or event order')
            if len(self.audio) + len(chunk.audio) > MAX_BYTES:
                return await self.fail('Audio exceeds 60 seconds')
            self.audio.extend(chunk.audio)
        elif AudioStop.is_type(event.type):
            if not self.started or not self.audio or len(self.audio) % 2:
                return await self.fail('Empty or malformed audio')
            try:
                async with asyncio.timeout(120):
                    async with self.lock:
                        start = time.monotonic()
                        response = await self.client.post(
                            'http://127.0.0.1:8080/inference',
                            files={'file': ('audio.wav', wav_bytes(self.audio), 'audio/wav')},
                            data={'response_format': 'json', 'language': 'en', 'temperature': '0.0'},
                        )
                        response.raise_for_status()
                        text = response.json()['text']
                        if not isinstance(text, str):
                            raise ValueError('Invalid transcript')
                        LOG.info('Transcribed %.2fs audio in %.2fs', len(self.audio) / 32000, time.monotonic() - start)
                await self.write_event(Transcript(text=text.strip()).event())
            except (httpx.HTTPError, TimeoutError, ValueError, KeyError):
                LOG.warning('Transcription failed; audio and response content omitted')
                return await self.fail('Speech engine failed or timed out')
            finally:
                self.audio.clear()
                self.started = False
            return False
        return True


async def serve(model):
    attribution = Attribution(name='whisper.cpp', url='https://github.com/ggml-org/whisper.cpp')
    info = Info(asr=[AsrProgram(name='whisper-vulkan', description='Local Whisper Vulkan', attribution=attribution,
        installed=True, version='0.1.0', models=[AsrModel(name=model, description=model,
        attribution=attribution, installed=True, languages=['en'])])])
    async with httpx.AsyncClient(timeout=110, trust_env=False) as client:
        await AsyncServer.from_uri('tcp://0.0.0.0:10300').run(partial(Handler, info, asyncio.Lock(), client))
