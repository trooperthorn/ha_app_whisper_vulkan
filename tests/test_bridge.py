import asyncio
import io
import sys
import wave
from pathlib import Path

import httpx
import pytest
from wyoming.audio import AudioStart, AudioChunk, AudioStop
from wyoming.info import Info

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'whisper_vulkan'))
from bridge import Handler, MAX_BYTES, wav_bytes
from app import options, command


def test_wav_header_and_audio():
    with wave.open(io.BytesIO(wav_bytes(b'\x01\x00' * 160)), 'rb') as wav:
        assert (wav.getframerate(), wav.getsampwidth(), wav.getnchannels()) == (16000, 2, 1)
        assert wav.readframes(160) == b'\x01\x00' * 160


@pytest.mark.parametrize('raw', [{'model': '../bad'}, {'threads': 0}, {'backend': 'cuda'}, {'beam_size': True}])
def test_invalid_options(raw):
    with pytest.raises(ValueError):
        options(raw)


def test_cpu_is_explicit():
    assert '--no-gpu' in command(options({'backend': 'cpu'}), '/model.bin')
    assert '--no-gpu' not in command(options({}), '/model.bin')


def run_case(mode):
    async def scenario():
        async def respond(request):
            assert b'audio.wav' in request.content
            assert b'RIFF' in request.content
            return httpx.Response(500 if mode == 'http_error' else 200, json={'text': ' Turn on lights '})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            handler = Handler(Info(), asyncio.Lock(), client, None, None)
            events = []
            async def capture(event):
                events.append(event)
            handler.write_event = capture
            await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
            payload = b'\x00\x00' * 160 if mode != 'oversize' else b'\x00' * (MAX_BYTES + 1)
            keep = await handler.handle_event(AudioChunk(rate=16000, width=2, channels=1, audio=payload).event())
            if keep:
                await handler.handle_event(AudioStop().event())
            assert not handler.audio
            return events
    return asyncio.run(scenario())


def test_transcription():
    assert run_case('ok')[0].data['text'] == 'Turn on lights'


@pytest.mark.parametrize('mode', ['http_error', 'oversize'])
def test_failures_are_protocol_errors(mode):
    assert run_case(mode)[0].type == 'error'
