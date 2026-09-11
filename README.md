# Whisper Vulkan for Home Assistant

Experimental Home Assistant App for local English speech recognition using upstream whisper.cpp and Mesa Vulkan. Target hardware: AMD Ryzen 7 4800U integrated graphics (PCI 1002:1636), amdgpu, HAOS generic-x86-64. This is an App repository, not a fork of the inference engine. Piper remains a separate text-to-speech App.

## Status

Initial implementation. Hardware visibility was reported by the owner; Vulkan inference, latency, HAOS installation, and AppArmor compatibility have **not yet been verified on the target**. A CI build does not establish hardware support. Do not replace your working voice pipeline until acceptance testing passes.

## Install for testing

Add `https://github.com/trooperthorn/ha_app_whisper_vulkan` under Settings → Apps → App store → Repositories. Enable advanced mode in your Home Assistant profile to see experimental Apps. Install **Whisper Vulkan**. This initial version builds locally; allow time for C++ compilation. No prebuilt image or release is promised yet.

Start with the default `small.en`, English, Vulkan, four threads and beam size five. The model downloads from Hugging Face on first start and stays in `/data/models`; cached models are reused without online checks. Set `download_model: false` to prevent new downloads. Model files are included in backups in this initial version.

After the logs report readiness, add the **Wyoming Protocol** integration manually with the App hostname shown on its Info page (typically `<repository-prefix>-whisper-vulkan`, using hyphens), port `10300`. Host port publishing is disabled by default; internal App networking works without it. Select this service in a separate Assist pipeline and select your existing Piper for speech output. Automatic discovery is not implemented yet.

## Configuration

| Option | Default | Meaning |
|---|---|---|
| model | small.en | tiny.en, base.en, or small.en |
| language | en | English only in this initial version |
| backend | vulkan | Vulkan preflight required, or explicit cpu mode |
| threads | 4 | 1–16 CPU threads; leave capacity for other Apps |
| beam_size | 5 | 1–10; lower may trade accuracy for latency |
| download_model | true | Allow initial download if model is missing |

The CPU mode passes `--no-gpu`. Vulkan mode requires an AMD device in `vulkaninfo --summary`. Preflight alone is not proof of inference offload: inspect whisper.cpp startup logs and benchmark both modes. Automatic fallback is intentionally absent so a GPU failure stays visible.

## Design and boundaries

The model stays loaded in a loopback-only whisper.cpp HTTP server. A Wyoming bridge accepts 16 kHz, 16-bit mono PCM, caps each recording at 60 seconds, serializes inference and returns errors for failed requests. No recordings are saved and the bridge does not log transcripts. The upstream engine's logging must also be reviewed during hardware validation. Speech requests use loopback only; model provisioning uses HTTPS.

GPU access uses Home Assistant's `video: true`; no host network, Docker socket, full access, kernel module privileges, or Supervisor token is requested. Default AppArmor remains enabled. The HTTP engine is not exposed outside the container. Do not expose Wyoming directly to the internet: the protocol endpoint has no authentication.

## Acceptance plan

1. Build image and run protocol tests in CI.
2. Install on HAOS 18.1; verify startup under AppArmor and AMD device enumeration.
3. Confirm engine logs identify Vulkan GPU use; successful device enumeration alone is insufficient.
4. Transcribe identical recordings in CPU and Vulkan modes with small.en, four threads and beam five. Record cold/warm timings, accuracy, RAM, and errors. Compare existing Faster Whisper separately since engines differ.
5. Verify real Assist → transcription → action → Piper response, restart, cached offline startup, malformed/oversized input, and recovery after child failure.
6. Check kiosk/video playback and other existing services under concurrent load.
7. Publish a prebuilt version only after build and smoke checks; label hardware support verified only after target results are captured.

## Upstream

- whisper.cpp v1.9.4, pinned commit `927cfce34f31707e17f2bff35c349632fb9e2c3a`: https://github.com/ggml-org/whisper.cpp
- Wyoming protocol library 1.10.2: https://github.com/OHF-Voice/wyoming
- Home Assistant App configuration: https://developers.home-assistant.io/docs/apps/configuration/

Whisper.cpp and Wyoming retain their respective upstream licenses. The original App code is MIT licensed. Model weights and distribution licenses remain upstream.
