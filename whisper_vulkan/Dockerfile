FROM debian:trixie-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates cmake ninja-build g++ libvulkan-dev glslc && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git init && git remote add origin https://github.com/ggml-org/whisper.cpp.git && git fetch --depth 1 origin 927cfce34f31707e17f2bff35c349632fb9e2c3a && git checkout FETCH_HEAD
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF -DWHISPER_BUILD_TESTS=OFF && cmake --build build --target whisper-server -j 2

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates libgomp1 libvulkan1 mesa-vulkan-drivers vulkan-tools && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/bin/whisper-server /usr/local/bin/whisper-server
COPY --from=build /src/LICENSE /usr/share/doc/whisper-cpp/LICENSE
WORKDIR /app
COPY requirements.txt .
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt
COPY app.py bridge.py ./
ENV PYTHONUNBUFFERED=1 XDG_CACHE_HOME=/data/cache
LABEL io.hass.type="app" io.hass.arch="amd64" io.hass.version="0.1.0"
CMD ["/opt/venv/bin/python", "/app/app.py"]
