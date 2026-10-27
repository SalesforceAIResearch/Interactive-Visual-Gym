# vhsim:f2 — headless VirtualHome Unity render via software GL (llvmpipe).
# ubuntu 20.04 ships mesa 21.2.6 -> llvmpipe gives GL 4.5 core, enough for the
# VirtualHome v2.3.0 Unity player in -batchmode. No GPU/RT needed (CPU software GL).
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      xvfb \
      libgl1-mesa-glx libgl1-mesa-dri libglu1-mesa mesa-utils \
      libx11-6 libxext6 libxrender1 libxtst6 libxi6 libxcursor1 \
      libxrandr2 libxinerama1 libxxf86vm1 libxfixes3 libxdamage1 libxcomposite1 \
      libxss1 libgconf-2-4 libasound2 libpulse0 libnss3 libgbm1 \
      libc6 libstdc++6 ca-certificates curl python3 \
    && rm -rf /var/lib/apt/lists/*
# software GL: force llvmpipe regardless of host
ENV LIBGL_ALWAYS_SOFTWARE=1 \
    GALLIUM_DRIVER=llvmpipe \
    LP_NUM_THREADS=9 \
    DISPLAY=:1
WORKDIR /unity_vol
