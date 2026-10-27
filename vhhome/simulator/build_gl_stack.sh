#!/usr/bin/env bash
# Build a private Mesa + Xvfb stack for the simulator when you cannot install packages as root (or when the
# machine's /usr is wiped between sessions). Downloads the Ubuntu .debs for the packages the simulator needs
# and unpacks them under $GLSTACK_PREFIX (default ~/.vhhome/glstack); writes simulator/gl_stack_env.sh, which
# launch_simulators.sh sources automatically when the system stack is missing.
#
#   bash simulator/build_gl_stack.sh            # one-time, needs network, ~2 min
#   source simulator/gl_stack_env.sh            # optional: put the stack on PATH in your own shell
#
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"   # absolute: we cd into $DEBS below
PREFIX="${GLSTACK_PREFIX:-$HOME/.vhhome/glstack}"
DEBS="$PREFIX/debs"
ROOT="$PREFIX/root"
mkdir -p "$DEBS" "$ROOT"

# x11-xkb-utils supplies xkbcomp. libxkbfile1 alone is NOT enough: Xvfb shells out to the
# xkbcomp BINARY to build its keymap and aborts if it is absent, so a stack without it builds
# fine and then refuses to start a display.
# mesa 25.2 split the gallium renderer into its own package (mesa-libgallium) and moved to
# LLVM 20. Without mesa-libgallium, libllvm20 and libpciaccess0, libGLX_mesa/libEGL_mesa have
# unmet dependencies, glvnd silently falls back to indirect rendering, and every GLX query
# answers "couldn t find RGB GLX visual or fbconfig" with no loader error printed anywhere.
# libllvm19 is NOT a substitute: mesa 25.2.8 links libLLVM.so.20.1 by soname.
PKGS="xvfb xserver-common x11-common libxfont2 libfontenc1 libpixman-1-0 libunwind8 libxkbfile1
x11-xkb-utils
      xkb-data libgl1 libglx-mesa0 libglapi-mesa libgl1-mesa-dri libglu1-mesa mesa-utils
      mesa-libgallium libllvm20 libpciaccess0
      mesa-utils-bin libglvnd0 libglx0 libgles2 libegl1 libegl-mesa0 libdrm2 libdrm-amdgpu1
      libdrm-intel1 libdrm-nouveau2 libdrm-radeon1 libllvm19 libsensors5 libz3-4 libedit2
      libxcb-dri2-0 libxcb-dri3-0 libxcb-glx0 libxcb-present0 libxcb-randr0 libxcb-shm0
      libxcb-sync1 libxcb-xfixes0 libxshmfence1 libxext6 libxrender1 libxtst6 libxi6
      libxcursor1 libxrandr2 libxinerama1 libxxf86vm1 libxfixes3 libxdamage1 libxcomposite1
      libxss1 libasound2t64 libnss3 libgbm1 libwayland-server0 libexpat1 libffi8 libzstd1
      libx11-6 libx11-xcb1 libxcb1 libxau6 libxdmcp6 libbsd0 libmd0 libxml2 libicu74"

echo "[glstack] downloading .debs -> $DEBS"
cd "$DEBS"
for p in $PKGS; do
  ls "${p}"_*.deb >/dev/null 2>&1 && continue
  apt-get download "$p" >/dev/null 2>&1 || echo "[glstack]   skip $p (unavailable)"
done

echo "[glstack] extracting -> $ROOT"
for d in "$DEBS"/*.deb; do dpkg -x "$d" "$ROOT"; done

cat > "$HERE/gl_stack_env.sh" <<EOF
# Source this at the start of every session — restores the X/GL stack with no apt, no network.
# Built by simulator/build_gl_stack.sh.
export GLSTACK_ROOT="$ROOT"
export PATH="\$GLSTACK_ROOT/usr/bin:\$PATH"
export LD_LIBRARY_PATH="\$GLSTACK_ROOT/usr/lib/x86_64-linux-gnu:\$GLSTACK_ROOT/usr/lib:\${LD_LIBRARY_PATH:-}"
# mesa needs its DRI drivers and llvmpipe explicitly when it is not at the system prefix
export LIBGL_DRIVERS_PATH="\$GLSTACK_ROOT/usr/lib/x86_64-linux-gnu/dri"
export XKB_BINDIR="\$GLSTACK_ROOT/usr/bin"
export XKB_CONFIG_ROOT="\$GLSTACK_ROOT/usr/share/X11/xkb"
export XORG_FONTPATH="\$GLSTACK_ROOT/usr/share/fonts/X11"
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
EOF

echo "[glstack] wrote $HERE/gl_stack_env.sh"
# shellcheck disable=SC1090
. "$HERE/gl_stack_env.sh"
echo "[glstack] Xvfb  -> $(command -v Xvfb || echo MISSING)"
echo "[glstack] glxinfo -> $(command -v glxinfo || echo MISSING)"
du -sh "$ROOT" | sed 's/^/[glstack] size /'
