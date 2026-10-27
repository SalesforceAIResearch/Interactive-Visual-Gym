#!/usr/bin/env bash
# Fetch VirtualHome into simulator/: the Python client (git) and the v2.3.0 Linux Unity build.
#
#   bash simulator/install_virtualhome.sh --unity-archive /path/to/linux_exec.zip
#
# Environment: VHHOME_SIMULATOR (destination), VHHOME_UNITY_URL (optional archive URL),
# VHHOME_UNITY_SHA256 (optional archive checksum). --client-only needs no Unity assets.
# The Unity build is a few GB and is never committed. After this, `bash simulator/launch_simulators.sh 1`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM="${VHHOME_SIMULATOR:-$HERE}"
URL="${VHHOME_UNITY_URL:-}"
ARCHIVE="${VHHOME_UNITY_ARCHIVE:-}"
CLIENT_URL="${VHHOME_CLIENT_URL:-https://github.com/xavierpuigf/virtualhome.git}"
CLIENT_REV="${VHHOME_CLIENT_REV:-58970fd80951c2eaa1af713e0917d1a105353ad8}"
CLIENT_ONLY=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --client-only) CLIENT_ONLY=1; shift ;;
    --unity-archive) [ "$#" -ge 2 ] || { echo "--unity-archive needs a ZIP path" >&2; exit 2; }; ARCHIVE="$2"; shift 2 ;;
    --help|-h) echo "Usage: bash simulator/install_virtualhome.sh [--client-only] [--unity-archive PATH]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$SIM"

if [ -d "$SIM/virtualhome/simulation/unity_simulator" ] || [ -d "$SIM/virtualhome/virtualhome/simulation/unity_simulator" ]; then
  echo "[install] client present: $SIM/virtualhome"
elif [ -e "$SIM/virtualhome" ]; then
  echo "[install] $SIM/virtualhome exists but has no supported client layout; move it aside or choose VHHOME_SIMULATOR" >&2
  exit 1
else
  echo "[install] cloning the VirtualHome client into $SIM/virtualhome"
  git clone --no-checkout --depth 1 "$CLIENT_URL" "$SIM/virtualhome"
  git -C "$SIM/virtualhome" fetch --depth 1 origin "$CLIENT_REV"
  git -C "$SIM/virtualhome" checkout --detach FETCH_HEAD
fi
[ "$CLIENT_ONLY" = 1 ] && { echo "[install] client ready"; exit 0; }

EXE="$SIM/unity_vol/linux_exec.v2.3.0.x86_64"
if [ -x "$EXE" ]; then
  echo "[install] Unity build present: $EXE"
else
  if [ -z "$ARCHIVE" ] && [ -z "$URL" ]; then
    echo "[install] VirtualHome v2.3.0 Unity assets are required. Supply --unity-archive /path/to/linux_exec.zip or VHHOME_UNITY_URL." >&2
    echo "[install] The upstream website's old download is unavailable; the installer will not substitute a different simulator version." >&2
    exit 1
  fi
  TMP="$(mktemp -d "$SIM/unity_dl.XXXX")"
  trap 'test -z "${TMP:-}" || rm -rf -- "$TMP"' EXIT
  if [ -n "$ARCHIVE" ]; then
    [ -f "$ARCHIVE" ] || { echo "[install] archive not found: $ARCHIVE" >&2; exit 1; }
    cp -- "$ARCHIVE" "$TMP/linux_exec.zip"
  else
    echo "[install] downloading the Unity build"
    curl -L --fail --retry 3 --connect-timeout 30 -o "$TMP/linux_exec.zip" "$URL"
  fi
  if [ -n "${VHHOME_UNITY_SHA256:-}" ]; then
    ACTUAL="$(sha256sum "$TMP/linux_exec.zip" | cut -d ' ' -f 1)"
    [ "$ACTUAL" = "$VHHOME_UNITY_SHA256" ] || { echo "[install] Unity archive checksum mismatch" >&2; exit 1; }
  fi
  unzip -q "$TMP/linux_exec.zip" -d "$TMP/unzipped"
  FOUND="$(find "$TMP/unzipped" -name 'linux_exec.v2.3.0.x86_64' -print -quit)"
  [ -n "$FOUND" ] || { echo "[install] archive does not contain VirtualHome v2.3.0" >&2; exit 1; }
  [ -d "$(dirname "$FOUND")/linux_exec.v2.3.0_Data" ] || { echo "[install] Unity data directory is missing" >&2; exit 1; }
  if [ -d "$SIM/unity_vol" ] && [ -n "$(ls -A "$SIM/unity_vol")" ]; then
    echo "[install] refusing to overwrite nonempty $SIM/unity_vol; choose another VHHOME_SIMULATOR" >&2
    exit 1
  fi
  mkdir -p "$SIM/unity_vol"
  mv "$(dirname "$FOUND")"/* "$SIM/unity_vol/"
  chmod +x "$EXE"
fi
mkdir -p "$SIM/unity_vol/Output"
echo "[install] done. Next: bash simulator/launch_simulators.sh 1   (501 on the port = healthy)"
