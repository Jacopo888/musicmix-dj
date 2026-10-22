#!/usr/bin/env bash
# Reference setup for a fresh Ubuntu 24.04 machine (run as root, CPU only):
#   1. system packages: ffmpeg/ffprobe (decoding), libsndfile1, jq, rubberband-cli (time-stretch)
#   2. uv (reproducible venvs with a pinned CPython 3.11)
#   3. Python venvs + pinned model weights via env/install_python.sh
# Everything heavy goes under $MUSICMIX_HOME (default /opt/musicmix).
#   sudo MUSICMIX_HOME=/opt/musicmix bash env/setup_ubuntu.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
OPT="${MUSICMIX_HOME:-/opt/musicmix}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OPT"
log() { echo "[musicmix-setup] $*"; }

need=()
for pkg in ffmpeg libsndfile1 jq rubberband-cli; do dpkg -s "$pkg" >/dev/null 2>&1 || need+=("$pkg"); done
if [ ${#need[@]} -gt 0 ]; then
  log "apt install ${need[*]}"
  apt-get update -qq && apt-get install -y -qq --no-install-recommends "${need[@]}"
fi

if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  python3 -m pip install -q --break-system-packages uv
fi

MUSICMIX_HOME="$OPT" bash "$here/install_python.sh"
log "done: core venv $OPT/venv, numpy<2 venv $OPT/venv-np1, weights $OPT/models"
log "All-In-One/madmom/essentia need a separate venv ($OPT/venv-allin1): see env/README.md"
