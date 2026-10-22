#!/usr/bin/env bash
# Build the Linux runtimes and fetch official weights (idempotent, stamped by requirements hash).
#   /opt/musicmix/venv      core: ingest, Beat This!, Demucs, faster-whisper, Silero, spaCy, LinkSeg, SSM-Net, Audiobox
#   /opt/musicmix/venv-np1  numpy<2 stack for CUE-DETR and SongEval/MuQ (transformers 4.42.3)
# MUSICMIX_CRITICS=1 also downloads the ~1.9 GB critic weights (Audiobox, MuQ, SongEval).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "$here")"
OPT="${MUSICMIX_HOME:-/opt/musicmix}"
INDEX=(--extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match)

build() {  # build <venv> <stamp-source files...> -- <requirements files...>
  local venv=$1; shift; local sources=(); while [ "$1" != "--" ]; do sources+=("$1"); shift; done; shift
  local want; want=$(cat "${sources[@]}" | sha256sum | cut -d' ' -f1)
  if [ -x "$venv/bin/python" ] && [ "$(cat "$venv.stamp" 2>/dev/null)" = "$want" ]; then return 0; fi
  echo "[musicmix-python] building $venv"
  uv venv --clear --python 3.11 "$venv"
  for req in "$@"; do uv pip install --python "$venv/bin/python" -r "$req" "${INDEX[@]}"; done
  uv pip install --python "$venv/bin/python" --no-deps -e "$repo"
  echo "$want" > "$venv.stamp"
}

build "$OPT/venv" "$here/requirements-core.txt" "$here/requirements-critics.txt" "$here/install_python.sh" -- \
  "$here/requirements-core.txt" "$here/requirements-critics.txt"
build "$OPT/venv-np1" "$here/requirements-legacy-np1.txt" "$here/install_python.sh" -- \
  "$here/requirements-legacy-np1.txt"

critics=(); [ "${MUSICMIX_CRITICS:-0}" = "1" ] && critics=(--critics)
"$OPT/venv/bin/python" "$here/fetch_models.py" --target "$OPT/models" "${critics[@]}"
