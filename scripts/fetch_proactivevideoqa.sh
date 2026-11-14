#!/usr/bin/env bash
# Clone the ProactiveVideoQA scorer at a pinned commit into third_party/ (gitignored).
set -euo pipefail
dest="$(dirname "$0")/../third_party/ProactiveVideoQA"
rev=6fe9a33
[ -d "$dest/.git" ] || git clone https://github.com/yellow-binary-tree/ProactiveVideoQA "$dest"
git -C "$dest" fetch -q origin && git -C "$dest" checkout -q "$rev"
echo "ProactiveVideoQA at $(git -C "$dest" rev-parse --short HEAD) -> $dest"
