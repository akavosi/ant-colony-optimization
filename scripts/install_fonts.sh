#!/usr/bin/env bash
# Installs the display fonts used throughout this project's visualizations
# (Fraunces for titles, IBM Plex Sans/Mono for body/numeric text), so
# rendered output matches exactly rather than silently falling back to
# generic system fonts (matplotlib does this gracefully -- see
# src/viz/fonts.py -- but the look will differ without them).
#
# Usage:
#   bash scripts/install_fonts.sh
set -euo pipefail

FONT_DIR="${HOME}/.fonts"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

mkdir -p "${FONT_DIR}"

echo "Downloading IBM Plex Mono..."
curl -sL "https://github.com/IBM/plex/releases/download/%40ibm%2Fplex-mono%401.1.0/ibm-plex-mono.zip" \
  -o "${TMP_DIR}/plex-mono.zip"

echo "Downloading IBM Plex Sans..."
curl -sL "https://github.com/IBM/plex/releases/download/%40ibm%2Fplex-sans%401.1.0/ibm-plex-sans.zip" \
  -o "${TMP_DIR}/plex-sans.zip"

echo "Downloading Fraunces..."
curl -sL "https://raw.githubusercontent.com/google/fonts/main/ofl/fraunces/Fraunces%5BSOFT%2CWONK%2Copsz%2Cwght%5D.ttf" \
  -o "${TMP_DIR}/fraunces_var.ttf"

echo "Extracting..."
unzip -q "${TMP_DIR}/plex-mono.zip" -d "${TMP_DIR}/mono"
unzip -q "${TMP_DIR}/plex-sans.zip" -d "${TMP_DIR}/sans"

echo "Installing to ${FONT_DIR}..."
find "${TMP_DIR}/mono" "${TMP_DIR}/sans" -iname "*.ttf" \
  | grep -E "(Regular|Medium|SemiBold)\.ttf$" | grep -vi italic \
  | xargs -I{} cp {} "${FONT_DIR}/"
cp "${TMP_DIR}/fraunces_var.ttf" "${FONT_DIR}/"

python3 - <<'PY'
import glob
import os
import matplotlib.font_manager as fm

for path in glob.glob(os.path.expanduser("~/.fonts/*.ttf")):
    fm.fontManager.addfont(path)

names = sorted({f.name for f in fm.fontManager.ttflist if "Plex" in f.name or "Fraunces" in f.name})
print("Registered fonts:", names)
assert {"IBM Plex Mono", "IBM Plex Sans", "Fraunces"}.issubset(set(names)), \
    "Font installation did not produce all three expected font families."
PY

echo "Done. Fonts installed to ${FONT_DIR}."
