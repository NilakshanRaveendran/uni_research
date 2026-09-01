#!/usr/bin/env bash
# Build the dissertation. Requires a TeX distribution with apacite and the packages listed in
# uwuthesis.sty. On this machine TinyTeX lives in ~/Library/TinyTeX.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$PATH:$HOME/Library/TinyTeX/bin/universal-darwin"
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
bibtex main >/dev/null || true
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
echo "built: $(pwd)/main.pdf  ($(pdfinfo main.pdf 2>/dev/null | awk '/Pages/{print $2}' || echo '?') pages)"
