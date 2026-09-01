#!/usr/bin/env bash
# Build the IEEE manuscript. Uses the unmodified IEEEtran class and IEEEtran.bst.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$PATH:$HOME/Library/TinyTeX/bin/universal-darwin"
pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null || true
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
echo "built: $(pwd)/main.pdf"
