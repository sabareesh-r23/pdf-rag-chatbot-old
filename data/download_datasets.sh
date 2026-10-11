#!/usr/bin/env bash
# Re-download the public arXiv evaluation PDFs (in case they are not redistributed with the submission).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p sample_pdfs long_pdfs
for id in 1706.03762 1810.04805 1908.10084 2005.11401; do
  curl -sSL -o "sample_pdfs/arxiv_$id.pdf" "https://arxiv.org/pdf/$id"
done
for id in 2005.14165 2307.09288; do
  curl -sSL -o "long_pdfs/arxiv_$id.pdf" "https://arxiv.org/pdf/$id"
done
echo "Downloaded: $(ls sample_pdfs long_pdfs | grep -c pdf) PDFs"
