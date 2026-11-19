#!/usr/bin/env bash
# Re-download the public arXiv evaluation PDFs (in case they are not redistributed with the submission).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p sample_pdfs long_pdfs domain_pdfs
for id in 1706.03762 1810.04805 1908.10084 2005.11401; do
  curl -sSL -o "sample_pdfs/arxiv_$id.pdf" "https://arxiv.org/pdf/$id"
done
for id in 2005.14165 2307.09288; do
  curl -sSL -o "long_pdfs/arxiv_$id.pdf" "https://arxiv.org/pdf/$id"
done
# Corpus C: NIST Cybersecurity Framework 2.0 (US government work, public domain)
curl -sSL -o "domain_pdfs/NIST.CSWP.29.pdf" "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf"
echo "Downloaded: $(ls sample_pdfs long_pdfs domain_pdfs | grep -c pdf) PDFs"
