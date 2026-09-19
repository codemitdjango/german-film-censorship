# German Film Censorship

This repository contains data, scripts, and notebooks for the computational
study of historical German film censorship. It focuses on film approval cards
from the Berlin Film Review Office, collection **R 9346-I**, held by the
[German Federal Archives](https://invenio.bundesarchiv.de/invenio/login.xhtml).

The project combines archival data collection, OCR, structured extraction with
a vision-language model, and statistical/content analysis.

## Workflow

1. Download digitized approval cards and metadata.
2. Transcribe page images with a vision-language model.
3. Merge page-level OCR into structured film records.
4. Compare OCR output with corrected reference transcriptions.
5. Analyze metadata and film content.

## Repository Structure

```text

data/
├── 01_raw/       # approval-card images and metadata (not public in repo)
├── 02_ocr/       # ocr output and corrected references (not public in repo)
├── 03_processed/ # merged, structured film records (not public in repo)
└── groundtruth/  # manually created ground truth

src/
├── get_data/     # Scraper, archive extraction, and sampling
├── pipeline/     # OCR pipeline, prompts, schemas, and few-shot examples
└── analysis/     # Jupyter notebooks, evaluation, and plots
```

The committed data is a small reference dataset. The full dataset is not part
of the repository for data protection and usage-rights reasons.


## Source and License

The archival material comes from the German Federal Archives, collection
R 9346-I. Use and redistribution of the digitized documents are subject to the
archives' terms and any applicable copyright restrictions.

This repository does not currently contain a `LICENSE` file. The license for
the code and derived data must be considered separately from the rights to the
archival material.
