from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


JsonInput = str | Path | list[dict[str, Any]] | dict[str, Any]

def _load_json(source: JsonInput) -> Any:
    if isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8") as file:
            return json.load(file)
    return source


def _get_pages(data: Any, source_name: str) -> list[dict[str, Any]]:
    if isinstance(data, list):
        pages = data
    elif isinstance(data, dict):
        pages = next(
            (
                data[key]
                for key in ("pages", "data", "results")
                if isinstance(data.get(key), list)
            ),
            None,
        )
        if pages is None and (
            "transcription" in data or "text" in data
        ):
            pages = [data]
    else:
        pages = None

    if pages is None or not all(isinstance(page, dict) for page in pages):
        raise ValueError(
            f"{source_name} muss eine Liste von Seiten oder ein Seitenobjekt enthalten."
        )
    return pages


def _page_key(page: dict[str, Any], index: int) -> str:
    if page.get("_page") is not None:
        return f"page:{page['_page']}"
    if page.get("_filename"):
        return f"filename:{page['_filename']}"
    return f"index:{index}"


def _index_pages(pages: Sequence[dict[str, Any]], source_name: str) -> dict[str, dict[str, Any]]:
    indexed = {}
    for index, page in enumerate(pages, start=1):
        key = _page_key(page, index)
        if key in indexed:
            raise ValueError(f"Doppelte Seitenkennung in {source_name}: {key}")
        indexed[key] = page
    return indexed


def _transcription(page: dict[str, Any] | None, normalize_whitespace: bool) -> str:
    if page is None:
        return ""

    text = page.get("transcription", page.get("text", ""))
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)

    if normalize_whitespace:
        text = " ".join(text.split())
    return text


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))

    for reference_index, reference_token in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_token in enumerate(hypothesis, start=1):
            substitution_cost = reference_token != hypothesis_token
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1] + substitution_cost,
                )
            )
        previous = current

    return previous[-1]


def _rate(distance: int, reference_length: int) -> float | None:
    if reference_length == 0:
        return 0.0 if distance == 0 else None
    return distance / reference_length


def _format_rate(rate: float | None) -> str:
    if rate is None:
        return "n/a"
    return f"{rate:.4f} ({rate * 100:.2f}%)"

def evaluate_transcriptions(
    ground_truth_json: JsonInput,
    hypothesis_json: JsonInput,
    *,
    normalize_whitespace: bool = False,
) -> dict[str, Any]:
    reference_data = _load_json(ground_truth_json)
    hypothesis_data = _load_json(hypothesis_json)

    reference_pages = _get_pages(reference_data, "ground_truth_json")
    hypothesis_pages = _get_pages(hypothesis_data, "hypothesis_json")
    reference_by_key = _index_pages(reference_pages, "ground_truth_json")
    hypothesis_by_key = _index_pages(hypothesis_pages, "hypothesis_json")

    page_keys = list(reference_by_key)
    page_keys.extend(
        key for key in hypothesis_by_key if key not in reference_by_key
    )

    page_results = []
    total_char_distance = 0
    total_reference_chars = 0
    total_word_distance = 0
    total_reference_words = 0

    print("\nSeitenauswertung")
    print("-" * 80)

    for key in page_keys:
        reference_page = reference_by_key.get(key)
        hypothesis_page = hypothesis_by_key.get(key)
        reference_text = _transcription(reference_page, normalize_whitespace)
        hypothesis_text = _transcription(hypothesis_page, normalize_whitespace)

        reference_chars = list(reference_text)
        hypothesis_chars = list(hypothesis_text)
        reference_words = reference_text.split()
        hypothesis_words = hypothesis_text.split()

        char_distance = _edit_distance(reference_chars, hypothesis_chars)
        word_distance = _edit_distance(reference_words, hypothesis_words)
        page_cer = _rate(char_distance, len(reference_chars))
        page_wer = _rate(word_distance, len(reference_words))

        total_char_distance += char_distance
        total_reference_chars += len(reference_chars)
        total_word_distance += word_distance
        total_reference_words += len(reference_words)

        page_number = (reference_page or hypothesis_page or {}).get("_page")
        filename = (reference_page or hypothesis_page or {}).get("_filename")
        label = f"Seite {page_number}" if page_number is not None else key
        if filename:
            label += f" ({filename})"

        if reference_page is None:
            status = "nur in hypothesis vorhanden"
        elif hypothesis_page is None:
            status = "in hypothesis fehlend"
        elif not reference_text and not hypothesis_text:
            status = "beide leer"
        elif not reference_text:
            status = "Text in leerer Referenzseite"
        else:
            status = ""

        status_suffix = f" | {status}" if status else ""
        print(
            f"{label}: CER = {_format_rate(page_cer)}, "
            f"WER = {_format_rate(page_wer)}{status_suffix}"
        )

        page_results.append(
            {
                "key": key,
                "page": page_number,
                "filename": filename,
                "cer": page_cer,
                "wer": page_wer,
                "char_distance": char_distance,
                "reference_chars": len(reference_chars),
                "word_distance": word_distance,
                "reference_words": len(reference_words),
                "reference_page_present": reference_page is not None,
                "hypothesis_page_present": hypothesis_page is not None,
            }
        )

    document_cer = _rate(total_char_distance, total_reference_chars)
    document_wer = _rate(total_word_distance, total_reference_words)

    print("\nGesamtes Dokument")
    print("-" * 80)
    print(f"CER = {_format_rate(document_cer)}")
    print(f"WER = {_format_rate(document_wer)}")
    print(f"Seiten in Referenz: {len(reference_pages)}")
    print(f"Seiten in Hypothese: {len(hypothesis_pages)}")

    return {
        "pages": page_results,
        "document": {
            "cer": document_cer,
            "wer": document_wer,
            "char_distance": total_char_distance,
            "reference_chars": total_reference_chars,
            "word_distance": total_word_distance,
            "reference_words": total_reference_words,
            "reference_pages": len(reference_pages),
            "hypothesis_pages": len(hypothesis_pages),
        },
    }

evaluate_transcriptions("/home/marisa/Dokumente/GitHub/german-film-censorship/data/groundtruth/02_ocr/corrected_R 9346-I_109_ocr.json", "/home/marisa/Dokumente/GitHub/german-film-censorship/data/groundtruth/02_ocr/R 9346-I_109_ocr.json")
evaluate_transcriptions("/home/marisa/Dokumente/GitHub/german-film-censorship/data/groundtruth/02_ocr/corrected_R 9346-I_13991_ocr.json", "/home/marisa/Dokumente/GitHub/german-film-censorship/data/groundtruth/02_ocr/R 9346-I_13991_ocr.json")
evaluate_transcriptions("/home/marisa/Dokumente/GitHub/german-film-censorship/data/groundtruth/02_ocr/corrected_R 9346-I_22050_ocr.json", "/home/marisa/Dokumente/GitHub/german-film-censorship/data/groundtruth/02_ocr/R 9346-I_22050_ocr.json")
evaluate_transcriptions("/home/marisa/Dokumente/GitHub/german-film-censorship/data/groundtruth/02_ocr/corrected_R 9346-I_27959_ocr.json", "/home/marisa/Dokumente/GitHub/german-film-censorship/data/groundtruth/02_ocr/R 9346-I_27959_ocr.json")
evaluate_transcriptions("/home/marisa/Dokumente/GitHub/german-film-censorship/data/groundtruth/02_ocr/corrected_R 9346-I_34828_ocr.json", "/home/marisa/Dokumente/GitHub/german-film-censorship/data/groundtruth/02_ocr/R 9346-I_34828_ocr.json")
