import base64
import json
import requests
import os
import re
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# 
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
IMAGE_FOLDER = PROJECT_ROOT / "data" / "01_raw" / "R_9346-I_Zulassungskarten"
PROMPT_DIR = SCRIPT_DIR / "prompts"

API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY", "fallback")
MODEL = "google/gemma-4-12b"
TEMPERATURE = 0.1 # bei 0 hängt das Modell im LOOP
TIMEOUT_SECONDS = 600

# Prompts
PAGE_PROMPT = (PROMPT_DIR / "page_prompt.txt").read_text(encoding="utf-8")
MERGE_PROMPT = (PROMPT_DIR / "merge_prompt.txt").read_text(encoding="utf-8")
# JSON
PAGE_SCHEMA = json.loads((PROMPT_DIR / "page_schema.json").read_text(encoding="utf-8"))
MERGE_SCHEMA = json.loads((PROMPT_DIR / "merge_schema.json").read_text(encoding="utf-8"))

def encode_image_b64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_model(content, schema) -> dict:
    payload = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "messages": [
            { "role": "user", "content": content }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "ocr_extraktion",
                "strict": True,
                "schema": schema
            }
        }
    }

    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=TIMEOUT_SECONDS
    )
    response.raise_for_status()

    raw_text = response.json()["choices"][0]["message"]["content"]
    return json.loads(raw_text)


def process_page(image_path: Path, page_number: int) -> dict:
    print(f"Verarbeite Seite {page_number}: {image_path.name}")
    b64 = encode_image_b64(image_path)
    content = [
        {"type": "text", "text": PAGE_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
    ]

    parsed = call_model(content, PAGE_SCHEMA)
    parsed["_seite"] = page_number
    parsed["_dateiname"] = image_path.name
    return parsed


def merge_pages(page_results: list[dict]) -> dict:
    # hier hin schreiben bei welchen dok man ist
    print("Führe alle Seiten zu einem Dokument zusammen...")

    pages_json = json.dumps(page_results, ensure_ascii=False, indent=2)
    content = MERGE_PROMPT.replace("{PAGES_JSON}", pages_json)
    return call_model(content, MERGE_SCHEMA)


def get_sorted_images(dir_path: Path) -> list[Path]:
    # search every image in a folder and sort
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    images = []
    for ext in extensions:
        images.extend(dir_path.glob(ext))
    
    #sort
    def natural_sort_key(path: Path):
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", path.name)
        ]

    return sorted(images, key=natural_sort_key)


def process_document_directory(doc_dir: Path):
    # one document folder
    print(f"Starte Verarbeitung für Dokument: {doc_dir.name}")

    image_files = get_sorted_images(doc_dir)

    if not image_files:
        print(f"Keine Bilder in {doc_dir.name} gefunden. Überspringe.")
        return
    
    print(f"{len(image_files)} Bild(er) gefunden. Starte OCR...")

    page_results = []
    for i, image_file in enumerate(image_files, start=1):
        try:
            page_result = process_page(image_file, i)
        except requests.exceptions.RequestException as e:
            print(f"Fehler bei Seite {i} ({image_file.name}): {e}")
            page_result = {
                "_seite": i,
                "_dateiname": image_file.name,
                "_error": str(e),
            }
        page_results.append(page_result)

    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    # TODO rename 
    intermediate_file = output_dir / f"{doc_dir.name}_seiten_einzeln.json"
    final_file = output_dir / f"{doc_dir.name}_zensurkarte.json"

    with open(intermediate_file, "w", encoding="utf-8") as f:
        json.dump(page_results, f, ensure_ascii=False, indent=2)

    try:
        merged = merge_pages(page_results)
    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Zusammenführen von {doc_dir.name}: {e}")
        return
    
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Fertig. Ergebnis gespeichert in {final_file}")


def main():
    if not IMAGE_FOLDER.exists() or not IMAGE_FOLDER.is_dir():
        print(f"Hauptverzeichnis {IMAGE_FOLDER} existiert nicht.")
        return
    
    doc_directories = sorted([d for d in IMAGE_FOLDER.iterdir() if d.is_dir()])

    if not doc_directories:
        print(f"Keine Dokumenteordner in {IMAGE_FOLDER} gefunden.")
        return
    
    print(f"{len(doc_directories)} Dokumentenordner gefunden.")

    # Pipeline for every Folder
    for doc_dir in doc_directories:
        process_document_directory(doc_dir)


if __name__ == "__main__":
    main()