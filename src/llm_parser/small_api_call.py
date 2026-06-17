import base64
import json
import requests
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Config
IMAGE_FOLDER = r"./data/01_raw"
API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY", "fallback")
MODEL = "google/gemma-4-12b"
OUTPUT_FILE = "zensurkarte.json"
TEMPERATURE = 0.1 # bei 0 hängt das Modell im LOOP
TIMEOUT_SECONDS = 600

# Prompts
PAGE_PROMPT = """
Das folgende Bild zeigt EINE Seite einer deutschen Filmzulassungskarte,
ausgestellt zwischen 1920 und 1945.

Aufgabe:

1. Führe eine vollständige OCR dieser einen Seite durch.
2. Berücksichtige Drucktext, Stempel, Randnotizen und Handschrift.
3. Bewahre die originale Schreibweise.
4. Ergänze keine Informationen, die nicht im Bild stehen.
5. Unsichere Lesungen markiere im Text mit [unsicher: ...].
"""

MERGE_PROMPT = """
Die folgenden Transkriptionen stammen von einzelnen, aufeinanderfolgenden
Seiten EINER deutschen Filmzulassungskarte (1920–1945). Du erhältst sie als
JSON-Liste, eine Transkription pro Seite, in der richtigen Seitenreihenfolge.

Aufgabe:

1. Fasse alle Seiten zu einem zusammenhängenden Dokument zusammen.
2. Falls Informationen über mehrere Seiten verteilt sind, führe sie logisch
   zusammen (z.B. wenn ein Feld auf Seite 1 beginnt und auf Seite 2 fortgesetzt wird).
3. Ergänze keine Informationen, die nicht in den Transkriptionen stehen.
4. Übernimm unsichere Lesungen und handschriftliche Notizen aus den Seiten.

Falls weitere Felder im Dokument vorkommen, ergänze sie zusätzlich.

Seiten-Transkriptionen:
{PAGES_JSON}
"""

PAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "transkription": {
            "type": "string",
            "description": "Vollständiger erkannter Text dieser Seite"
        },
        "handschriftliche_notizen": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Liste handschriftlicher Notizen"
        },
        "unsichere_lesungen": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["transkription", "handschriftliche_notizen", "unsichere_lesungen"],
    "additionalProperties": False # Zwingt das Modell, keine eigenen Felder zu erfinden
}

MERGE_SCHEMA = {}

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
    print(f"  Verarbeite Seite {page_number}: {image_path.name}")

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
    print("Führe alle Seiten zu einem Dokument zusammen...")

    pages_json = json.dumps(page_results, ensure_ascii=False, indent=2)
    content = MERGE_PROMPT.replace("{PAGES_JSON}", pages_json)
    return call_model(content, MERGE_SCHEMA)


def main():
    # search for images (.jpg, .jpeg, .png)
    image_files = sorted(list(Path(IMAGE_FOLDER).glob("*.jpg")) + list(Path(IMAGE_FOLDER).glob("*.jpeg")) + list(Path(IMAGE_FOLDER).glob("*.png")))

    if not image_files:
        print(f"Keine Bilder in {IMAGE_FOLDER} gefunden.")
        return

    print(f"{len(image_files)} Bild(er) gefunden. Starte Verarbeitung pro Seite...")

    # for every Image
    page_results = []
    for i, image_file in enumerate(image_files, start=1):
        try:
            page_result = process_page(image_file, i)
        except requests.exceptions.RequestException as e:
            print(f"  Fehler bei Seite {i} ({image_file.name}): {e}")
            page_result = {"_seite": i, "_dateiname": image_file.name, "_error": str(e)}
        page_results.append(page_result)

    # Zwischenstand sichern, falls der Merge-Schritt fehlschlägt
    intermediate_file = "zensurkarte_seiten_einzeln.json"
    with open(intermediate_file, "w", encoding="utf-8") as f:
        json.dump(page_results, f, ensure_ascii=False, indent=2)
    print(f"Einzelseiten-Ergebnisse gesichert in {intermediate_file}")

    try:
        merged = merge_pages(page_results)
    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Zusammenführen: {e}")
        print("Einzelseiten-Ergebnisse bleiben in", intermediate_file, "verfügbar.")
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Fertig. Ergebnis gespeichert in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()