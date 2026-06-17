import base64
import json
from pathlib import Path

import requests

# -----------------------------
# Konfiguration
# -----------------------------

IMAGE_FOLDER = r"./censorship_cards"
API_URL = "API_URL"  # Beispiel: Ollama OpenAI-kompatibler Endpunkt
API_KEY = "API_KEY"  # bei Ollama/LM Studio meist irrelevant, aber Header wird trotzdem mitgeschickt
MODEL = "google/gemma-4-12b"  # ANPASSEN: exakter Modellname, wie er in Ollama/LM Studio gelistet ist
OUTPUT_FILE = "zensurkarte.json"
TEMPERATURE = 0
TIMEOUT_SECONDS = 600


# --------------------------------------------------
# Prompt: pro Seite einzeln, nur reine Transkription
# --------------------------------------------------

PAGE_PROMPT = """
Das folgende Bild zeigt EINE Seite einer deutschen Filmzulassungskarte,
ausgestellt zwischen 1920 und 1945.

Aufgabe:

1. Führe eine vollständige OCR dieser einen Seite durch.
2. Berücksichtige Drucktext, Stempel, Randnotizen und Handschrift.
3. Bewahre die originale Schreibweise.
4. Ergänze keine Informationen, die nicht im Bild stehen.
5. Unsichere Lesungen markiere im Text mit [unsicher: ...].

Gib ausschließlich gültiges JSON in folgendem Format zurück, ohne
zusätzlichen Text, ohne Markdown-Codeblock:

{
  "transkription": "vollständiger erkannter Text dieser Seite",
  "handschriftliche_notizen": ["Liste handschriftlicher Notizen, falls vorhanden"],
  "unsichere_lesungen": ["Liste unsicherer Stellen, falls vorhanden"]
}
"""

# Prompt für den finalen Zusammenführungs-Schritt
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

Gib ausschließlich gültiges JSON in folgendem Format zurück, ohne
zusätzlichen Text, ohne Markdown-Codeblock:

{
  "pruefnummer": "",
  "Ursprungsfirma": "",
  "Filmtitel": "",
  "Mitwirkende": {"Person": "Rolle"},
  "Inhaltsbeschreibung": "",
  "Laenge": [],
  "Streichung_Aenderung": [],
  "handschriftliche_notizen": [],
  "unsichere_lesungen": [],
  "seiten": [
    {"seite": 1, "transkription": ""}
  ],
  "vollstaendige_transkription": ""
}

Falls weitere Felder im Dokument vorkommen, ergänze sie zusätzlich.

Seiten-Transkriptionen:
{PAGES_JSON}
"""


def encode_image_b64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_model(content) -> str:
    """Schickt eine Chat-Completion-Anfrage und gibt den rohen Antworttext zurück."""
    payload = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=TIMEOUT_SECONDS
    )
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]


def extract_json(raw_text: str) -> dict:
    """
    Versucht, JSON aus der Modellantwort zu extrahieren.
    Kleinere Modelle packen das JSON manchmal in ```json ... ``` oder
    fügen Text davor/danach hinzu. Wir versuchen mehrere Strategien.
    """
    text = raw_text.strip()

    # Strategie 1: direkt parsen
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategie 2: Markdown-Codeblock entfernen
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

    # Strategie 3: ersten { ... letzten } extrahieren
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Nichts hat funktioniert: rohen Text zurückgeben, damit nichts verloren geht
    return {"_parse_error": True, "_raw_response": raw_text}


def process_page(image_path: Path, page_number: int) -> dict:
    print(f"  Verarbeite Seite {page_number}: {image_path.name}")

    b64 = encode_image_b64(image_path)
    content = [
        {"type": "text", "text": PAGE_PROMPT},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        }
    ]

    raw_text = call_model(content)
    parsed = extract_json(raw_text)
    parsed["_seite"] = page_number
    parsed["_dateiname"] = image_path.name
    return parsed


def merge_pages(page_results: list[dict]) -> dict:
    print("Führe alle Seiten zu einem Dokument zusammen...")

    pages_json = json.dumps(page_results, ensure_ascii=False, indent=2)
    prompt = MERGE_PROMPT.replace("{PAGES_JSON}", pages_json)

    content = [{"type": "text", "text": prompt}]
    raw_text = call_model(content)
    return extract_json(raw_text)


def main():
    image_files = sorted(
        list(Path(IMAGE_FOLDER).glob("*.jpg")) +
        list(Path(IMAGE_FOLDER).glob("*.jpeg")) +
        list(Path(IMAGE_FOLDER).glob("*.png"))
    )

    if not image_files:
        print(f"Keine Bilder in {IMAGE_FOLDER} gefunden.")
        return

    print(f"{len(image_files)} Bild(er) gefunden. Starte Verarbeitung pro Seite...")

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