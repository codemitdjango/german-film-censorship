import json
from collections import defaultdict
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from pathlib import Path

json_schema = {
  "type": "object",
  "properties": {
    "prüf-nr": {"type": "integer"},
    "Gesetz": {"type": "string"},
    "Ursprungs-Firma": {"type": "string"},
    "Titel des Bildes": {"type": "string"},
    "Text unter Titel des Bildes (Kurzbeschreibung des Filmes)": {"type": "string"},
    "Spielleitung / Personen der Handlung:": {"type": "string"},
    "Photographie": {"type": "string"},
    "Archiketur": {"type": "string"},
    "Personenverzeichnis": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "rolle": {"type": "string"}
        },
        "required": ["name", "rolle"]
      }
    },
    "Text": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "akt": {"type": "integer"},
          "text": {"type": "string"}
        },
        "required": ["akt", "text"]
      }
    },
    "Auschnitte": {"type": "string"},
    "Länge": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "akt": {"type": "integer"},
          "meter": {"type": "integer"},
          "nach kürzung": {"type": "integer"},
          "gesamtlänge": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "gesamtlänge": {"type": "integer"},
                "nach kürzung": {"type": "integer"}
              }
            }
          }
        }
      }
    },
    "Entscheidung": {"type": "string"},
    "ort und datum": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "ort": {"type": "string"},
          "datum": {"type": "string"}
        },
        "required": ["ort", "datum"]
      }
    },
    "Filmprüfstelle": {"type": "string"}
  },
  "required": [
      "prüf-nr", "Gesetz", "Ursprungs-Firma", "Titel des Bildes",
      "Text unter Titel des Bildes (Kurzbeschreibung des Filmes)",
      "Spielleitung / Personen der Handlung:", "Photographie", "Archiketur",
      "Personenverzeichnis", "Text", "Auschnitte", "Länge", "Entscheidung",
      "Filmprüfstelle"
  ]
}

system_prompt = """Du bist ein hochpräzises visuelles Daten-Extraktions-System. 
Deine einzige Aufgabe ist es, Informationen direkt aus den bereitgestellten Dokumenten-Bildern zu extrahieren und in ein strikt vorgegebenes JSON-Schema zu überführen. Das Dokument besteht aus mehreren Seiten, die dir in korrekter Reihenfolge vorliegen.

REGELN FÜR DIE EXTRAKTION:
1. Visuelle Analyse: Lies den Text exakt so aus, wie er auf den Bildern steht. Führe die Informationen der Einzelseiten logisch zusammen.
2. Strikte Faktentreue: Erfinde NIEMALS Informationen hinzu. Leite keine Daten ab, die nicht explizit auf den Bildern erkennbar sind.
3. Fehlende Werte: Wenn eine Information für ein Feld des JSON-Schemas nicht auffindbar ist, setze den Wert ZWINGEND auf `null`.
4. Relevanz-Filter: Ignoriere Randnotizen, Stempel oder Kopf-/Fußzeilen, sofern diese nicht explizit im Ziel-Schema abgefragt werden.
5. Formatierung: Halte dich exakt an die Datentypen des Schemas.

Gib AUSSCHLIESSLICH das finale JSON-Objekt aus. Keine Erklärungen, kein Markdown-Codeblock, keine einleitenden Worte.
"""

image_dir = Path('data/01_raw')
processed_dir = Path('data/03_processed')
processed_dir.mkdir(parents=True, exist_ok=True)

model_name = "Qwen/Qwen2-VL-7B-Instruct"

def main():

    image_groups = defaultdict(list)
    image_extensions = ('*.png', '*.jpg', '*.jpeg')
    
    for ext in image_extensions:
        for img_path in image_dir.rglob(ext):
            group_name = img_path.parent.name
            image_groups[group_name].append(img_path)
            
            # sort
            for group in image_groups:
                image_groups[group].sort()
                
    # Bilder innerhalb einer Gruppe chronologisch sortieren (0001, 0002, ...)
    for prefix in image_groups:
        image_groups[prefix].sort()

    if not image_groups:
        print("Keine Bilder gefunden.")
        return

    # Ermittle die maximale Anzahl an Bildern pro Dokument für vLLM
    max_images_per_prompt = max(len(imgs) for imgs in image_groups.values())
    print(f"Maximale Bilder pro Dokument: {max_images_per_prompt}")

    # 2. Modell initialisieren
    llm = LLM(
        model=model_name,
        limit_mm_per_prompt={"image": max_images_per_prompt},
        max_model_len=131072,
        allowed_local_media_path=str(image_dir.absolute()),
        tensor_parallel_size=4, # bis zu 8 möglich auf paula
        gpu_memory_utilization=0.95,
        dtype="bfloat16"
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        seed=42,
        max_tokens=4096,
        structured_outputs=StructuredOutputsParams(json_schema)
    )

    # 3. Inferenz durchführen
    for doc_id, img_paths in image_groups.items():
        print(f"\nProcessing Document: {doc_id} ({len(img_paths)} Bilder)")

        # Content-Liste für den User-Prompt aufbauen
        user_content = []
        for img in img_paths:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"file://{img.absolute()}"}
            })
        
        # Text-Anweisung am Ende anfügen
        user_content.append({
            "type": "text", 
            "text": "Extrahiere alle relevanten Daten aus diesen Seiten entsprechend des JSON-Schemas."
        })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        try:  
            output = llm.chat(messages=messages, sampling_params=sampling_params, use_tqdm=True)

            output_text = output[0].outputs[0].text
            print(output_text)

            out_file = processed_dir / f"{doc_id}.json"

            with open(out_file, mode="w", encoding="utf-8") as f:
                f.write(output_text)
            
            print(f"-> saved as: {out_file.name}\n")
        
        except Exception as e:
            print(f"Error on Document {doc_id}: {e}\n")

    print("VLM parser done")

if __name__ == "__main__":
    main()