import json
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from pathlib import Path

json_schema = {
  "type": "object",
  "properties": {
    "prüf-nr": {
      "type": "integer"
    },
    "Gesetz": {
      "type": "string"
    },
    "Ursprungs-Firma": {
      "type": "string"
    },
    "Titel des Bildes": {
      "type": "string"
    },
    "Text unter Titel des Bildes (Kurzbeschreibung des Filmes)": {
      "type": "string"
    },
    "Spielleitung / Personen der Handlung:": {
      "type": "string"
    },
    "Photographie": {
      "type": "string"
    },
    "Archiketur": {
      "type": "string"
    },
    "Personenverzeichnis": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string"
          },
          "rolle": {
            "type": "string"
          }
        },
        "required": [
          "name",
          "rolle"
        ]
      }
    },
    "Text": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "akt": {
            "type": "integer"
          },
          "text": {
            "type": "string"
          }
        },
        "required": [
          "akt",
          "text"
        ]
      }
    },
    "Auschnitte": {
      "type": "string"
    },
    "Länge": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "akt": {
            "type": "integer"
          },
          "meter": {
            "type": "integer"
          },
          "nach kürzung": {
            "type": "integer"
          },
          "gesamtlänge": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "gesamtlänge": {
                  "type": "integer"
                },
                "nach kürzung": {
                  "type": "integer"
                }
              }
              # TODO: required für das innere Array hier einfügen, falls nötig
            }
          }
        }
      }
    },
    "Entscheidung": {
      "type": "string"
    },
    "ort und datum": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "ort": {
            "type": "string"
          },
          "datum": {
            "type": "string"
          }
        },
        "required": [
          "ort",
          "datum"
        ]
      }
    },
    "Filmprüfstelle": {
      "type": "string"
    }
  },
  "required": [
      "prüf-nr",
      "Gesetz",
      "Ursprungs-Firma",
      "Titel des Bildes",
      "Text unter Titel des Bildes (Kurzbeschreibung des Filmes)",
      "Spielleitung / Personen der Handlung:",
      "Photographie",
      "Archiketur",
      "Personenverzeichnis",
      "Text",
      "Auschnitte",
      "Länge",
      "Entscheidung",
      "Filmprüfstelle"
  ]
}

system_prompt = """Du bist ein hochpräzises Daten-Extraktions-System. 
Deine einzige Aufgabe ist es, Informationen aus einem unstrukturierten, teils fehlerhaften OCR-Text zu extrahieren und in ein strikt vorgegebenes JSON-Schema zu überführen.

REGELN FÜR DIE EXTRAKTION:
1. OCR-Korrektur: Der Eingabetext enthält typische OCR-Fehler (falsche Zeichen, verdrehte Reihenfolge, fehlende Leerzeichen). Analysiere den semantischen Kontext und korrigiere offensichtliche Zeichenfehler automatisch.
2. Strikte Faktentreue: Erfinde NIEMALS Informationen hinzu. Leite keine Daten ab, die nicht explizit oder als klarer OCR-Fehler im Text stehen.
3. Fehlende Werte: Wenn eine Information für ein Feld des JSON-Schemas im Text nicht auffindbar ist, setze den Wert ZWINGEND auf `null`. Verwende keine Platzhalter wie "unbekannt" oder "N/A".
4. Relevanz-Filter: Der OCR-Text enthält wahrscheinlich irrelevante Kopfzeilen, Fußzeilen oder Zusatzinformationen. Ignoriere alles, was nicht im Ziel-Schema abgefragt wird.
5. Formatierung: Halte dich exakt an die Datentypen des Schemas (Zahlen als Number, nicht als String; Datumsformate exakt wie gefordert).

Gib AUSSCHLIESSLICH das finale JSON-Objekt aus. Keine Erklärungen, kein Markdown-Codeblock, keine einleitenden Worte.
"""

ocr_dir = Path('data/02_ocr')
ocr_dir.mkdir(parents=True, exist_ok=True)

processed_dir = Path('../../data/03_processed')
processed_dir.mkdir(parents=True, exist_ok=True)

model_name = "Qwen/Qwen2.5-14B-Instruct-AWQ"

def main():

  llm = LLM(model=model_name)
  sampling_params = SamplingParams(
      temperature=0.0,
      seed=42,
      structured_outputs=StructuredOutputsParams(json_schema) # maybe als class 
      #guided_decoding=GuidedDecodingParams(json=schema_str)
  )

  for txt_file in ocr_dir.glob('*.txt'):
      print(f"Processing: {txt_file.name}")

      ocr_text = txt_file.read_text()

      messages=[
          {
              "role": "system", 
              "content": system_prompt
            },
            {
                "role": "user",
                "content" : ocr_text 
            }
      ]

      try:  
          output = llm.chat(messages=messages, sampling_params=sampling_params, use_tqdm=True)

          output_text = output.outputs[0].text
          print(output_text)

          out_file = processed_dir / f"{txt_file.stem}.json"

          with open(out_file, mode="w") as f:
              f.write(output_text)
            
          print(f"-> saved as: {out_file.name}\n")
      
      except Exception as e:
          print(f"Error on File {txt_file.name}: {e}\n")

  print("LLM parser done")


if __name__ == "__main__":
    main()