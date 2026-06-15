import json
from collections import defaultdict
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from pathlib import Path

# load JSON 
SCRIPT_DIR = Path(__file__).parent
json_schema = json.loads(Path(SCRIPT_DIR / "schema.json").read_text(encoding="utf-8"))

system_prompt = """
You are an expert visual data extraction system.
Your specific task is to extract information from images of historical German censorship/examination records and format them strictly according to the provided JSON schema. The document consists of multiple pages provided in chronological order.

EXTRACTION RULES:
1. Language & Fidelity: Extract the text exactly as written in the original German. Do not translate. Never invent or infer information that is not explicitly visible.
2. Handling Line Breaks: If a word is hyphenated across two lines, merge it into a single word. Remove arbitrary line breaks within continuous sentences.
3. Missing Information: If the specific information for a field cannot be found on the pages, you MUST set the value to `null`. Do not use placeholder strings like "N/A", "unbekannt", or "-".
4. Noise Reduction: Ignore marginalia, stamps, handwritten scribbles, or headers/footers unless they directly answer a requested field in the schema.
5. Numerical Values: For lengths ("Länge") or document numbers ("prüf-nr"), extract ONLY the numeric integer value, stripping out units like "Meter" or "Nr." unless the schema explicitly requires a string.

FIELD MAPPING & CONTEXT:
- "Ursprungs-Firma": The production company or studio.
- "Text unter Titel des Bildes": Treat this as the short description or synopsis of the film.
- "Spielleitung": This is the historical term for the director.
- "Architektur": This refers to set design or art direction.
- "Auschnitte" / "Entscheidung": Transcribe the censor's decisions or required cuts accurately.
"""

image_dir = Path('data/01_raw')
processed_dir = Path('data/03_processed')
processed_dir.mkdir(parents=True, exist_ok=True)

model_name = "Qwen/Qwen2-VL-7B-Instruct"

def main():

    image_groups = defaultdict(list)
    image_extensions = ('*.png', '*.jpg', '*.jpeg')
    
    # collect images
    for ext in image_extensions:
        for img_path in image_dir.rglob(ext):
            group_name = img_path.parent.name
            image_groups[group_name].append(img_path)
                
    # sort
    for prefix in image_groups:
        image_groups[prefix].sort()

    if not image_groups:
        print("Keine Bilder gefunden.")
        return

    # Ermittle die maximale Anzahl an Bildern pro Dokument für vLLM
    max_images_per_prompt = max(len(imgs) for imgs in image_groups.values())
    print(f"Maximale Bilder pro Dokument: {max_images_per_prompt}")

    # 2. Modell initialisieren
    #llm2 = LLM(
    #    model=model_name,
    #    limit_mm_per_prompt={"image": max_images_per_prompt},
    #    #max_model_len=32768,
    #    allowed_local_media_path=str(image_dir.absolute()),
    #    tensor_parallel_size=4, # bis zu 8 möglich auf paula
    #    gpu_memory_utilization=0.95,
    #    dtype="bfloat16",
    #    disable_flashinfer_sampling=True
    #)

    # Wenn du vLLM via Python-Skript initialisierst:
    llm = LLM(
        model="Qwen/Qwen2-VL-7B-Instruct",
        tensor_parallel_size=4,
        disable_custom_all_reduce=True,
        enforce_eager=True # Hilft oft zusätzlich auf Clustern
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        seed=42,
        #max_tokens=4096,
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