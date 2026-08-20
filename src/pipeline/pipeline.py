import base64
import json
import logging
import os
import re
from pathlib import Path
from dotenv import load_dotenv
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()

# path configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

PROMPT_DIR = SCRIPT_DIR / "prompts"
FEW_SHOTS_DIR = SCRIPT_DIR / "few_shots"

DATA_BASE_DIR = Path(os.getenv("DATA_BASE_DIR"))

IMAGE_DIR = DATA_BASE_DIR / "R_9346-I_Zulassungskarten"
OCR_OUTPUT_DIR = DATA_BASE_DIR / "02_ocr"
PROCESSED_OUTPUT_DIR = DATA_BASE_DIR/ "03_processed"
LOG_DIR = SCRIPT_DIR / "logs"

# setup logging
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# api parameters
API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY", "fallback")
MODEL = "google/gemma-4-12b"
TEMPERATURE= 1.0 # standardized sampling configuration # TEMPERATURE = 0.1 # bei 0 hängt das Modell im LOOP
MAX_TOKENS = 262144 
# FREQUENCY_PENALTY = 0.1 #1.2 # Bestraft das Modell, wenn es dieselben Wörter oft wiederholt
TIMEOUT_SECONDS = 600

# encode image to base64
def encode_image_b64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()
    
# load prompts, schemas and  few-shot
PAGE_PROMPT = (PROMPT_DIR / "page_prompt.txt").read_text(encoding="utf-8")
MERGE_PROMPT = (PROMPT_DIR / "merge_prompt.txt").read_text(encoding="utf-8")
PAGE_SCHEMA = json.loads((PROMPT_DIR / "page_schema.json").read_text(encoding="utf-8"))
MERGE_SCHEMA = json.loads((PROMPT_DIR / "merge_schema.json").read_text(encoding="utf-8"))

FS_OCR_INPUT_B64 = encode_image_b64(FEW_SHOTS_DIR / "few_shot_ocr_input.jpg")
FS_OCR_OUTPUT_STR = json.dumps(
    json.loads((FEW_SHOTS_DIR / "few_shot_ocr_output.json").read_text(encoding="utf-8")), 
    ensure_ascii=False
)

FS2_OCR_INPUT_B64 = encode_image_b64(FEW_SHOTS_DIR / "few_shot_ocr_input2.jpg")
FS2_OCR_OUTPUT_STR = json.dumps(
    json.loads((FEW_SHOTS_DIR / "few_shot_ocr_output2.json").read_text(encoding="utf-8")), 
    ensure_ascii=False
)

FS_MERGE_INPUT_STR = json.dumps(
    json.loads((FEW_SHOTS_DIR / "few_shot_merge_input.json").read_text(encoding="utf-8")), 
    ensure_ascii=False, separators=(',', ':')
)
FS_MERGE_OUTPUT_STR = json.dumps(
    json.loads((FEW_SHOTS_DIR / "few_shot_merge_output.json").read_text(encoding="utf-8")), 
    ensure_ascii=False, separators=(',', ':')
)

# coordinates the entire pipeline by iterating through all document directories
def main():
    # ensure output directories exist
    for directory in [OCR_OUTPUT_DIR, PROCESSED_OUTPUT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    if not IMAGE_DIR.exists() or not IMAGE_DIR.is_dir():
        logging.warning("hauptverzeichnis %s exisiert nicht", IMAGE_DIR)
        return
    
    doc_directories = sorted([d for d in IMAGE_DIR.iterdir() if d.is_dir()])

    if not doc_directories:
        logging.warning("keine dokumentenordner in %s gefunden", IMAGE_DIR)
        return
    
    logging.info("%d dokumentenordner gefunden", len(doc_directories))

    # Pipeline for every Folder
    for doc_dir in doc_directories:
        process_document_directory(doc_dir)


# processes a single document folder: extracts data from all pages, merges them, and exports the results
def process_document_directory(doc_dir: Path):
    final_file = PROCESSED_OUTPUT_DIR / f"{doc_dir.name}_processed.json"
    doc_ocr_dir = OCR_OUTPUT_DIR / doc_dir.name
    doc_ocr_dir.mkdir(parents=True, exist_ok=True)

    if final_file.exists():
        logging.info("%s bereits verarbeitet, überspringe", doc_dir.name)
        return

    logging.info("starte verarbeitung für ordner: %s", doc_dir.name)
    image_files = get_sorted_images(doc_dir)

    if not image_files:
        logging.warning("keine bilder in %s gefunden", doc_dir.name)
        return
    
    page_results = []

    for i, image_file in enumerate(image_files, start=1):
        page_cache_file = doc_ocr_dir / f"page_{i}_{image_file.stem}.json"

        # load from cache if exists
        if page_cache_file.exists():
            try:
                with open(page_cache_file, "r", encoding="utf-8") as f:
                    page_result = json.load(f)
                page_results.append(page_result)
                continue
            except json.JSONDecodeError:
                logging.warning("defekter cache für %s, verarbeite neu", page_cache_file.name)

        try:
            page_result = process_page(image_file, i)
        except Exception as e:
            logging.error("dauerhafter fehler bei seite %d (%s): %s", i, image_file.name, e)
            page_result = {"_page": i, "_filename": image_file.name, "_error": str(e)}

        with open(page_cache_file, "w", encoding="utf-8") as f:
            json.dump(page_result, f, ensure_ascii=False, indent=2)

        page_results.append(page_result)

    intermediate_file = OCR_OUTPUT_DIR / f"{doc_dir.name}_ocr.json"
    with open(intermediate_file, "w", encoding="utf-8") as f:
        json.dump(page_results, f, ensure_ascii=False, indent=2)

    # filter failed pages before merge
    valid_pages = [page for page in page_results if "_error" not in page]
    if not valid_pages:
        logging.error("keine validen seiten für %s. abbruch des merges", doc_dir.name)
        return
    
    try:
        merged = merge_pages(valid_pages, doc_dir.name)
        with open(final_file, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        logging.info("erfolgreich abgeschlossen: %s", final_file)
    except Exception as e:
        logging.error("fehler beim zusammenführen von %s: %s", doc_dir.name, e)


# process page to dict and inject few-shots
def process_page(image_path: Path, page_number: int) -> dict:
    logging.info("verarbeite seite %d: %s", page_number, image_path.name)
    b64 = encode_image_b64(image_path)

    # prepare main content
    content = [
        {"type": "text", "text": PAGE_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
    ]

    # build few-shot messages
    few_shots = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PAGE_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{FS_OCR_INPUT_B64}"}}
            ]
        },
        {
            "role": "assistant",
            "content": FS_OCR_OUTPUT_STR
        },
        # second few shot
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PAGE_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{FS2_OCR_INPUT_B64}"}}
            ]
        },
        {
            "role": "assistant",
            "content": FS2_OCR_OUTPUT_STR
        }
    ]
    
    # call model and append metadata
    parsed = call_model(content, PAGE_SCHEMA, few_shots)
    # parsed = call_model(content, PAGE_SCHEMA)
    parsed["_page"] = page_number
    parsed["_filename"] = image_path.name
    return parsed


# consolidates pages to a single coument structure
def merge_pages(page_results: list[dict], doc_name: str) -> dict:
    logging.info("zusammenführung gestartet: %s", doc_name)

    all_intertitles = []
    cleaned_pages = []

    for page in page_results:
        page_copy = dict(page)
        intertitles = page_copy.pop("intertitles", [])
        if isinstance(intertitles, list):
            all_intertitles.extend(intertitles)
        cleaned_pages.append(page_copy)

    # dynamically adjust merge schema to exclude intertitle form output generation
    trimmed_merge_schema = json.loads(json.dumps(MERGE_SCHEMA))
    schema_props = trimmed_merge_schema.get("schema", {}).get("properties", {})
    if "intertitles" in schema_props:
        del schema_props["intertitles"]
    schema_reqs = trimmed_merge_schema.get("schema", {}).get("required", [])
    if "intertitles" in schema_reqs:
        schema_reqs.remove("intertitles")

    # build few-shot messages
    few_shots = [
        {
            "role": "user", 
            "content": FS_MERGE_INPUT_STR
        },
        {
            "role": "assistant", 
            "content": FS_MERGE_OUTPUT_STR
        }
    ]

    # prepare content and call model
    pages_json = json.dumps(cleaned_pages, ensure_ascii=False, separators=(',', ':'))
    content = MERGE_PROMPT.replace("{PAGES_JSON}", pages_json)

    merged_result = call_model(content, trimmed_merge_schema, few_shots)
    # merged_result = call_model(content, trimmed_merge_schema)

    # reattach combined intertitles deterministacially
    merged_result["intertitles"] = all_intertitles

    return merged_result


# retrieves all image files from a specific directory in natural alphanumerical order
def get_sorted_images(dir_path: Path) -> list[Path]:
    # search every image in a folder and sort
    extensions = ("*.jpg", "*.jpeg", "*.png")
    images = []
    for ext in extensions:
        images.extend(dir_path.glob(ext))
    
    # sort
    def natural_sort_key(path: Path):
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", path.name)
        ]

    return sorted(images, key=natural_sort_key)
# retry policy: up to 5 retries with exponential backoff for network/http errors
@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    retry=retry_if_exception_type((requests.exceptions.RequestException, json.JSONDecodeError))
)
# executes the api call to the LLM
# suuports optional few-shot examples via messages array
def call_model(content, schema, few_shots=None) -> dict:
    messages = []

    # insert few-shot conversation history before actual prompt
    if few_shots:
        messages.extend(few_shots)

    messages.append({ "role": "user", "content": content})

    payload = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS, 
        # "frequency_penalty": FREQUENCY_PENALTY,  
        "top_p": 0.95,
        "top_k": 64,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": schema
        }
    }

    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=TIMEOUT_SECONDS
    )

    if not response.ok:
        raise requests.exceptions.RequestException(f"API Error {response.status_code}: {response.text}")
    response.raise_for_status()

    # parse response and check termination reason
    raw_text = response.json()["choices"][0]["message"]["content"]
    return json.loads(raw_text)

# script execution
if __name__ == "__main__":
    main()