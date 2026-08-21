import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from google import genai
from google.genai import types
from helpers import get_sorted_images, load_json, save_json

load_dotenv()

# path configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

PROMPT_DIR = SCRIPT_DIR / "prompts"
FEW_SHOTS_DIR = SCRIPT_DIR / "few_shots"

DATA_BASE_DIR = Path(os.getenv("DATA_BASE_DIR"))

IMAGE_DIR = DATA_BASE_DIR / "R_9346-I_Zulassungskarten"
OCR_OUTPUT_DIR = DATA_BASE_DIR / "02_ocr"
PROCESSED_OUTPUT_DIR = DATA_BASE_DIR / "03_processed"
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

# api parameters and client initialization
API_KEY = os.getenv("GEMINI_API_KEY")
# MODEL = "gemma-4-26b-a4b-it"
MODEL = "gemma-4-31b-it"
TEMPERATURE = 1.0

client = genai.Client(api_key=API_KEY)

# load prompts, schemas and few-shot data
PAGE_PROMPT = (PROMPT_DIR / "page_prompt.txt").read_text(encoding="utf-8")
MERGE_PROMPT = (PROMPT_DIR / "merge_prompt.txt").read_text(encoding="utf-8")
PAGE_SCHEMA = load_json(PROMPT_DIR / "page_schema.json")
MERGE_SCHEMA = load_json(PROMPT_DIR / "merge_schema.json")

FS_OCR_INPUT_BYTES = (FEW_SHOTS_DIR / "few_shot_ocr_input.jpg").read_bytes()
FS_OCR_OUTPUT_STR = json.dumps(load_json(FEW_SHOTS_DIR / "few_shot_ocr_output.json"), ensure_ascii=False)

FS2_OCR_INPUT_BYTES = (FEW_SHOTS_DIR / "few_shot_ocr_input2.jpg").read_bytes()
FS2_OCR_OUTPUT_STR = json.dumps(load_json(FEW_SHOTS_DIR / "few_shot_ocr_output2.json"), ensure_ascii=False)

FS_MERGE_INPUT_STR = json.dumps(load_json(FEW_SHOTS_DIR / "few_shot_merge_input.json"), ensure_ascii=False, separators=(',', ':'))
FS_MERGE_OUTPUT_STR = json.dumps(load_json(FEW_SHOTS_DIR / "few_shot_merge_output.json"), ensure_ascii=False, separators=(',', ':'))


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

    # pipeline for every folder
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
                page_result = load_json(page_cache_file)
                page_results.append(page_result)
                continue
            except json.JSONDecodeError:
                logging.warning("defekter cache für %s, verarbeite neu", page_cache_file.name)

        try:
            page_result = process_page(image_file, i)
        except Exception as e:
            logging.error("dauerhafter fehler bei seite %d (%s): %s", i, image_file.name, e)
            page_result = {"_page": i, "_filename": image_file.name, "_error": str(e)}

        save_json(page_cache_file, page_result)
        page_results.append(page_result)

    intermediate_file = OCR_OUTPUT_DIR / f"{doc_dir.name}_ocr.json"
    save_json(intermediate_file, page_results)

    # filter failed pages before merge
    valid_pages = [page for page in page_results if "_error" not in page]
    if not valid_pages:
        logging.error("keine validen seiten für %s. abbruch des merges", doc_dir.name)
        return
    
    try:
        merged = merge_pages(valid_pages, doc_dir.name)
        save_json(final_file, merged)
        logging.info("erfolgreich abgeschlossen: %s", final_file)
    except Exception as e:
        logging.error("fehler beim zusammenführen von %s: %s", doc_dir.name, e)


# process page to dict and inject few-shots
def process_page(image_path: Path, page_number: int) -> dict:
    logging.info("verarbeite seite %d: %s", page_number, image_path.name)
    image_bytes = image_path.read_bytes()

    # prepare main parts for current request
    content_parts = [
        types.Part.from_text(text=PAGE_PROMPT),
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    ]

    # build few-shot conversation history
    few_shots = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=PAGE_PROMPT),
                types.Part.from_bytes(data=FS_OCR_INPUT_BYTES, mime_type="image/jpeg")
            ]
        ),
        types.Content(
            role="model",
            parts=[types.Part.from_text(text=FS_OCR_OUTPUT_STR)]
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=PAGE_PROMPT),
                types.Part.from_bytes(data=FS2_OCR_INPUT_BYTES, mime_type="image/jpeg")
            ]
        ),
        types.Content(
            role="model",
            parts=[types.Part.from_text(text=FS2_OCR_OUTPUT_STR)]
        )
    ]

    # call model and append metadata
    parsed = call_model(content_parts, PAGE_SCHEMA, few_shots)
    parsed["_page"] = page_number
    parsed["_filename"] = image_path.name
    return parsed


# consolidates pages to a single document structure
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

    # dynamically adjust merge schema to exclude intertitle from output generation
    trimmed_merge_schema = load_json(PROMPT_DIR / "merge_schema.json")
    schema_props = trimmed_merge_schema.get("schema", {}).get("properties", {})
    if "intertitles" in schema_props:
        del schema_props["intertitles"]
    schema_reqs = trimmed_merge_schema.get("schema", {}).get("required", [])
    if "intertitles" in schema_reqs:
        schema_reqs.remove("intertitles")

    # build few-shot content
    few_shots = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=FS_MERGE_INPUT_STR)]
        ),
        types.Content(
            role="model",
            parts=[types.Part.from_text(text=FS_MERGE_OUTPUT_STR)]
        )
    ]

    # prepare prompt content
    pages_json = json.dumps(cleaned_pages, ensure_ascii=False, separators=(',', ':'))
    prompt_text = MERGE_PROMPT.replace("{PAGES_JSON}", pages_json)
    content_parts = [types.Part.from_text(text=prompt_text)]

    merged_result = call_model(content_parts, trimmed_merge_schema, few_shots)

    # reattach combined intertitles deterministically
    merged_result["intertitles"] = all_intertitles

    return merged_result


# executes the api call to the LLM via official google-genai SDK
@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    retry=retry_if_exception_type(Exception)
)
def call_model(content_parts: list[types.Part], schema: dict, few_shots: list[types.Content] = None) -> dict:
    contents = []

    # insert few-shots if provided
    if few_shots:
        contents.extend(few_shots)

    # append actual user query
    contents.append(
        types.Content(
            role="user",
            parts=content_parts
        )
    )

    # build content generation configuration
    generate_content_config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        top_p=0.95,
        top_k=64,
        response_mime_type="application/json",
        response_schema=schema,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
        thinking_config=types.ThinkingConfig(
            thinking_budget=0
        ),
        # thinking_config=types.ThinkingConfig(
        #     # thinking_level="HIGH",
        #     # thinking_level="LOW",
        #     thinking_level="OFF",
        # ),
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=generate_content_config
    )

    if not response.text:
        raise ValueError("empty response received from api")

    return json.loads(response.text)


# script execution
if __name__ == "__main__":
    main()