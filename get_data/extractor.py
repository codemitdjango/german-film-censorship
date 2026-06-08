import zipfile
from pathlib import Path

def extract_archives():
    # define base directory
    base_dir = Path(__file__).parent.parent / "data" / "01_raw"

    # find all zip files recursively
    for zip_path in base_dir.rglob("*.zip"):
        
        # define extraction target (same name, no .zip extension)
        extract_dir = zip_path.parent
        
        print(f"Extracting: {zip_path.name}")
        
        # extract archive
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        # delete original zip file
        # zip_path.unlink()

if __name__ == "__main__":
    extract_archives()