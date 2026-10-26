import urllib.request
from pathlib import Path

raw_dir = Path("scratch/meme_data/raw/hasoc_meme_2025")
raw_dir.mkdir(parents=True, exist_ok=True)

files = {
    "Hindi Train HASOC 2025 - Updated.csv": "https://raw.githubusercontent.com/fatimahansari/hindi-HASOC-2025/main/Hindi%20Train%20HASOC%202025%20-%20Updated.csv",
    "Hindi Test HASOC 2025 - Updated.csv": "https://raw.githubusercontent.com/fatimahansari/hindi-HASOC-2025/main/Hindi%20Test%20HASOC%202025%20-%20Updated.csv"
}

for name, url in files.items():
    dest = raw_dir / name
    if not dest.exists():
        print(f"Downloading {name}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            dest.write_bytes(resp.read())
        print(f"  Saved {name} ({dest.stat().st_size:,} bytes)")
    else:
        print(f"Found cached {name}")
