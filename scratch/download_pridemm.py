import os
import sys
import time
import requests
import re
import zipfile

def download_pridemm():
    dest_dir = 'scratch/meme_data/raw/pridemm'
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, 'PrideMM.zip')
    
    file_id = '17WozXiXfq44Z6kkWsPPDHRzqIH2daUaQ'
    session = requests.Session()
    
    # Check if zip already exists and is complete
    expected_size = 3431157114
    if os.path.exists(zip_path) and os.path.getsize(zip_path) == expected_size:
        print(f"PrideMM.zip already fully downloaded ({os.path.getsize(zip_path)} bytes).")
    else:
        print("Obtaining download confirmation from Google Drive...")
        url = f'https://drive.google.com/uc?export=download&id={file_id}'
        r = session.get(url)
        uuid_match = re.search(r'name="uuid"\s+value="([^"]+)"', r.text)
        if not uuid_match:
            print("Failed to find download uuid in Drive response.")
            sys.exit(1)
            
        uuid_val = uuid_match.group(1)
        dl_url = f'https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t&uuid={uuid_val}'
        
        print(f"Starting stream download to {zip_path}...")
        start_time = time.time()
        with session.get(dl_url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            print(f"Total size to download: {total_size / (1024*1024):.2f} MB")
            
            downloaded = 0
            last_log = time.time()
            chunk_size = 1024 * 1024 * 4 # 4MB chunks
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if time.time() - last_log >= 10: # Log every 10 seconds
                            elapsed = time.time() - start_time
                            speed_mb = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                            pct = (downloaded / total_size) * 100 if total_size > 0 else 0
                            print(f"Downloaded {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({pct:.1f}%) at {speed_mb:.2f} MB/s")
                            last_log = time.time()
                            
        print(f"Download complete: {os.path.getsize(zip_path)} bytes in {time.time() - start_time:.1f}s.")
        
    # Extract the zip file
    print("Extracting PrideMM.zip...")
    extract_start = time.time()
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_dir)
    print(f"Extraction complete in {time.time() - extract_start:.1f}s.")
    
    # List top level contents of dest_dir
    print("Extracted contents in", dest_dir, ":")
    for item in os.listdir(dest_dir):
        p = os.path.join(dest_dir, item)
        if os.path.isdir(p):
            print(f"  [DIR]  {item} ({len(os.listdir(p))} items)")
        else:
            print(f"  [FILE] {item} ({os.path.getsize(p)} bytes)")

if __name__ == '__main__':
    download_pridemm()
