import requests
import re

session = requests.Session()
file_id = '17WozXiXfq44Z6kkWsPPDHRzqIH2daUaQ'
url = f'https://drive.google.com/uc?export=download&id={file_id}'
r = session.get(url)
uuid_match = re.search(r'name="uuid"\s+value="([^"]+)"', r.text)

if uuid_match:
    uuid_val = uuid_match.group(1)
    dl_url = f'https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t&uuid={uuid_val}'
    r_dl = session.get(dl_url, stream=True)
    print('Download response status:', r_dl.status_code)
    print('Content-Type:', r_dl.headers.get('Content-Type'))
    print('Content-Length:', r_dl.headers.get('Content-Length'))
    chunk = next(r_dl.iter_content(chunk_size=1024))
    print('First 10 bytes (magic number):', chunk[:10])
    if chunk[:2] == b'PK':
        print('SUCCESS: Valid ZIP archive header detected!')
else:
    print('No uuid found')
