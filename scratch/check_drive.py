import requests
import re

file_id = '17WozXiXfq44Z6kkWsPPDHRzqIH2daUaQ'
session = requests.Session()
url = f'https://drive.google.com/uc?export=download&id={file_id}'
r = session.get(url)
print("=== FULL HTML ===")
print(r.text)
