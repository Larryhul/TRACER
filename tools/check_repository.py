"""Check distributed asset hashes and local documentation links; no ML dependencies."""
import csv
import hashlib
import re
from pathlib import Path
from urllib.parse import unquote

root = Path(__file__).resolve().parents[1]
errors = []
count = 0
for row in csv.DictReader((root / 'provenance/assets.csv').open(encoding='utf-8')):
    path = root / row['path']
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
        errors.append('Asset hash mismatch: ' + row['path'])
    count += 1
links = 0
for doc in root.rglob('*.md'):
    if any(part in {'.git', '.venv', 'build'} for part in doc.parts):
        continue
    text = re.sub(r'```[\s\S]*?```', '', doc.read_text(encoding='utf-8'))
    for target in re.findall(r'\]\(([^\s)]+)(?:\s+"[^"]*")?\)', text):
        if re.match(r'[a-zA-Z][\w+.-]*:', target) or target.startswith('#'):
            continue
        local = unquote(target.split('#')[0])
        if local and not (doc.parent / local).exists():
            errors.append(f'Broken link: {doc.relative_to(root)} -> {target}')
        links += 1
if errors:
    raise SystemExit('\n'.join(errors))
print(f'OK: {count} asset hashes and {links} local documentation links')
