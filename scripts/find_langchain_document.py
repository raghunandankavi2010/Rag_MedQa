import langchain
import os, glob
p = langchain.__path__[0]
print('langchain path:', p)
matches = []
for f in glob.glob(os.path.join(p, '**', '*.py'), recursive=True):
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            txt = fh.read()
    except Exception:
        continue
    if 'class Document' in txt or "class Document(" in txt or "def Document(" in txt or 'class BaseDocument' in txt:
        matches.append(f)
print('Matches (first 50):')
for m in matches[:50]:
    print(m)
print('Total matches:', len(matches))
