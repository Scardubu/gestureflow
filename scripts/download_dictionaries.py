“””
Download dictionary files for different languages.
“””
import requests
from pathlib import Path
import sys

sys.path.append(str(Path(**file**).parent.parent))

from src.config import DICTIONARIES_DIR, SUPPORTED_LANGUAGES

def download_dictionary(language: str) -> bool:
“””
Download dictionary file for a language.

```
Args:
    language: Language code
    
Returns:
    True if successful
"""
if language not in SUPPORTED_LANGUAGES:
    print(f"Unsupported language: {language}")
    return False

lang_config = SUPPORTED_LANGUAGES[language]
url = lang_config["dictionary_url"]
output_file = DICTIONARIES_DIR / lang_config["dictionary_file"]

print(f"Downloading {lang_config['name']} dictionary...")
print(f"  URL: {url}")
print(f"  Output: {output_file}")

try:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    # Save to file
    with open(output_file, 'wb') as f:
        f.write(response.content)
    
    # Count words
    with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
        word_count = sum(1 for line in f if line.strip())
    
    print(f"  ✓ Downloaded {word_count:,} words")
    return True
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    return False
```

def main():
“”“Download all dictionaries.”””
print(”=” * 80)
print(“Downloading Dictionary Files”)
print(”=” * 80)
print()

```
DICTIONARIES_DIR.mkdir(parents=True, exist_ok=True)

success_count = 0
for lang_code in SUPPORTED_LANGUAGES:
    if download_dictionary(lang_code):
        success_count += 1
    print()

print("=" * 80)
print(f"Downloaded {success_count}/{len(SUPPORTED_LANGUAGES)} dictionaries")
print("=" * 80)
```

if **name** == “**main**”:
main()
