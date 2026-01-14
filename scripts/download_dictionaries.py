#!/usr/bin/env python3
"""
Download dictionary files for different languages.
"""
import urllib.request
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DICTIONARY_DIR


# Dictionary sources (using existing dictionaries as example)
DICTIONARY_URLS = {
    'en_US': 'https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt',
    'es_ES': 'https://github.com/olea/lemarios/raw/master/lemario-general-del-espanol.txt',
    'fr_FR': 'https://raw.githubusercontent.com/chrplr/openlexicon/master/datasets-info/Liste-de-mots-francais-Gutenberg/liste.de.mots.francais.frgut.txt'
}


def download_dictionary(language_code: str, url: str, output_path: Path):
    """
    Download a dictionary file.
    
    Args:
        language_code: Language code (e.g., 'en_US')
        url: URL to download from
        output_path: Path to save file
    """
    try:
        print(f"Downloading {language_code} dictionary...")
        print(f"URL: {url}")
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download file
        urllib.request.urlretrieve(url, output_path)
        
        # Count words
        with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
            word_count = sum(1 for line in f if line.strip())
        
        print(f"Downloaded {word_count} words to {output_path}")
        print(f"✓ {language_code} dictionary downloaded successfully\n")
        
    except Exception as e:
        print(f"✗ Failed to download {language_code}: {e}\n")


def main():
    """Download all dictionaries."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download dictionary files")
    parser.add_argument(
        '--language', '-l',
        type=str,
        choices=['en_US', 'es_ES', 'fr_FR', 'all'],
        default='all',
        help='Language to download (default: all)'
    )
    
    args = parser.parse_args()
    
    print("GestureFlow Dictionary Downloader")
    print("=" * 50)
    print()
    
    if args.language == 'all':
        for lang_code, url in DICTIONARY_URLS.items():
            output_path = DICTIONARY_DIR / f"{lang_code}.txt"
            download_dictionary(lang_code, url, output_path)
    else:
        lang_code = args.language
        if lang_code in DICTIONARY_URLS:
            url = DICTIONARY_URLS[lang_code]
            output_path = DICTIONARY_DIR / f"{lang_code}.txt"
            download_dictionary(lang_code, url, output_path)
        else:
            print(f"Unknown language: {lang_code}")
            return 1
    
    print("=" * 50)
    print("Note: Downloaded dictionaries may need filtering and processing.")
    print("Existing dictionary files in data/dictionaries/ contain")
    print("curated word lists suitable for gesture typing.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
