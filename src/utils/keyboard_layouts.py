"""Keyboard layout definitions for different languages."""

from typing import Dict, Tuple

# QWERTY layout (used for English)
QWERTY_LAYOUT = {
    'q': (0, 0), 'w': (1, 0), 'e': (2, 0), 'r': (3, 0), 't': (4, 0),
    'y': (5, 0), 'u': (6, 0), 'i': (7, 0), 'o': (8, 0), 'p': (9, 0),
    'a': (0, 1), 's': (1, 1), 'd': (2, 1), 'f': (3, 1), 'g': (4, 1),
    'h': (5, 1), 'j': (6, 1), 'k': (7, 1), 'l': (8, 1),
    'z': (0, 2), 'x': (1, 2), 'c': (2, 2), 'v': (3, 2), 'b': (4, 2),
    'n': (5, 2), 'm': (6, 2)
}

# AZERTY layout (used for French)
AZERTY_LAYOUT = {
    'a': (0, 0), 'z': (1, 0), 'e': (2, 0), 'r': (3, 0), 't': (4, 0),
    'y': (5, 0), 'u': (6, 0), 'i': (7, 0), 'o': (8, 0), 'p': (9, 0),
    'q': (0, 1), 's': (1, 1), 'd': (2, 1), 'f': (3, 1), 'g': (4, 1),
    'h': (5, 1), 'j': (6, 1), 'k': (7, 1), 'l': (8, 1), 'm': (9, 1),
    'w': (0, 2), 'x': (1, 2), 'c': (2, 2), 'v': (3, 2), 'b': (4, 2),
    'n': (5, 2)
}

# Spanish uses QWERTY layout with additional characters
SPANISH_LAYOUT = QWERTY_LAYOUT.copy()
SPANISH_LAYOUT.update({
    'ñ': (7.5, 1),
})


def get_keyboard_layout(language: str = 'en_US') -> Dict[str, Tuple[float, float]]:
    """Get keyboard layout for a specific language.
    
    Args:
        language: Language code (en_US, es_ES, fr_FR)
        
    Returns:
        Dictionary mapping characters to (x, y) coordinates
    """
    layouts = {
        'en_US': QWERTY_LAYOUT,
        'es_ES': SPANISH_LAYOUT,
        'fr_FR': AZERTY_LAYOUT
    }
    return layouts.get(language, QWERTY_LAYOUT)


def word_to_gesture(word: str, layout: Dict[str, Tuple[float, float]]) -> list:
    """Convert a word to gesture coordinates based on keyboard layout.
    
    Args:
        word: Word to convert
        layout: Keyboard layout dictionary
        
    Returns:
        List of (x, y) coordinate tuples
    """
    coordinates = []
    for char in word.lower():
        if char in layout:
            coordinates.append(layout[char])
    return coordinates
