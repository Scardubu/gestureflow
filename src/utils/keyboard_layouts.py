"""
Keyboard layout definitions.
"""
from typing import Dict, Tuple

# QWERTY layout (normalized coordinates)
QWERTY_LAYOUT = {
    'q': (0.0, 0.0), 'w': (0.1, 0.0), 'e': (0.2, 0.0), 'r': (0.3, 0.0),
    't': (0.4, 0.0), 'y': (0.5, 0.0), 'u': (0.6, 0.0), 'i': (0.7, 0.0),
    'o': (0.8, 0.0), 'p': (0.9, 0.0),
    'a': (0.05, 0.33), 's': (0.15, 0.33), 'd': (0.25, 0.33), 'f': (0.35, 0.33),
    'g': (0.45, 0.33), 'h': (0.55, 0.33), 'j': (0.65, 0.33), 'k': (0.75, 0.33),
    'l': (0.85, 0.33),
    'z': (0.15, 0.67), 'x': (0.25, 0.67), 'c': (0.35, 0.67), 'v': (0.45, 0.67),
    'b': (0.55, 0.67), 'n': (0.65, 0.67), 'm': (0.75, 0.67),
}

# AZERTY layout (French)
AZERTY_LAYOUT = {
    'a': (0.0, 0.0), 'z': (0.1, 0.0), 'e': (0.2, 0.0), 'r': (0.3, 0.0),
    't': (0.4, 0.0), 'y': (0.5, 0.0), 'u': (0.6, 0.0), 'i': (0.7, 0.0),
    'o': (0.8, 0.0), 'p': (0.9, 0.0),
    'q': (0.05, 0.33), 's': (0.15, 0.33), 'd': (0.25, 0.33), 'f': (0.35, 0.33),
    'g': (0.45, 0.33), 'h': (0.55, 0.33), 'j': (0.65, 0.33), 'k': (0.75, 0.33),
    'l': (0.85, 0.33), 'm': (0.95, 0.33),
    'w': (0.15, 0.67), 'x': (0.25, 0.67), 'c': (0.35, 0.67), 'v': (0.45, 0.67),
    'b': (0.55, 0.67), 'n': (0.65, 0.67),
}

# QWERTZ layout (German)
QWERTZ_LAYOUT = {
    'q': (0.0, 0.0), 'w': (0.1, 0.0), 'e': (0.2, 0.0), 'r': (0.3, 0.0),
    't': (0.4, 0.0), 'z': (0.5, 0.0), 'u': (0.6, 0.0), 'i': (0.7, 0.0),
    'o': (0.8, 0.0), 'p': (0.9, 0.0),
    'a': (0.05, 0.33), 's': (0.15, 0.33), 'd': (0.25, 0.33), 'f': (0.35, 0.33),
    'g': (0.45, 0.33), 'h': (0.55, 0.33), 'j': (0.65, 0.33), 'k': (0.75, 0.33),
    'l': (0.85, 0.33),
    'y': (0.15, 0.67), 'x': (0.25, 0.67), 'c': (0.35, 0.67), 'v': (0.45, 0.67),
    'b': (0.55, 0.67), 'n': (0.65, 0.67), 'm': (0.75, 0.67),
}


def get_layout(layout_name: str) -> Dict[str, Tuple[float, float]]:
    """
    Get keyboard layout by name.
    
    Args:
        layout_name: Layout name ('qwerty', 'azerty', 'qwertz')
        
    Returns:
        Layout dictionary
    """
    layouts = {
        'qwerty': QWERTY_LAYOUT,
        'azerty': AZERTY_LAYOUT,
        'qwertz': QWERTZ_LAYOUT,
    }
    
    return layouts.get(layout_name.lower(), QWERTY_LAYOUT)


def get_key_position(key: str, layout_name: str = 'qwerty') -> Tuple[float, float]:
    """
    Get position of a key in a layout.
    
    Args:
        key: Key character
        layout_name: Layout name
        
    Returns:
        (x, y) position
    """
    layout = get_layout(layout_name)
    return layout.get(key.lower(), (0.5, 0.5))


def calculate_distance(
    pos1: Tuple[float, float],
    pos2: Tuple[float, float]
) -> float:
    """
    Calculate Euclidean distance between two positions.
    
    Args:
        pos1: First position
        pos2: Second position
        
    Returns:
        Distance
    """
    import math
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def get_adjacent_keys(
    key: str,
    layout_name: str = 'qwerty',
    threshold: float = 0.15
) -> list:
    """
    Get keys adjacent to a given key.
    
    Args:
        key: Key character
        layout_name: Layout name
        threshold: Distance threshold for adjacency
        
    Returns:
        List of adjacent keys
    """
    layout = get_layout(layout_name)
    key_pos = layout.get(key.lower())
    
    if not key_pos:
        return []
    
    adjacent = []
    for k, pos in layout.items():
        if k != key.lower():
            distance = calculate_distance(key_pos, pos)
            if distance <= threshold:
                adjacent.append(k)
    
    return adjacent


def main():
    """Test keyboard layouts."""
    print("Available layouts:")
    for layout_name in ['qwerty', 'azerty', 'qwertz']:
        layout = get_layout(layout_name)
        print(f"\n{layout_name.upper()}: {len(layout)} keys")
        print(f"Sample keys: {list(layout.keys())[:5]}")
    
    print("\n\nTesting QWERTY layout:")
    test_key = 'd'
    pos = get_key_position(test_key)
    print(f"Position of '{test_key}': {pos}")
    
    adjacent = get_adjacent_keys(test_key)
    print(f"Adjacent keys to '{test_key}': {adjacent}")


if __name__ == "__main__":
    main()
