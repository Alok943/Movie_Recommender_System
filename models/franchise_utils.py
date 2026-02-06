"""
Step 1.8: Bidirectional Franchise Detection
============================================
Handle: User searches for a sequel (Iron Man 3)
Goal: Show all movies in the franchise
"""

import pandas as pd
import re

def extract_sequel_number(title):
    """Extract base title and sequel number."""
    if not isinstance(title, str):
        return ("", 1)
    
    title_lower = title.lower()
    
    # Explicit numbers
    number_match = re.search(r'\b(\d+)\s*$', title_lower)
    if number_match:
        num = int(number_match.group(1))
        base = title_lower[:number_match.start()].strip()
        base = re.sub(r'[:\-]\s*$', '', base).strip()
        return (base, num)
    
    # Roman numerals
    roman_map = {'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6,
                 'vii': 7, 'viii': 8, 'ix': 9, 'x': 10}
    roman_match = re.search(r'\b(' + '|'.join(roman_map.keys()) + r')\s*$', title_lower)
    if roman_match:
        num = roman_map[roman_match.group(1)]
        base = title_lower[:roman_match.start()].strip()
        base = re.sub(r'[:\-]\s*$', '', base).strip()
        return (base, num)
    
    # "Part X"
    part_match = re.search(r'\bpart\s+(\d+|' + '|'.join(roman_map.keys()) + r')\s*$', title_lower)
    if part_match:
        part_str = part_match.group(1)
        num = int(part_str) if part_str.isdigit() else roman_map.get(part_str, 1)
        base = title_lower[:part_match.start()].strip()
        base = re.sub(r'\bpart\s*$', '', base).strip()
        base = re.sub(r'[:\-]\s*$', '', base).strip()
        return (base, num)
    
    # Subtitle sequels (conservative detection)
    ''' TODO: subtitle-based sequel detection (requires dataset context)
    if ':' in title_lower or ' - ' in title_lower:
        separator = ':' if ':' in title_lower else ' - '
        potential_base = title_lower.split(separator)[0].strip()
        base_normalized = re.sub(r'[^a-z0-9 ]', '', potential_base).strip()
        
        # Check if base exists in database
        matching = df[df['title'].str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True) == base_normalized]
        if len(matching) > 0:
            return (potential_base, 2)
    '''
    
    return (title_lower, 1)

def normalize_title_base(title):
    """Get clean base name."""
    base, _ = extract_sequel_number(title)
    base = re.sub(r'[^a-z0-9 ]', '', base)
    return ' '.join(base.split()).strip()

def are_same_franchise(title1, title2):
    """
    Check if two titles are from the same franchise.
    Returns: True/False
    """
    base1 = normalize_title_base(title1)
    base2 = normalize_title_base(title2)
    
    # Exact base match
    if base1 == base2:
        return True
    
    # One contains the other (for partial matches)
    if len(base1) > 3 and base1 in base2:
        return True
    if len(base2) > 3 and base2 in base1:
        return True
    
    return False

def get_franchise_score(search_title, search_companies, 
                       candidate_title, candidate_companies):
    """
    Score how related a candidate is to the search query.
    
    CRITICAL: If user searches "Iron Man 3", that movie should score highest!
    """
    
    # 1. EXACT MATCH - The movie they searched for
    if search_title.lower().strip() == candidate_title.lower().strip():
        return 1.0  # Perfect match
    
    # 2. Check if same franchise
    if not are_same_franchise(search_title, candidate_title):
        return 0.0  # Different franchise
    
    # 3. Same franchise - score based on production company
    score = 0.6  # Base score for franchise match
    
    if pd.notna(search_companies) and pd.notna(candidate_companies):
        search_set = {c.strip().lower() for c in str(search_companies).split(',')}
        cand_set = {c.strip().lower() for c in str(candidate_companies).split(',')}
        
        overlap = len(search_set & cand_set)
        if overlap > 0:
            score += 0.3  # Boost for same production company
    
    return score


'''
**Key Changes:**

1. **Exact match check first** → Score 1.0 if user searches for that exact movie
2. **Franchise check second** → Score 0.6-0.9 for same franchise
3. **Handles missing movies** → If "Avatar Fire and Ash" isn't in DB, still matches based on title

**Expected Output for "Iron Man 3":**
```
Score: 1.00 | Iron Man 3          ← The one they searched
Score: 0.90 | Iron Man (original)
Score: 0.90 | Iron Man 2
'''