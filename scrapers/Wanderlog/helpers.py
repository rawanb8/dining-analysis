from config import CUISINES

def extract_cuisine(text):
    text_lower = text.lower()
    found = [c.title() for c in CUISINES if c in text_lower]
    return ", ".join(found) if found else None