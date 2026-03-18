import re

def clean_text(text):
    # Remove HTML breaks often found in this dataset
    text = re.sub(r'<br\s*/?>', ' ', text)
    # Keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercase and trim
    text = text.lower().strip()
    return text