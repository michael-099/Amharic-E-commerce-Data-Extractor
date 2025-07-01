import re
import json

def normalize_amharic(text):
    if not text:
        return ''
    
    # Expanded emoji removal (including more Unicode ranges)
    text = re.sub("["
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        u"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251"  # Enclosed characters
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "]+", '', text, flags=re.UNICODE)

    # Remove links, usernames, and English words
    text = re.sub(r'https?://\S+|www\.\S+|@\w+|[a-zA-Z]+', '', text)

    # Remove all punctuation and symbols, including dots
    text = re.sub(r'[.()_+=\\\[\]{}<>:"\'“”#|*~`!@^$%&?,/;-]', '', text)

    # Keep only Amharic characters, numbers, and specific Amharic punctuation
    text = re.sub(r'[^\u1200-\u137F0-9፡።፣፤፥፦፧]', '', text)

    # Collapse multiple spaces and strip
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Read -> Clean -> Write
cleaned_data = []

with open('raw_telegram_data.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        original = obj.get('text', '')
        cleaned = normalize_amharic(original)
        obj['clean_text'] = cleaned
        cleaned_data.append(obj)

        # Debug print just to verify
        if i < 2:  # Show only first 2 lines for confirmation
            print(f"\nOriginal:\n{original}\nCleaned:\n{cleaned}\n")

# Save to cleaned_data.jsonl
with open('cleaned_data.jsonl', 'w', encoding='utf-8') as f:
    for entry in cleaned_data:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')
