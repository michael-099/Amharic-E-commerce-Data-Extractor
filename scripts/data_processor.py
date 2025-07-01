"""
Enhanced data processing for Amharic e-commerce data extraction.
This module handles data cleaning, preprocessing, and preparation for NER training.
"""

import json
import re
from typing import List, Dict, Any, Tuple
import os
from pathlib import Path

# Try to import pandas, but make it optional
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class AmharicDataProcessor:
    """Enhanced processor for Amharic e-commerce data."""
    
    def __init__(self, data_dir: str = "Data"):
        self.data_dir = Path(data_dir)
        self.amharic_pattern = re.compile(r'[\u1200-\u137F]')
        
    def normalize_amharic_text(self, text: str) -> str:
        """Enhanced Amharic text normalization."""
        if not text:
            return ''
        
        # Remove emojis and special characters
        text = re.sub("["
            u"\U0001F600-\U0001F64F"  # Emoticons
            u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
            u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
            u"\U0001F1E0-\U0001F1FF"  # Flags
            u"\U00002702-\U000027B0"  # Dingbats
            u"\U000024C2-\U0001F251"  # Enclosed characters
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols
            u"\U0001FA70-\U0001FAFF"  # Extended symbols
            "]+", '', text, flags=re.UNICODE)

        # Remove URLs, usernames, and English words
        text = re.sub(r'https?://\S+|www\.\S+|@\w+|[a-zA-Z]+', '', text)
        
        # Remove punctuation but keep Amharic punctuation
        text = re.sub(r'[.()_+=\\\[\]{}<>:"\'""#|*~`!@^$%&?,/;-]', ' ', text)
        
        # Keep only Amharic characters, numbers, and Amharic punctuation
        text = re.sub(r'[^\u1200-\u137F0-9፡።፣፤፥፦፧\s]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_entities_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extract potential entities from Amharic text."""
        entities = {
            'products': [],
            'prices': [],
            'locations': []
        }
        
        # Price patterns
        price_patterns = [
            r'ዋጋ[፡:]?\s*(\d+)\s*ብር',
            r'(\d+)\s*ብር',
            r'በ\s*(\d+)\s*ብር',
            r'ዋጋ\s*(\d+)',
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['prices'].extend(matches)
        
        # Location patterns (common Ethiopian locations)
        location_patterns = [
            r'(አዲስ\s*አበባ|አዲስአበባ)',
            r'(ቦሌ)',
            r'(መገናኛ)',
            r'(መርካቶ)',
            r'(ፒያሳ)',
            r'(ካዛንቺስ)',
            r'(ሰሚት)',
            r'(ለቡ)',
            r'(ሞል)',
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['locations'].extend([match if isinstance(match, str) else match[0] for match in matches])
        
        return entities
    
    def load_scraped_data(self, file_path: str = "raw_telegram_data.jsonl"):
        """Load and process scraped Telegram data."""
        data = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    if obj.get('text'):  # Only process messages with text
                        # Process the text
                        clean_text = self.normalize_amharic_text(obj['text'])
                        has_amharic = bool(self.amharic_pattern.search(clean_text))
                        text_length = len(clean_text)

                        # Only include messages with Amharic content
                        if has_amharic and text_length > 10:
                            obj['clean_text'] = clean_text
                            obj['has_amharic'] = has_amharic
                            obj['text_length'] = text_length
                            obj['extracted_entities'] = self.extract_entities_from_text(clean_text)
                            data.append(obj)
                except json.JSONDecodeError:
                    continue

        return data
    
    def prepare_training_data(self, data: List[Dict], sample_size: int = 100) -> List[Dict]:
        """Prepare a subset of data for manual labeling."""
        import random
        random.seed(42)

        # Sample diverse messages
        sampled_data = random.sample(data, min(sample_size, len(data)))

        # Prioritize messages with potential entities
        entity_messages = [
            item for item in data
            if (len(item['extracted_entities']['products']) > 0 or
                len(item['extracted_entities']['prices']) > 0 or
                len(item['extracted_entities']['locations']) > 0)
        ]

        if len(entity_messages) > 0:
            entity_sample = random.sample(entity_messages, min(50, len(entity_messages)))
            # Combine and remove duplicates
            all_ids = set(item.get('id', id(item)) for item in sampled_data)
            for item in entity_sample:
                item_id = item.get('id', id(item))
                if item_id not in all_ids:
                    sampled_data.append(item)
                    all_ids.add(item_id)

        return sampled_data
    
    def save_processed_data(self, data: List[Dict], output_path: str = "processed_telegram_data.json"):
        """Save processed data to JSON."""
        # Flatten extracted entities for easier access
        processed_data = []
        for item in data:
            processed_item = item.copy()
            entities = item.get('extracted_entities', {})
            processed_item['extracted_products'] = ', '.join(entities.get('products', []))
            processed_item['extracted_prices'] = ', '.join(entities.get('prices', []))
            processed_item['extracted_locations'] = ', '.join(entities.get('locations', []))
            processed_data.append(processed_item)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False, default=str)

        print(f"Processed data saved to {output_path}")
        return processed_data


def main():
    """Main processing pipeline."""
    processor = AmharicDataProcessor()

    print("Loading scraped data...")
    data = processor.load_scraped_data()
    print(f"Loaded {len(data)} messages with Amharic content")

    print("Preparing training data sample...")
    training_sample = processor.prepare_training_data(data, sample_size=100)
    print(f"Prepared {len(training_sample)} messages for labeling")

    print("Saving processed data...")
    processor.save_processed_data(data, "Data/processed_telegram_data.json")
    processor.save_processed_data(training_sample, "Data/training_sample.json")

    # Print some statistics
    print("\n=== Data Statistics ===")
    print(f"Total messages: {len(data)}")

    text_lengths = [item['text_length'] for item in data]
    avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    print(f"Average text length: {avg_length:.1f}")

    price_messages = sum(1 for item in data if len(item['extracted_entities']['prices']) > 0)
    location_messages = sum(1 for item in data if len(item['extracted_entities']['locations']) > 0)
    print(f"Messages with prices: {price_messages}")
    print(f"Messages with locations: {location_messages}")

    unique_channels = set(item['channel'] for item in data)
    print(f"Unique channels: {len(unique_channels)}")

    return data, training_sample


if __name__ == "__main__":
    df, training_sample = main()
