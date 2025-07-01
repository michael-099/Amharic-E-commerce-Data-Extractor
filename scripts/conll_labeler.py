"""
CoNLL format data labeling tool for Amharic NER.
This module helps create labeled training data in CoNLL format.
"""

import re
from typing import List, Tuple, Dict
import json


class CoNLLLabeler:
    """Tool for creating CoNLL format labeled data."""
    
    def __init__(self):
        self.entity_patterns = {
            'PRODUCT': [
                r'(ሻሚዝ|ቀሚስ|ሱሪ|ጫማ|ቦርሳ|ሰዓት|ስልክ|ኮምፒውተር|ላፕቶፕ)',
                r'(መጽሐፍ|ብዕር|ወረቀት|ፋይል)',
                r'(ሻይ|ቡና|ስኳር|ዘይት|ሩዝ|ዳቦ)',
                r'(ሳሙና|ሻምፖ|ክሬም|ፍርፋሪ)',
                r'(ቴሌቪዥን|ራዲዮ|ስፒከር|ካሜራ)',
            ],
            'PRICE': [
                r'(ዋጋ[፡:]?\s*\d+\s*ብር)',
                r'(\d+\s*ብር)',
                r'(በ\s*\d+\s*ብር)',
                r'(ዋጋ\s*\d+)',
                r'(\d+\s*ፍሬ)',
            ],
            'LOC': [
                r'(አዲስ\s*አበባ|አዲስአበባ)',
                r'(ቦሌ|መገናኛ|መርካቶ|ፒያሳ|ካዛንቺስ|ሰሚት|ለቡ)',
                r'(ሞል|ሱቅ|ማዕከል|ቢሮ)',
                r'(አድራሻ)',
            ]
        }
    
    def tokenize_amharic(self, text: str) -> List[str]:
        """Simple tokenization for Amharic text."""
        # Split on whitespace and punctuation
        tokens = re.findall(r'[\u1200-\u137F0-9]+|[፡።፣፤፥፦፧]', text)
        return [token for token in tokens if token.strip()]
    
    def auto_label_text(self, text: str) -> List[Tuple[str, str]]:
        """Automatically label text using pattern matching."""
        tokens = self.tokenize_amharic(text)
        labels = ['O'] * len(tokens)
        
        # Join tokens back to text for pattern matching
        full_text = ' '.join(tokens)
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
                for match in matches:
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Find which tokens this match covers
                    current_pos = 0
                    start_token = None
                    end_token = None
                    
                    for i, token in enumerate(tokens):
                        token_start = current_pos
                        token_end = current_pos + len(token)
                        
                        if start_token is None and token_start <= start_pos < token_end:
                            start_token = i
                        if end_token is None and token_start < end_pos <= token_end:
                            end_token = i
                            break
                        
                        current_pos = token_end + 1  # +1 for space
                    
                    # Label the tokens
                    if start_token is not None and end_token is not None:
                        for j in range(start_token, end_token + 1):
                            if j == start_token:
                                labels[j] = f'B-{entity_type}'
                            else:
                                labels[j] = f'I-{entity_type}'
        
        return list(zip(tokens, labels))
    
    def create_conll_from_messages(self, messages: List[str], output_file: str = "Data/auto_labeled_conll.txt"):
        """Create CoNLL format file from list of messages."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, message in enumerate(messages):
                if not message.strip():
                    continue
                
                labeled_tokens = self.auto_label_text(message)
                
                # Write tokens and labels
                for token, label in labeled_tokens:
                    f.write(f"{token} {label}\n")
                
                # Empty line between sentences
                f.write("\n")
        
        print(f"Created CoNLL file with {len(messages)} messages: {output_file}")
    
    def load_existing_conll(self, file_path: str) -> List[List[Tuple[str, str]]]:
        """Load existing CoNLL format data."""
        sentences = []
        current_sentence = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        token = parts[0]
                        label = parts[1]
                        current_sentence.append((token, label))
        
        if current_sentence:
            sentences.append(current_sentence)
        
        return sentences
    
    def merge_conll_files(self, file_paths: List[str], output_file: str):
        """Merge multiple CoNLL files."""
        all_sentences = []
        
        for file_path in file_paths:
            try:
                sentences = self.load_existing_conll(file_path)
                all_sentences.extend(sentences)
                print(f"Loaded {len(sentences)} sentences from {file_path}")
            except FileNotFoundError:
                print(f"File not found: {file_path}")
        
        # Write merged data
        with open(output_file, 'w', encoding='utf-8') as f:
            for sentence in all_sentences:
                for token, label in sentence:
                    f.write(f"{token} {label}\n")
                f.write("\n")
        
        print(f"Merged {len(all_sentences)} sentences to {output_file}")
    
    def validate_conll_format(self, file_path: str) -> Dict[str, int]:
        """Validate CoNLL format and return statistics."""
        stats = {
            'total_sentences': 0,
            'total_tokens': 0,
            'entity_counts': {},
            'errors': []
        }
        
        sentences = self.load_existing_conll(file_path)
        stats['total_sentences'] = len(sentences)
        
        for i, sentence in enumerate(sentences):
            stats['total_tokens'] += len(sentence)
            
            for j, (token, label) in enumerate(sentence):
                # Count entities
                if label != 'O':
                    entity_type = label.split('-')[-1]
                    stats['entity_counts'][entity_type] = stats['entity_counts'].get(entity_type, 0) + 1
                
                # Validate label format
                if label not in ['O'] and not re.match(r'^[BI]-[A-Z]+$', label):
                    stats['errors'].append(f"Sentence {i+1}, Token {j+1}: Invalid label '{label}'")
        
        return stats


def create_sample_labeled_data():
    """Create sample labeled data for demonstration."""
    labeler = CoNLLLabeler()
    
    # Sample Amharic messages
    sample_messages = [
        "ዋጋ 500 ብር የሆነ ሻሚዝ በቦሌ ይሸጣል",
        "አዲስ አበባ መገናኛ ሞል ውስጥ ላፕቶፕ ዋጋ 25000 ብር",
        "ቡና 150 ብር ሻይ 80 ብር በመርካቶ",
        "ሳሙና እና ሻምፖ በ 200 ብር ለቡ አድራሻ",
        "ስልክ ዋጋ 8000 ብር ቦሌ ሱቅ ውስጥ"
    ]
    
    labeler.create_conll_from_messages(sample_messages, "Data/sample_labeled.txt")
    
    # Validate the created file
    stats = labeler.validate_conll_format("Data/sample_labeled.txt")
    print("\n=== Validation Results ===")
    print(f"Total sentences: {stats['total_sentences']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Entity counts: {stats['entity_counts']}")
    if stats['errors']:
        print(f"Errors found: {len(stats['errors'])}")
        for error in stats['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")


def main():
    """Main function to process data and create labeled dataset."""
    labeler = CoNLLLabeler()

    # Load processed data from cleaned_data.jsonl
    try:
        messages = []
        with open("cleaned_data.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    clean_text = obj.get('clean_text', '')
                    if clean_text and len(clean_text) > 10:
                        messages.append(clean_text)
                except json.JSONDecodeError:
                    continue

        if messages:
            print(f"Processing {len(messages)} messages for labeling...")
            labeler.create_conll_from_messages(messages[:50], "Data/auto_labeled_training.txt")  # Limit to 50 for now

            # Merge with existing labeled data if available
            existing_files = ["Data/labeled_telegram_product_price_location.txt", "Data/auto_labeled_training.txt"]
            labeler.merge_conll_files(existing_files, "Data/merged_labeled_data.txt")

            # Validate merged data
            stats = labeler.validate_conll_format("Data/merged_labeled_data.txt")
            print("\n=== Final Dataset Statistics ===")
            print(f"Total sentences: {stats['total_sentences']}")
            print(f"Total tokens: {stats['total_tokens']}")
            print(f"Entity counts: {stats['entity_counts']}")
        else:
            print("No messages found. Creating sample labeled data...")
            create_sample_labeled_data()

    except FileNotFoundError:
        print("Cleaned data not found. Creating sample labeled data...")
        create_sample_labeled_data()


if __name__ == "__main__":
    main()
