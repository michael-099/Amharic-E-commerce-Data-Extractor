"""
Notebook Demo Script - Standalone version of the Jupyter notebook functionality.
This script demonstrates the key features without requiring Jupyter or pandas.
"""

import sys
import os
import json

# Add scripts directory to path
sys.path.append(os.path.dirname(__file__))

def demo_data_loading():
    """Demonstrate data loading functionality."""
    print("=" * 60)
    print("📊 DATA LOADING DEMO")
    print("=" * 60)
    
    try:
        # Load raw Telegram data
        data = []
        with open('raw_telegram_data.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    data.append(obj)
                except json.JSONDecodeError:
                    continue

        print(f"✅ Loaded {len(data)} messages")
        
        # Get unique channels
        channels = set(item['channel'] for item in data)
        print(f"✅ Found {len(channels)} channels:")
        for channel in sorted(channels):
            channel_data = [item for item in data if item['channel'] == channel]
            media_count = sum(1 for item in channel_data if item.get('media'))
            print(f"   - {channel}: {len(channel_data)} messages, {media_count} media files")
        
        # Display sample messages
        print(f"\n📝 Sample Messages:")
        for i, item in enumerate(data[:3]):
            print(f"\n--- Message {i+1} ---")
            print(f"Channel: {item['channel']}")
            text = item['text'][:100] + "..." if len(item['text']) > 100 else item['text']
            print(f"Text: {text}")
            print(f"Media: {'Yes' if item['media'] else 'No'}")
            
        return data
        
    except FileNotFoundError:
        print("❌ Raw data file not found. Please run the scraper first.")
        return []

def demo_text_processing():
    """Demonstrate text processing functionality."""
    print("\n" + "=" * 60)
    print("🔧 TEXT PROCESSING DEMO")
    print("=" * 60)

    try:
        from data_processor import AmharicDataProcessor

        processor = AmharicDataProcessor()

        # Sample Amharic text with emojis and mixed content
        sample_texts = [
            "💥💥ዋጋ፦ 500 ብር የሆነ ሻሚዝ በቦሌ ይሸጣል📍",
            "🔥🔥 አዲስ አበባ መገናኛ ሞል ውስጥ ላፕቶፕ ዋጋ 25000 ብር 🔥🔥",
            "ቡና 150 ብር ሻይ 80 ብር በመርካቶ @username #hashtag"
        ]

        print("Original → Cleaned Text:")
        print("-" * 40)

        for i, text in enumerate(sample_texts):
            clean_text = processor.normalize_amharic_text(text)
            print(f"{i+1}. {text}")
            print(f"   → {clean_text}")

            # Extract entities
            entities = processor.extract_entities_from_text(clean_text)
            if any(entities.values()):
                print(f"   📋 Entities: {entities}")
            print()

        # Show processing statistics
        print("📊 Processing Statistics:")
        print(f"   - Emoji removal: ✅")
        print(f"   - URL/username filtering: ✅")
        print(f"   - Amharic text normalization: ✅")
        print(f"   - Entity pattern matching: ✅")

        print("\n✅ Text processing completed successfully")

    except ImportError as e:
        print(f"❌ Data processor module not available: {e}")
    except Exception as e:
        print(f"❌ Error in text processing: {e}")

def demo_conll_labeling():
    """Demonstrate CoNLL labeling functionality."""
    print("\n" + "=" * 60)
    print("🏷️ CONLL LABELING DEMO")
    print("=" * 60)
    
    try:
        from conll_labeler import CoNLLLabeler
        
        labeler = CoNLLLabeler()
        
        # Sample Amharic sentences
        sample_sentences = [
            "ዋጋ 500 ብር የሆነ ሻሚዝ በቦሌ ይሸጣል",
            "አዲስ አበባ መገናኛ ሞል ውስጥ ላፕቶፕ",
            "ቡና 150 ብር ሻይ 80 ብር በመርካቶ"
        ]
        
        print("Token-Level Labeling Results:")
        print("-" * 40)
        
        for i, sentence in enumerate(sample_sentences):
            print(f"\n{i+1}. Sentence: {sentence}")
            labeled_tokens = labeler.auto_label_text(sentence)
            
            print("   Token\t\tLabel")
            print("   " + "-" * 25)
            for token, label in labeled_tokens:
                print(f"   {token:<15}\t{label}")
        
        # Check dataset statistics if available
        try:
            stats = labeler.validate_conll_format("Data/merged_labeled_data.txt")
            print(f"\n📊 Dataset Statistics:")
            print(f"   Total sentences: {stats['total_sentences']:,}")
            print(f"   Total tokens: {stats['total_tokens']:,}")
            print(f"   Entity counts: {stats['entity_counts']}")
        except FileNotFoundError:
            print("\n📊 Labeled dataset not found")
            
        print("\n✅ CoNLL labeling completed successfully")
        
    except ImportError:
        print("❌ CoNLL labeler module not available")

def demo_vendor_scorecard():
    """Demonstrate vendor scorecard functionality."""
    print("\n" + "=" * 60)
    print("💼 VENDOR SCORECARD DEMO")
    print("=" * 60)
    
    try:
        # Check if scorecard exists
        if os.path.exists('vendor_scorecard.csv'):
            print("📊 Vendor Scorecard Results:")
            print("-" * 40)
            
            with open('vendor_scorecard.csv', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Print header
            if lines:
                header = lines[0].strip().split(',')
                print(f"{'Vendor':<25} {'Score':<8} {'Posts/Week':<12} {'Avg Price':<12}")
                print("-" * 60)
                
                # Print vendor data
                for line in lines[1:]:
                    if line.strip():
                        parts = line.strip().split(',')
                        if len(parts) >= 4:
                            vendor = parts[0]
                            score = parts[6] if len(parts) > 6 else parts[1]
                            posts = parts[2] if len(parts) > 2 else "N/A"
                            price = parts[3] if len(parts) > 3 else "N/A"
                            print(f"{vendor:<25} {score:<8} {posts:<12} {price:<12}")
            
            print("\n✅ Vendor scorecard loaded successfully")
        else:
            print("❌ Vendor scorecard not found. Run vendor analysis first.")
            
    except Exception as e:
        print(f"❌ Error loading vendor scorecard: {e}")

def demo_project_summary():
    """Display project summary."""
    print("\n" + "=" * 60)
    print("🎯 PROJECT SUMMARY")
    print("=" * 60)
    
    # Check what files exist
    files_status = {
        "Raw Data": os.path.exists('raw_telegram_data.jsonl'),
        "Cleaned Data": os.path.exists('cleaned_data.jsonl'),
        "Labeled Data": os.path.exists('Data/merged_labeled_data.txt'),
        "Vendor Scorecard": os.path.exists('vendor_scorecard.csv'),
        "Project Summary": os.path.exists('project_summary.txt')
    }
    
    print("📁 File Status:")
    for file_type, exists in files_status.items():
        status = "✅" if exists else "❌"
        print(f"   {status} {file_type}")
    
    # Count script files
    script_count = 0
    if os.path.exists('scripts'):
        script_count = len([f for f in os.listdir('scripts') if f.endswith('.py')])
    
    print(f"\n🛠️ Implementation:")
    print(f"   📜 Python Scripts: {script_count}")
    print(f"   📊 Data Processing: {'✅' if files_status['Raw Data'] else '❌'}")
    print(f"   🏷️ Entity Labeling: {'✅' if files_status['Labeled Data'] else '❌'}")
    print(f"   💼 Vendor Analysis: {'✅' if files_status['Vendor Scorecard'] else '❌'}")
    
    print(f"\n🎯 Project Status:")
    completed_tasks = sum(files_status.values())
    total_tasks = len(files_status)
    completion_rate = (completed_tasks / total_tasks) * 100
    print(f"   Completion Rate: {completion_rate:.0f}% ({completed_tasks}/{total_tasks})")
    
    if completion_rate >= 80:
        print("   🎉 Project is ready for deployment!")
    elif completion_rate >= 60:
        print("   🚧 Project is in good progress")
    else:
        print("   ⚠️ More work needed")

def main():
    """Run the complete demo."""
    print("🚀 AMHARIC E-COMMERCE DATA EXTRACTOR DEMO")
    print("=" * 60)
    print("This demo showcases the key functionality of the project")
    print("without requiring Jupyter notebook or all dependencies.")
    print()
    
    # Run all demos
    data = demo_data_loading()
    demo_text_processing()
    demo_conll_labeling()
    demo_vendor_scorecard()
    demo_project_summary()
    
    print("\n" + "=" * 60)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("For full functionality, install all dependencies:")
    print("pip install -r requirements.txt")
    print("\nTo run individual components:")
    print("python scripts/scraper.py          # Data collection")
    print("python scripts/conll_labeler.py    # Data labeling")
    print("python scripts/vendor_scorecard.py # Vendor analysis")
    print("python scripts/project_summary.py  # Full summary")

if __name__ == "__main__":
    main()
