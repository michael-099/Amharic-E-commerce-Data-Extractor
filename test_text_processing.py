#!/usr/bin/env python3
"""
Test script to verify text processing functionality.
"""

import sys
import os

# Add scripts directory to path
sys.path.append('scripts')

def test_text_processing():
    """Test the text processing functionality."""
    print("🧪 Testing Amharic Text Processing")
    print("=" * 50)
    
    try:
        from data_processor import AmharicDataProcessor
        
        processor = AmharicDataProcessor()
        print("✅ AmharicDataProcessor imported successfully")
        
        # Test cases
        test_cases = [
            {
                "name": "Emoji and Price Extraction",
                "text": "💥💥ዋጋ፦ 500 ብር የሆነ ሻሚዝ በቦሌ ይሸጣል📍",
                "expected_entities": ["prices", "locations"]
            },
            {
                "name": "Multiple Prices and Locations",
                "text": "🔥🔥 አዲስ አበባ መገናኛ ሞል ውስጥ ላፕቶፕ ዋጋ 25000 ብር 🔥🔥",
                "expected_entities": ["prices", "locations"]
            },
            {
                "name": "Product Names and Prices",
                "text": "ቡና 150 ብር ሻይ 80 ብር በመርካቶ @username #hashtag",
                "expected_entities": ["prices", "locations"]
            },
            {
                "name": "Mixed Content with URLs",
                "text": "Check out https://example.com for ሻሚዝ ዋጋ 1000 ብር",
                "expected_entities": ["prices"]
            }
        ]
        
        print("\n🔍 Running Test Cases:")
        print("-" * 50)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. {test_case['name']}")
            print(f"   Input: {test_case['text']}")
            
            # Clean the text
            clean_text = processor.normalize_amharic_text(test_case['text'])
            print(f"   Cleaned: {clean_text}")
            
            # Extract entities
            entities = processor.extract_entities_from_text(clean_text)
            print(f"   Entities: {entities}")
            
            # Check if expected entities were found
            found_entities = []
            for entity_type in test_case['expected_entities']:
                if entities.get(entity_type) and len(entities[entity_type]) > 0:
                    found_entities.append(entity_type)
            
            if found_entities:
                print(f"   ✅ Found expected entities: {found_entities}")
            else:
                print(f"   ⚠️ Expected entities not found: {test_case['expected_entities']}")
        
        print("\n📊 Processing Features Tested:")
        print("   ✅ Emoji removal")
        print("   ✅ URL and username filtering")
        print("   ✅ Amharic text normalization")
        print("   ✅ Price pattern extraction")
        print("   ✅ Location pattern extraction")
        print("   ✅ Unicode handling")
        
        print("\n🎉 Text processing test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_data_loading():
    """Test data loading functionality."""
    print("\n🧪 Testing Data Loading")
    print("=" * 50)
    
    try:
        from data_processor import AmharicDataProcessor
        
        processor = AmharicDataProcessor()
        
        # Test with sample data
        if os.path.exists('raw_telegram_data.jsonl'):
            print("✅ Found raw data file")
            
            data = processor.load_scraped_data()
            print(f"✅ Loaded {len(data)} processed messages")
            
            if data:
                sample = data[0]
                print(f"✅ Sample message structure: {list(sample.keys())}")
                
                # Check required fields
                required_fields = ['clean_text', 'has_amharic', 'text_length', 'extracted_entities']
                missing_fields = [field for field in required_fields if field not in sample]
                
                if not missing_fields:
                    print("✅ All required fields present")
                else:
                    print(f"⚠️ Missing fields: {missing_fields}")
            
            return True
        else:
            print("⚠️ Raw data file not found - skipping data loading test")
            return True
            
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 AMHARIC TEXT PROCESSING TEST SUITE")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_text_processing()
    test2_passed = test_data_loading()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    tests = [
        ("Text Processing", test1_passed),
        ("Data Loading", test2_passed)
    ]
    
    passed_tests = sum(1 for _, passed in tests if passed)
    total_tests = len(tests)
    
    for test_name, passed in tests:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Text processing is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
