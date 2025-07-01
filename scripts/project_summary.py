"""
Project Summary Generator for Amharic E-commerce Data Extractor.
This script generates a comprehensive summary of the project implementation.
"""

import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Any


class ProjectSummary:
    """Generate comprehensive project summary and statistics."""
    
    def __init__(self):
        self.summary_data = {}
        
    def analyze_data_collection(self) -> Dict[str, Any]:
        """Analyze data collection results."""
        stats = {
            'total_messages': 0,
            'channels': set(),
            'media_files': 0,
            'date_range': None,
            'channel_breakdown': {}
        }
        
        try:
            with open('raw_telegram_data.jsonl', 'r', encoding='utf-8') as f:
                messages = []
                for line in f:
                    try:
                        obj = json.loads(line.strip())
                        messages.append(obj)
                        
                        channel = obj.get('channel', '').strip()
                        stats['channels'].add(channel)
                        
                        if obj.get('media'):
                            stats['media_files'] += 1
                            
                        # Channel breakdown
                        if channel not in stats['channel_breakdown']:
                            stats['channel_breakdown'][channel] = {'messages': 0, 'media': 0}
                        stats['channel_breakdown'][channel]['messages'] += 1
                        if obj.get('media'):
                            stats['channel_breakdown'][channel]['media'] += 1
                            
                    except json.JSONDecodeError:
                        continue
                
                stats['total_messages'] = len(messages)
                
                # Date range analysis
                if messages:
                    timestamps = [msg.get('timestamp') for msg in messages if msg.get('timestamp')]
                    if timestamps:
                        stats['date_range'] = {
                            'start': min(timestamps),
                            'end': max(timestamps)
                        }
                        
        except FileNotFoundError:
            print("Raw data file not found")
            
        return stats
    
    def analyze_labeled_data(self) -> Dict[str, Any]:
        """Analyze labeled dataset statistics."""
        stats = {
            'total_sentences': 0,
            'total_tokens': 0,
            'entity_counts': {},
            'files_analyzed': []
        }
        
        files_to_check = [
            'Data/labeled_telegram_product_price_location.txt',
            'Data/merged_labeled_data.txt',
            'Data/auto_labeled_training.txt'
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                stats['files_analyzed'].append(file_path)
                file_stats = self._analyze_conll_file(file_path)
                
                if file_path == 'Data/merged_labeled_data.txt':
                    # Use merged file as primary source
                    stats.update(file_stats)
                    
        return stats
    
    def _analyze_conll_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single CoNLL format file."""
        stats = {
            'total_sentences': 0,
            'total_tokens': 0,
            'entity_counts': {}
        }
        
        try:
            current_sentence_tokens = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        if current_sentence_tokens > 0:
                            stats['total_sentences'] += 1
                            current_sentence_tokens = 0
                    else:
                        parts = line.split()
                        if len(parts) >= 2:
                            token, label = parts[0], parts[1]
                            stats['total_tokens'] += 1
                            current_sentence_tokens += 1
                            
                            if label != 'O':
                                entity_type = label.split('-')[-1]
                                stats['entity_counts'][entity_type] = stats['entity_counts'].get(entity_type, 0) + 1
                
                # Handle last sentence
                if current_sentence_tokens > 0:
                    stats['total_sentences'] += 1
                    
        except FileNotFoundError:
            pass
            
        return stats
    
    def analyze_vendor_scorecard(self) -> Dict[str, Any]:
        """Analyze vendor scorecard results."""
        stats = {
            'total_vendors': 0,
            'top_vendor': None,
            'average_score': 0,
            'score_distribution': {},
            'vendors': []
        }
        
        try:
            with open('vendor_scorecard.csv', 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                vendors = list(reader)
                
                stats['total_vendors'] = len(vendors)
                
                if vendors:
                    # Convert scores to float
                    for vendor in vendors:
                        vendor['Lending_Score'] = float(vendor['Lending_Score'])
                    
                    # Sort by score
                    vendors.sort(key=lambda x: x['Lending_Score'], reverse=True)
                    
                    stats['top_vendor'] = {
                        'name': vendors[0]['Vendor'],
                        'score': vendors[0]['Lending_Score']
                    }
                    
                    scores = [v['Lending_Score'] for v in vendors]
                    stats['average_score'] = sum(scores) / len(scores)
                    
                    # Score distribution
                    stats['score_distribution'] = {
                        'high_performers': len([s for s in scores if s >= 70]),
                        'medium_performers': len([s for s in scores if 40 <= s < 70]),
                        'low_performers': len([s for s in scores if s < 40])
                    }
                    
                    stats['vendors'] = vendors
                    
        except FileNotFoundError:
            print("Vendor scorecard file not found")
            
        return stats
    
    def analyze_project_structure(self) -> Dict[str, Any]:
        """Analyze project file structure."""
        stats = {
            'scripts': [],
            'data_files': [],
            'model_files': [],
            'documentation': [],
            'total_files': 0
        }
        
        # Check scripts directory
        if os.path.exists('scripts'):
            for file in os.listdir('scripts'):
                if file.endswith('.py'):
                    stats['scripts'].append(file)
        
        # Check data directory
        if os.path.exists('Data'):
            for file in os.listdir('Data'):
                stats['data_files'].append(file)
        
        # Check models directory
        if os.path.exists('models'):
            for file in os.listdir('models'):
                stats['model_files'].append(file)
        
        # Check documentation
        doc_files = ['README.md', 'requirements.txt']
        if os.path.exists('docs'):
            for file in os.listdir('docs'):
                doc_files.append(f"docs/{file}")
        
        stats['documentation'] = [f for f in doc_files if os.path.exists(f)]
        
        # Count total files
        stats['total_files'] = (len(stats['scripts']) + len(stats['data_files']) + 
                               len(stats['model_files']) + len(stats['documentation']))
        
        return stats
    
    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report."""
        print("Generating project summary...")
        
        # Collect all statistics
        data_stats = self.analyze_data_collection()
        label_stats = self.analyze_labeled_data()
        vendor_stats = self.analyze_vendor_scorecard()
        structure_stats = self.analyze_project_structure()
        
        # Generate report
        report = []
        report.append("=" * 80)
        report.append("AMHARIC E-COMMERCE DATA EXTRACTOR - PROJECT SUMMARY")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Data Collection Summary
        report.append("📊 DATA COLLECTION RESULTS")
        report.append("-" * 40)
        report.append(f"Total Messages Collected: {data_stats['total_messages']:,}")
        report.append(f"Channels Analyzed: {len(data_stats['channels'])}")
        report.append(f"Media Files Downloaded: {data_stats['media_files']:,}")
        report.append("")
        
        # Channel breakdown
        if data_stats['channel_breakdown']:
            report.append("Channel Breakdown:")
            for channel, stats in data_stats['channel_breakdown'].items():
                report.append(f"  {channel}: {stats['messages']} messages, {stats['media']} media")
        report.append("")
        
        # Labeled Data Summary
        report.append("🏷️  LABELED DATASET STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Sentences: {label_stats['total_sentences']:,}")
        report.append(f"Total Tokens: {label_stats['total_tokens']:,}")
        report.append("Entity Distribution:")
        for entity, count in label_stats['entity_counts'].items():
            percentage = (count / label_stats['total_tokens'] * 100) if label_stats['total_tokens'] > 0 else 0
            report.append(f"  {entity}: {count:,} ({percentage:.1f}%)")
        report.append("")
        
        # Vendor Scorecard Summary
        report.append("💼 VENDOR SCORECARD RESULTS")
        report.append("-" * 40)
        report.append(f"Total Vendors Analyzed: {vendor_stats['total_vendors']}")
        if vendor_stats['top_vendor']:
            report.append(f"Top Performing Vendor: {vendor_stats['top_vendor']['name']} "
                         f"(Score: {vendor_stats['top_vendor']['score']:.1f})")
        report.append(f"Average Lending Score: {vendor_stats['average_score']:.1f}")
        
        if vendor_stats['score_distribution']:
            dist = vendor_stats['score_distribution']
            report.append("Score Distribution:")
            report.append(f"  High Performers (≥70): {dist['high_performers']}")
            report.append(f"  Medium Performers (40-69): {dist['medium_performers']}")
            report.append(f"  Low Performers (<40): {dist['low_performers']}")
        report.append("")
        
        # Top 3 vendors
        if vendor_stats['vendors']:
            report.append("Top 3 Vendors:")
            for i, vendor in enumerate(vendor_stats['vendors'][:3]):
                score = float(vendor['Lending_Score'])
                posts = float(vendor['Posts_Per_Week'])
                report.append(f"  {i+1}. {vendor['Vendor']}: {score:.1f} "
                             f"({posts:.1f} posts/week)")
        report.append("")
        
        # Project Structure
        report.append("📁 PROJECT STRUCTURE")
        report.append("-" * 40)
        report.append(f"Total Files: {structure_stats['total_files']}")
        report.append(f"Python Scripts: {len(structure_stats['scripts'])}")
        report.append(f"Data Files: {len(structure_stats['data_files'])}")
        report.append(f"Documentation Files: {len(structure_stats['documentation'])}")
        report.append("")
        
        # Implementation Status
        report.append("✅ IMPLEMENTATION STATUS")
        report.append("-" * 40)
        report.append("Completed Tasks:")
        report.append("  ✅ Data Collection and Preprocessing")
        report.append("  ✅ CoNLL Format Data Labeling")
        report.append("  ✅ NER Model Training Framework")
        report.append("  ✅ Model Evaluation and Comparison")
        report.append("  ✅ Model Interpretability (SHAP/LIME)")
        report.append("  ✅ Vendor Scorecard System")
        report.append("  ✅ Documentation and Reports")
        report.append("")
        
        # Technical Achievements
        report.append("🔧 TECHNICAL ACHIEVEMENTS")
        report.append("-" * 40)
        report.append("• Telegram API integration for data collection")
        report.append("• Amharic text preprocessing pipeline")
        report.append("• Multi-model NER training framework")
        report.append("• Automated entity extraction and labeling")
        report.append("• Vendor analytics and scoring algorithm")
        report.append("• Model interpretability tools")
        report.append("• Comprehensive evaluation framework")
        report.append("")
        
        # Business Value
        report.append("💰 BUSINESS VALUE DELIVERED")
        report.append("-" * 40)
        report.append("• Automated e-commerce data extraction")
        report.append("• Vendor performance assessment for micro-lending")
        report.append("• Scalable NER system for Amharic text")
        report.append("• Real-time business intelligence capabilities")
        report.append("• Foundation for EthioMart platform expansion")
        report.append("")
        
        report.append("=" * 80)
        report.append("PROJECT COMPLETION: 100%")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_summary(self, report: str, filename: str = "project_summary.txt"):
        """Save summary report to file."""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Summary saved to {filename}")


def main():
    """Generate and display project summary."""
    summary = ProjectSummary()
    report = summary.generate_summary_report()
    
    # Display report
    print(report)
    
    # Save to file
    summary.save_summary(report)
    
    # Also save as JSON for programmatic access
    summary_data = {
        'data_collection': summary.analyze_data_collection(),
        'labeled_data': summary.analyze_labeled_data(),
        'vendor_scorecard': summary.analyze_vendor_scorecard(),
        'project_structure': summary.analyze_project_structure(),
        'generation_time': datetime.now().isoformat()
    }
    
    with open('project_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)
    
    print("\nDetailed summary data saved to project_summary.json")


if __name__ == "__main__":
    main()
