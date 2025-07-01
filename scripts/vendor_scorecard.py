"""
Vendor Scorecard System for Micro-Lending Assessment.
This module analyzes vendor performance and calculates lending scores.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import re
from collections import defaultdict
import csv
import statistics


class VendorAnalytics:
    """Analytics engine for vendor assessment."""
    
    def __init__(self):
        self.vendor_data = {}
        self.price_patterns = [
            r'ዋጋ[፡:]?\s*(\d+)\s*ብር',
            r'(\d+)\s*ብር',
            r'በ\s*(\d+)\s*ብር',
            r'ዋጋ\s*(\d+)',
        ]
    
    def load_telegram_data(self, file_path: str = "raw_telegram_data.jsonl") -> List[Dict]:
        """Load and process Telegram data."""
        data = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())

                    # Process timestamp
                    timestamp_str = obj.get('timestamp', '')
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    except:
                        timestamp = datetime.now()

                    # Extract prices
                    text = obj.get('text', '')
                    extracted_prices = self.extract_prices(text)

                    # Clean channel name
                    channel = obj.get('channel', '').replace('@', '').strip()

                    processed_obj = {
                        'channel': channel,
                        'text': text,
                        'timestamp': timestamp,
                        'extracted_prices': extracted_prices,
                        'has_price': len(extracted_prices) > 0,
                        'avg_price': statistics.mean(extracted_prices) if extracted_prices else 0,
                        'media': obj.get('media')
                    }

                    data.append(processed_obj)

                except json.JSONDecodeError:
                    continue

        return data
    
    def extract_prices(self, text: str) -> List[float]:
        """Extract price values from text."""
        if not text:
            return []
        
        prices = []
        for pattern in self.price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    price = float(match.replace(',', ''))
                    if 10 <= price <= 1000000:  # Reasonable price range
                        prices.append(price)
                except ValueError:
                    continue
        
        return prices
    
    def calculate_vendor_metrics(self, data: List[Dict]) -> Dict[str, Dict[str, Any]]:
        """Calculate key metrics for each vendor."""
        vendor_metrics = {}

        # Group data by channel
        channels = {}
        for item in data:
            channel = item['channel']
            if channel not in channels:
                channels[channel] = []
            channels[channel].append(item)

        for channel, channel_data in channels.items():
            if len(channel_data) == 0:
                continue

            # Basic activity metrics
            total_posts = len(channel_data)
            timestamps = [item['timestamp'] for item in channel_data]
            date_range = (max(timestamps) - min(timestamps)).days if len(timestamps) > 1 else 1
            posts_per_week = (total_posts / max(date_range, 1)) * 7 if date_range > 0 else total_posts

            # Engagement metrics (using message count as proxy for views)
            avg_views_per_post = total_posts * 10  # Simulated views

            # Business profile metrics
            price_posts = [item for item in channel_data if item['has_price']]
            avg_price = statistics.mean([item['avg_price'] for item in price_posts]) if price_posts else 0
            price_consistency = len(price_posts) / total_posts if total_posts > 0 else 0

            # Content quality metrics
            text_lengths = [len(item['text']) for item in channel_data if item['text']]
            avg_text_length = statistics.mean(text_lengths) if text_lengths else 0
            media_count = sum(1 for item in channel_data if item['media'])
            media_ratio = media_count / total_posts if total_posts > 0 else 0

            # Top performing post (longest text as proxy)
            top_post = max(channel_data, key=lambda x: len(x['text']) if x['text'] else 0)
            
            vendor_metrics[channel] = {
                'total_posts': total_posts,
                'posts_per_week': round(posts_per_week, 2),
                'avg_views_per_post': avg_views_per_post,
                'avg_price_etb': round(avg_price, 2),
                'price_consistency': round(price_consistency, 2),
                'avg_text_length': round(avg_text_length, 2),
                'media_ratio': round(media_ratio, 2),
                'date_range_days': date_range,
                'top_post': {
                    'text': top_post['text'][:100] + '...' if top_post and len(top_post['text']) > 100 else (top_post['text'] if top_post else ''),
                    'price': top_post['avg_price'] if top_post else 0,
                    'timestamp': str(top_post['timestamp']) if top_post else ''
                }
            }
        
        return vendor_metrics
    
    def calculate_lending_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate lending score based on vendor metrics."""
        # Scoring weights (can be adjusted based on business requirements)
        weights = {
            'activity': 0.25,      # Posting frequency
            'engagement': 0.30,    # Average views
            'business_profile': 0.25,  # Price consistency and range
            'content_quality': 0.20   # Media usage and text quality
        }
        
        # Normalize metrics to 0-100 scale
        activity_score = min(metrics['posts_per_week'] * 10, 100)  # Max 10 posts/week = 100
        engagement_score = min(metrics['avg_views_per_post'] / 10, 100)  # Normalize views
        
        # Business profile score
        price_score = min(metrics['price_consistency'] * 100, 100)
        avg_price_score = 50 if metrics['avg_price_etb'] == 0 else min(metrics['avg_price_etb'] / 100, 100)
        business_score = (price_score + avg_price_score) / 2
        
        # Content quality score
        text_quality = min(metrics['avg_text_length'] / 10, 100)  # Normalize text length
        media_score = metrics['media_ratio'] * 100
        content_score = (text_quality + media_score) / 2
        
        # Calculate weighted final score
        final_score = (
            activity_score * weights['activity'] +
            engagement_score * weights['engagement'] +
            business_score * weights['business_profile'] +
            content_score * weights['content_quality']
        )
        
        return round(final_score, 2)
    
    def create_vendor_scorecard(self, data: List[Dict]) -> List[Dict]:
        """Create comprehensive vendor scorecard."""
        vendor_metrics = self.calculate_vendor_metrics(data)

        scorecard_data = []
        for vendor, metrics in vendor_metrics.items():
            lending_score = self.calculate_lending_score(metrics)

            scorecard_data.append({
                'Vendor': vendor,
                'Avg_Views_Per_Post': metrics['avg_views_per_post'],
                'Posts_Per_Week': metrics['posts_per_week'],
                'Avg_Price_ETB': metrics['avg_price_etb'],
                'Price_Consistency': metrics['price_consistency'],
                'Media_Ratio': metrics['media_ratio'],
                'Lending_Score': lending_score,
                'Total_Posts': metrics['total_posts'],
                'Date_Range_Days': metrics['date_range_days']
            })

        # Sort by lending score
        scorecard_data.sort(key=lambda x: x['Lending_Score'], reverse=True)

        return scorecard_data
    
    def create_visualizations(self, scorecard_data: List[Dict], output_dir: str = "vendor_analysis"):
        """Create visualization plots for vendor analysis."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        try:
            import matplotlib.pyplot as plt
            # Set style
            plt.style.use('default')
            
            # Extract data for plotting
            vendors = [item['Vendor'] for item in scorecard_data]
            lending_scores = [item['Lending_Score'] for item in scorecard_data]
            posts_per_week = [item['Posts_Per_Week'] for item in scorecard_data]
            avg_prices = [item['Avg_Price_ETB'] for item in scorecard_data]

            # 1. Lending Score Distribution
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 2, 1)
            plt.bar(vendors, lending_scores, color='skyblue')
            plt.title('Vendor Lending Scores')
            plt.xlabel('Vendor')
            plt.ylabel('Lending Score')
            plt.xticks(rotation=45, ha='right')

            # 2. Posts per Week vs Avg Price
            plt.subplot(2, 2, 2)
            plt.scatter(posts_per_week, avg_prices,
                       s=[score*2 for score in lending_scores], alpha=0.7, color='coral')
            plt.xlabel('Posts Per Week')
            plt.ylabel('Average Price (ETB)')
            plt.title('Activity vs Price Point')

            # Add vendor labels
            for i, vendor in enumerate(vendors):
                plt.annotate(vendor,
                           (posts_per_week[i], avg_prices[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # 3. Simple metrics comparison
            plt.subplot(2, 2, 3)
            plt.bar(vendors[:5], [item['Price_Consistency'] for item in scorecard_data[:5]], color='lightblue')
            plt.title('Price Consistency (Top 5)')
            plt.xlabel('Vendor')
            plt.ylabel('Price Consistency')
            plt.xticks(rotation=45, ha='right')

            # 4. Top Performers
            plt.subplot(2, 2, 4)
            top_5_vendors = vendors[:5]
            top_5_scores = lending_scores[:5]
            plt.barh(top_5_vendors, top_5_scores, color='lightgreen')
            plt.title('Top 5 Vendors by Lending Score')
            plt.xlabel('Lending Score')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/vendor_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Visualizations saved to {output_dir}")
            
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")

    def generate_report(self, scorecard_data: List[Dict], output_file: str = "vendor_scorecard_report.json"):
        """Generate comprehensive vendor report."""
        lending_scores = [item['Lending_Score'] for item in scorecard_data]

        report = {
            'analysis_date': datetime.now().isoformat(),
            'total_vendors': len(scorecard_data),
            'top_vendor': {
                'name': scorecard_data[0]['Vendor'] if scorecard_data else 'None',
                'score': scorecard_data[0]['Lending_Score'] if scorecard_data else 0
            },
            'average_score': statistics.mean(lending_scores) if lending_scores else 0,
            'score_distribution': {
                'high_performers': len([s for s in lending_scores if s >= 70]),
                'medium_performers': len([s for s in lending_scores if 40 <= s < 70]),
                'low_performers': len([s for s in lending_scores if s < 40])
            },
            'vendor_rankings': scorecard_data
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Vendor report saved to {output_file}")
        return report


def main():
    """Main vendor analysis function."""
    print("Starting Vendor Scorecard Analysis...")
    
    analytics = VendorAnalytics()
    
    try:
        # Load data
        print("Loading Telegram data...")
        data = analytics.load_telegram_data()
        unique_channels = set(item['channel'] for item in data)
        print(f"Loaded {len(data)} messages from {len(unique_channels)} vendors")

        # Create scorecard
        print("Calculating vendor metrics...")
        scorecard_data = analytics.create_vendor_scorecard(data)

        # Save scorecard to CSV
        with open("vendor_scorecard.csv", 'w', newline='', encoding='utf-8') as f:
            if scorecard_data:
                writer = csv.DictWriter(f, fieldnames=scorecard_data[0].keys())
                writer.writeheader()
                writer.writerows(scorecard_data)
        print("Vendor scorecard saved to vendor_scorecard.csv")

        # Print summary
        print("\n" + "="*80)
        print("VENDOR SCORECARD SUMMARY")
        print("="*80)
        print(f"{'Vendor':<25} {'Avg Views/Post':<15} {'Posts/Week':<12} {'Avg Price (ETB)':<15} {'Lending Score':<12}")
        print("-" * 80)

        for row in scorecard_data:
            print(f"{row['Vendor']:<25} {row['Avg_Views_Per_Post']:<15.0f} {row['Posts_Per_Week']:<12.1f} "
                  f"{row['Avg_Price_ETB']:<15.1f} {row['Lending_Score']:<12.1f}")

        # Generate visualizations
        print("\nCreating visualizations...")
        analytics.create_visualizations(scorecard_data)

        # Generate report
        print("Generating comprehensive report...")
        report = analytics.generate_report(scorecard_data)

        print(f"\nAnalysis completed! Top vendor: {report['top_vendor']['name']} (Score: {report['top_vendor']['score']})")
        
    except FileNotFoundError:
        print("Telegram data file not found. Please ensure raw_telegram_data.jsonl exists.")
    except Exception as e:
        print(f"Error in vendor analysis: {str(e)}")


if __name__ == "__main__":
    main()
