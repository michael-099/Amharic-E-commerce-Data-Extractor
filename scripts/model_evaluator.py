"""
Model Evaluation and Comparison for Amharic NER.
This module handles evaluation and comparison of different NER models.
"""

import os
import json
import time
import torch
import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import transformers components
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Please install: pip install transformers torch")


class ModelEvaluator:
    """Evaluator for comparing NER models."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def load_model(self, model_name: str, model_path: str):
        """Load a trained model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")
            
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForTokenClassification.from_pretrained(model_path)
            
            # Create pipeline
            ner_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple"
            )
            
            self.models[model_name] = {
                'pipeline': ner_pipeline,
                'tokenizer': tokenizer,
                'model': model,
                'path': model_path
            }
            
            print(f"Loaded model: {model_name}")
            
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
    
    def load_test_data(self, file_path: str) -> List[List[Tuple[str, str]]]:
        """Load test data in CoNLL format."""
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
    
    def evaluate_model(self, model_name: str, test_sentences: List[List[Tuple[str, str]]]) -> Dict[str, Any]:
        """Evaluate a single model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        pipeline = self.models[model_name]['pipeline']
        
        all_true_labels = []
        all_pred_labels = []
        inference_times = []
        
        print(f"Evaluating {model_name}...")
        
        for sentence in test_sentences[:100]:  # Limit for faster evaluation
            tokens = [token for token, _ in sentence]
            true_labels = [label for _, label in sentence]
            
            # Join tokens for pipeline input
            text = ' '.join(tokens)
            
            # Measure inference time
            start_time = time.time()
            try:
                predictions = pipeline(text)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Align predictions with true labels
                pred_labels = ['O'] * len(tokens)
                
                for pred in predictions:
                    # Simple alignment based on text position
                    entity_text = pred['word'].replace('▁', '').strip()
                    entity_label = pred['entity_group']
                    
                    # Find matching tokens
                    for i, token in enumerate(tokens):
                        if entity_text in token or token in entity_text:
                            pred_labels[i] = f"B-{entity_label}"
                            break
                
                all_true_labels.extend(true_labels)
                all_pred_labels.extend(pred_labels)
                
            except Exception as e:
                print(f"Error processing sentence: {str(e)}")
                # Add O labels for failed predictions
                all_true_labels.extend(true_labels)
                all_pred_labels.extend(['O'] * len(tokens))
                inference_times.append(0.1)  # Default time
        
        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_true_labels, all_pred_labels, average='weighted', zero_division=0
        )
        
        # Entity-level metrics
        entity_report = classification_report(
            all_true_labels, all_pred_labels, output_dict=True, zero_division=0
        )
        
        results = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_inference_time': np.mean(inference_times),
            'total_inference_time': np.sum(inference_times),
            'entity_report': entity_report,
            'num_sentences': len(test_sentences[:100])
        }
        
        self.results[model_name] = results
        return results
    
    def compare_models(self, test_data_path: str):
        """Compare all loaded models."""
        print("Loading test data...")
        test_sentences = self.load_test_data(test_data_path)
        print(f"Loaded {len(test_sentences)} test sentences")
        
        # Evaluate each model
        for model_name in self.models.keys():
            self.evaluate_model(model_name, test_sentences)
        
        # Print comparison
        self.print_comparison()
        
        # Save results
        self.save_results()
        
        # Create visualizations
        self.create_visualizations()
    
    def print_comparison(self):
        """Print model comparison results."""
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        
        print(f"{'Model':<30} {'F1 Score':<10} {'Precision':<10} {'Recall':<10} {'Avg Time (s)':<12}")
        print("-" * 80)
        
        for model_name, results in self.results.items():
            print(f"{model_name:<30} {results['f1_score']:<10.3f} {results['precision']:<10.3f} "
                  f"{results['recall']:<10.3f} {results['avg_inference_time']:<12.4f}")
        
        # Find best model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        print(f"\nBest performing model: {best_model} (F1: {self.results[best_model]['f1_score']:.3f})")
    
    def save_results(self, output_file: str = "model_comparison_results.json"):
        """Save comparison results to JSON."""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to {output_file}")
    
    def create_visualizations(self):
        """Create comparison visualizations."""
        try:
            # Prepare data for plotting
            models = list(self.results.keys())
            f1_scores = [self.results[model]['f1_score'] for model in models]
            precision_scores = [self.results[model]['precision'] for model in models]
            recall_scores = [self.results[model]['recall'] for model in models]
            inference_times = [self.results[model]['avg_inference_time'] for model in models]
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # F1 Score comparison
            ax1.bar(models, f1_scores, color='skyblue')
            ax1.set_title('F1 Score Comparison')
            ax1.set_ylabel('F1 Score')
            ax1.tick_params(axis='x', rotation=45)
            
            # Precision vs Recall
            ax2.scatter(precision_scores, recall_scores, s=100, alpha=0.7)
            for i, model in enumerate(models):
                ax2.annotate(model, (precision_scores[i], recall_scores[i]), 
                           xytext=(5, 5), textcoords='offset points')
            ax2.set_xlabel('Precision')
            ax2.set_ylabel('Recall')
            ax2.set_title('Precision vs Recall')
            
            # Inference Time comparison
            ax3.bar(models, inference_times, color='lightcoral')
            ax3.set_title('Average Inference Time')
            ax3.set_ylabel('Time (seconds)')
            ax3.tick_params(axis='x', rotation=45)
            
            # Combined metrics
            metrics_data = np.array([f1_scores, precision_scores, recall_scores]).T
            im = ax4.imshow(metrics_data, cmap='YlOrRd', aspect='auto')
            ax4.set_xticks(range(3))
            ax4.set_xticklabels(['F1', 'Precision', 'Recall'])
            ax4.set_yticks(range(len(models)))
            ax4.set_yticklabels(models)
            ax4.set_title('Metrics Heatmap')
            
            # Add colorbar
            plt.colorbar(im, ax=ax4)
            
            plt.tight_layout()
            plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
            print("Visualization saved as model_comparison.png")
            
        except ImportError:
            print("Matplotlib not available. Skipping visualizations.")
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")


def demo_evaluation():
    """Demo evaluation with sample data."""
    print("Running demo evaluation...")
    
    # Create sample test data
    sample_sentences = [
        [("ዋጋ", "O"), ("500", "B-PRICE"), ("ብር", "I-PRICE"), ("ሻሚዝ", "B-PRODUCT"), ("በቦሌ", "B-LOC")],
        [("ላፕቶፕ", "B-PRODUCT"), ("ዋጋ", "O"), ("25000", "B-PRICE"), ("ብር", "I-PRICE")],
        [("አዲስ", "B-LOC"), ("አበባ", "I-LOC"), ("መገናኛ", "B-LOC"), ("ሞል", "I-LOC")]
    ]
    
    # Save sample data
    with open("sample_test_data.txt", 'w', encoding='utf-8') as f:
        for sentence in sample_sentences:
            for token, label in sentence:
                f.write(f"{token} {label}\n")
            f.write("\n")
    
    evaluator = ModelEvaluator()
    
    # Check for trained models
    model_dirs = ["models/xlm-roberta-base", "models/bert-base-multilingual-cased"]
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            model_name = os.path.basename(model_dir)
            evaluator.load_model(model_name, model_dir)
    
    if evaluator.models:
        evaluator.compare_models("sample_test_data.txt")
    else:
        print("No trained models found. Please train models first using ner_trainer.py")


def main():
    """Main evaluation function."""
    if not TRANSFORMERS_AVAILABLE:
        print("Please install required packages:")
        print("pip install transformers torch matplotlib seaborn")
        return
    
    evaluator = ModelEvaluator()
    
    # Look for trained models
    models_dir = "models"
    if os.path.exists(models_dir):
        for model_name in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_name)
            if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
                evaluator.load_model(model_name, model_path)
    
    if evaluator.models:
        # Use test data if available, otherwise create sample
        test_file = "Data/merged_labeled_data.txt"
        if not os.path.exists(test_file):
            print("Test data not found. Running demo evaluation...")
            demo_evaluation()
        else:
            evaluator.compare_models(test_file)
    else:
        print("No trained models found. Running demo evaluation...")
        demo_evaluation()


if __name__ == "__main__":
    main()
