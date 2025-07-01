"""
Model Interpretability for Amharic NER Models.
This module provides SHAP and LIME explanations for model predictions.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import required libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available. Install with: pip install lime")

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers torch")


class NERInterpreter:
    """Interpretability tool for NER models."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.load_model()
        
    def load_model(self):
        """Load the trained model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            self.pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple"
            )
            print(f"Model loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def predict_with_confidence(self, text: str) -> List[Dict[str, Any]]:
        """Get predictions with confidence scores."""
        try:
            predictions = self.pipeline(text)
            return predictions
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return []
    
    def explain_with_lime(self, text: str, num_features: int = 10) -> Dict[str, Any]:
        """Explain predictions using LIME."""
        if not LIME_AVAILABLE:
            print("LIME not available")
            return {}
        
        def predict_proba(texts):
            """Prediction function for LIME."""
            results = []
            for text in texts:
                predictions = self.predict_with_confidence(text)
                # Convert to probability-like scores
                if predictions:
                    max_score = max([pred['score'] for pred in predictions])
                    results.append([1 - max_score, max_score])
                else:
                    results.append([0.5, 0.5])
            return np.array(results)
        
        # Create LIME explainer
        explainer = LimeTextExplainer(class_names=['No Entity', 'Entity'])
        
        try:
            # Generate explanation
            explanation = explainer.explain_instance(
                text, 
                predict_proba, 
                num_features=num_features
            )
            
            # Extract explanation data
            lime_data = {
                'text': text,
                'explanations': explanation.as_list(),
                'score': explanation.score,
                'intercept': explanation.intercept[1] if hasattr(explanation, 'intercept') else 0
            }
            
            return lime_data
            
        except Exception as e:
            print(f"Error in LIME explanation: {str(e)}")
            return {'error': str(e)}
    
    def explain_with_shap(self, texts: List[str], max_evals: int = 100) -> Dict[str, Any]:
        """Explain predictions using SHAP."""
        if not SHAP_AVAILABLE:
            print("SHAP not available")
            return {}
        
        try:
            # Create a wrapper function for SHAP
            def model_wrapper(texts):
                results = []
                for text in texts:
                    predictions = self.predict_with_confidence(text)
                    if predictions:
                        # Get the highest confidence score
                        max_score = max([pred['score'] for pred in predictions])
                        results.append([1 - max_score, max_score])
                    else:
                        results.append([0.5, 0.5])
                return np.array(results)
            
            # Create SHAP explainer
            explainer = shap.Explainer(model_wrapper, self.tokenizer)
            
            # Generate SHAP values
            shap_values = explainer(texts[:min(len(texts), 5)])  # Limit for performance
            
            shap_data = {
                'texts': texts[:min(len(texts), 5)],
                'shap_values': shap_values.values.tolist() if hasattr(shap_values, 'values') else [],
                'base_values': shap_values.base_values.tolist() if hasattr(shap_values, 'base_values') else []
            }
            
            return shap_data
            
        except Exception as e:
            print(f"Error in SHAP explanation: {str(e)}")
            return {'error': str(e)}
    
    def analyze_difficult_cases(self, test_sentences: List[str]) -> Dict[str, Any]:
        """Analyze cases where the model struggles."""
        difficult_cases = []
        
        for text in test_sentences:
            predictions = self.predict_with_confidence(text)
            
            # Identify difficult cases
            if predictions:
                avg_confidence = np.mean([pred['score'] for pred in predictions])
                if avg_confidence < 0.7:  # Low confidence threshold
                    difficult_cases.append({
                        'text': text,
                        'predictions': predictions,
                        'avg_confidence': avg_confidence
                    })
        
        # Sort by confidence (lowest first)
        difficult_cases.sort(key=lambda x: x['avg_confidence'])
        
        return {
            'total_cases': len(test_sentences),
            'difficult_cases': difficult_cases[:10],  # Top 10 difficult cases
            'difficulty_rate': len(difficult_cases) / len(test_sentences) if test_sentences else 0
        }
    
    def create_interpretation_report(self, sample_texts: List[str], output_dir: str = "interpretability_results"):
        """Create comprehensive interpretation report."""
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            'model_path': self.model_path,
            'sample_texts': sample_texts,
            'predictions': [],
            'lime_explanations': [],
            'shap_explanations': {},
            'difficult_cases': {}
        }
        
        print("Generating predictions...")
        for text in sample_texts:
            predictions = self.predict_with_confidence(text)
            report['predictions'].append({
                'text': text,
                'predictions': predictions
            })
        
        print("Generating LIME explanations...")
        for i, text in enumerate(sample_texts[:5]):  # Limit for performance
            lime_explanation = self.explain_with_lime(text)
            report['lime_explanations'].append(lime_explanation)
            print(f"LIME explanation {i+1}/5 completed")
        
        print("Generating SHAP explanations...")
        shap_explanation = self.explain_with_shap(sample_texts[:5])
        report['shap_explanations'] = shap_explanation
        
        print("Analyzing difficult cases...")
        difficult_analysis = self.analyze_difficult_cases(sample_texts)
        report['difficult_cases'] = difficult_analysis
        
        # Save report
        with open(f"{output_dir}/interpretation_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # Create visualizations
        self.create_visualizations(report, output_dir)
        
        print(f"Interpretation report saved to {output_dir}")
        return report
    
    def create_visualizations(self, report: Dict[str, Any], output_dir: str):
        """Create visualization plots."""
        try:
            # Confidence distribution
            confidences = []
            for pred_data in report['predictions']:
                for pred in pred_data['predictions']:
                    confidences.append(pred['score'])
            
            if confidences:
                plt.figure(figsize=(10, 6))
                plt.hist(confidences, bins=20, alpha=0.7, color='skyblue')
                plt.xlabel('Confidence Score')
                plt.ylabel('Frequency')
                plt.title('Model Confidence Distribution')
                plt.savefig(f"{output_dir}/confidence_distribution.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Entity type distribution
            entity_types = {}
            for pred_data in report['predictions']:
                for pred in pred_data['predictions']:
                    entity_type = pred['entity_group']
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            if entity_types:
                plt.figure(figsize=(8, 6))
                plt.bar(entity_types.keys(), entity_types.values(), color='lightcoral')
                plt.xlabel('Entity Type')
                plt.ylabel('Count')
                plt.title('Predicted Entity Type Distribution')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/entity_distribution.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            print("Visualizations saved")
            
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")


def demo_interpretability():
    """Demo interpretability analysis."""
    # Sample Amharic texts for analysis
    sample_texts = [
        "ዋጋ 500 ብር የሆነ ሻሚዝ በቦሌ ይሸጣል",
        "አዲስ አበባ መገናኛ ሞል ውስጥ ላፕቶፕ ዋጋ 25000 ብር",
        "ቡና 150 ብር ሻይ 80 ብር በመርካቶ",
        "ሳሙና እና ሻምፖ በ 200 ብር ለቡ አድራሻ",
        "ስልክ ዋጋ 8000 ብር ቦሌ ሱቅ ውስጥ"
    ]
    
    # Look for trained models
    model_dirs = ["models/xlm-roberta-base", "models/bert-base-multilingual-cased"]
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            print(f"\nAnalyzing model: {model_dir}")
            try:
                interpreter = NERInterpreter(model_dir)
                report = interpreter.create_interpretation_report(
                    sample_texts, 
                    f"interpretability_results/{os.path.basename(model_dir)}"
                )
                
                # Print summary
                print(f"\nInterpretability Summary for {model_dir}:")
                print(f"- Analyzed {len(sample_texts)} texts")
                print(f"- Generated {len(report['lime_explanations'])} LIME explanations")
                print(f"- Difficulty rate: {report['difficult_cases']['difficulty_rate']:.2%}")
                
            except Exception as e:
                print(f"Error analyzing {model_dir}: {str(e)}")
    
    if not any(os.path.exists(d) for d in model_dirs):
        print("No trained models found. Please train models first using ner_trainer.py")


def main():
    """Main interpretability function."""
    if not TRANSFORMERS_AVAILABLE:
        print("Please install required packages:")
        print("pip install transformers torch shap lime matplotlib")
        return
    
    demo_interpretability()


if __name__ == "__main__":
    main()
