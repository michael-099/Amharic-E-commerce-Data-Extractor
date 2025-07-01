"""
NER Model Fine-tuning for Amharic E-commerce Data.
This module handles training and evaluation of NER models.
"""

import os
import json
import torch
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

# Try to import transformers components
try:
    from transformers import (
        AutoTokenizer, AutoModelForTokenClassification,
        TrainingArguments, Trainer, DataCollatorForTokenClassification,
        EarlyStoppingCallback
    )
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Please install: pip install transformers datasets torch")


@dataclass
class NERConfig:
    """Configuration for NER training."""
    model_name: str = "xlm-roberta-base"
    max_length: int = 128
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    output_dir: str = "models"
    

class AmharicNERTrainer:
    """Trainer for Amharic NER models."""
    
    def __init__(self, config: NERConfig):
        self.config = config
        self.label2id = {}
        self.id2label = {}
        self.tokenizer = None
        self.model = None
        
    def load_conll_data(self, file_path: str) -> List[List[Tuple[str, str]]]:
        """Load CoNLL format data."""
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
    
    def create_label_mappings(self, sentences: List[List[Tuple[str, str]]]):
        """Create label to ID mappings."""
        labels = set()
        for sentence in sentences:
            for _, label in sentence:
                labels.add(label)
        
        labels = sorted(list(labels))
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        
        print(f"Found {len(labels)} unique labels: {labels}")
    
    def prepare_dataset(self, sentences: List[List[Tuple[str, str]]]) -> Dataset:
        """Prepare dataset for training."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")
            
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Prepare data
        tokenized_inputs = []
        labels = []
        
        for sentence in sentences:
            tokens = [token for token, _ in sentence]
            sentence_labels = [label for _, label in sentence]
            
            # Tokenize
            encoding = self.tokenizer(
                tokens,
                is_split_into_words=True,
                padding='max_length',
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            # Align labels with tokenized input
            word_ids = encoding.word_ids()
            aligned_labels = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    aligned_labels.append(-100)  # Special token
                elif word_idx != previous_word_idx:
                    if word_idx < len(sentence_labels):
                        aligned_labels.append(self.label2id[sentence_labels[word_idx]])
                    else:
                        aligned_labels.append(-100)
                else:
                    aligned_labels.append(-100)  # Subword token
                previous_word_idx = word_idx
            
            tokenized_inputs.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(aligned_labels)
            })
        
        # Convert to dataset
        dataset_dict = {
            'input_ids': [item['input_ids'] for item in tokenized_inputs],
            'attention_mask': [item['attention_mask'] for item in tokenized_inputs],
            'labels': [item['labels'] for item in tokenized_inputs]
        }
        
        return Dataset.from_dict(dataset_dict)
    
    def train_model(self, train_dataset: Dataset, eval_dataset: Dataset = None):
        """Train the NER model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")
            
        # Initialize model
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_f1" if eval_dataset else None,
            greater_is_better=True,
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics if eval_dataset else None,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if eval_dataset else None,
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        print(f"Model saved to {self.config.output_dir}")
        
        return trainer
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Flatten for sklearn metrics
        flat_true_labels = [label for sublist in true_labels for label in sublist]
        flat_predictions = [pred for sublist in true_predictions for pred in sublist]
        
        # Calculate F1 score
        f1 = f1_score(flat_true_labels, flat_predictions, average='weighted')
        
        return {"f1": f1}
    
    def save_config(self):
        """Save training configuration."""
        config_dict = {
            'model_name': self.config.model_name,
            'max_length': self.config.max_length,
            'learning_rate': self.config.learning_rate,
            'batch_size': self.config.batch_size,
            'num_epochs': self.config.num_epochs,
            'label2id': self.label2id,
            'id2label': self.id2label
        }
        
        with open(f"{self.config.output_dir}/config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)


def train_single_model(model_name: str, data_file: str = "Data/merged_labeled_data.txt"):
    """Train a single NER model."""
    print(f"\n=== Training {model_name} ===")
    
    # Configuration
    config = NERConfig(
        model_name=model_name,
        output_dir=f"models/{model_name.replace('/', '_')}"
    )
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = AmharicNERTrainer(config)
    
    # Load data
    print("Loading data...")
    sentences = trainer.load_conll_data(data_file)
    print(f"Loaded {len(sentences)} sentences")
    
    # Create label mappings
    trainer.create_label_mappings(sentences)
    
    # Split data
    train_sentences, eval_sentences = train_test_split(
        sentences, test_size=0.2, random_state=42
    )
    
    print(f"Train: {len(train_sentences)}, Eval: {len(eval_sentences)}")
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset = trainer.prepare_dataset(train_sentences)
    eval_dataset = trainer.prepare_dataset(eval_sentences)
    
    # Train model
    model_trainer = trainer.train_model(train_dataset, eval_dataset)
    
    # Save configuration
    trainer.save_config()
    
    print(f"Training completed for {model_name}")
    
    return trainer, model_trainer


def main():
    """Main training function."""
    if not TRANSFORMERS_AVAILABLE:
        print("Please install required packages:")
        print("pip install transformers datasets torch")
        return
    
    # Models to train
    models = [
        "xlm-roberta-base",
        "bert-base-multilingual-cased",
        "distilbert-base-multilingual-cased"
    ]
    
    results = {}
    
    for model_name in models:
        try:
            trainer, model_trainer = train_single_model(model_name)
            results[model_name] = "Success"
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            results[model_name] = f"Error: {str(e)}"
    
    print("\n=== Training Summary ===")
    for model, result in results.items():
        print(f"{model}: {result}")


if __name__ == "__main__":
    main()
