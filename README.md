# Amharic E-commerce Data Extractor

## Project Overview

This project implements a comprehensive Named Entity Recognition (NER) system for extracting key business entities (Product, Price, Location) from Amharic text in Ethiopian e-commerce Telegram channels. The system is designed to support EthioMart's vision of becoming a centralized hub for Telegram-based e-commerce activities in Ethiopia.

## Business Objectives

1. **Data Ingestion**: Automated collection from multiple Ethiopian e-commerce Telegram channels
2. **Entity Extraction**: High-accuracy NER for Product, Price, and Location entities in Amharic text
3. **Model Comparison**: Systematic evaluation of multiple transformer-based models
4. **Interpretability**: SHAP and LIME explanations for model predictions
5. **Vendor Analytics**: Micro-lending scorecard system for vendor assessment

## Project Structure

```
├── Data/                           # Data files and datasets
│   ├── labeled_telegram_product_price_location.txt  # Original labeled data
│   ├── merged_labeled_data.txt     # Combined training dataset
│   └── channels_to_crawl.xlsx      # Channel information
├── scripts/                        # Core implementation scripts
│   ├── scraper.py                  # Telegram data scraping
│   ├── data_processor.py           # Enhanced data preprocessing
│   ├── conll_labeler.py            # CoNLL format data labeling
│   ├── ner_trainer.py              # Model fine-tuning
│   ├── model_evaluator.py          # Model comparison framework
│   ├── model_interpretability.py   # SHAP/LIME explanations
│   └── vendor_scorecard.py         # Vendor analytics engine
├── models/                         # Trained model artifacts
├── media/                          # Downloaded images from channels
├── notebooks/                      # Jupyter notebooks for analysis
└── requirements.txt                # Project dependencies
```

## Key Features

### 1. Data Collection and Preprocessing
- **Telegram Scraper**: Automated data collection from 6+ Ethiopian e-commerce channels
- **Text Normalization**: Amharic-specific preprocessing with emoji removal and text cleaning
- **Entity Extraction**: Pattern-based extraction of prices, products, and locations

### 2. NER Model Training
- **Multi-model Support**: XLM-Roberta, mBERT, DistilBERT fine-tuning
- **CoNLL Format**: Standardized data labeling with B-I-O tagging scheme
- **Evaluation Metrics**: F1-score, precision, recall with entity-level analysis

### 3. Model Interpretability
- **SHAP Integration**: Feature importance analysis for model predictions
- **LIME Explanations**: Local interpretable explanations for individual predictions
- **Difficult Case Analysis**: Identification and analysis of low-confidence predictions

### 4. Vendor Scorecard System
- **Activity Metrics**: Posting frequency and consistency analysis
- **Engagement Analysis**: View counts and interaction metrics
- **Business Profiling**: Price point analysis and product consistency
- **Lending Score**: Weighted scoring system for micro-lending assessment

## Installation and Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd "Building an Amharic E-commerce Data Extractor"
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up Telegram API credentials**:
Create a `.env` file with:
```
api_id=your_telegram_api_id
api_hash=your_telegram_api_hash
phone=your_phone_number
```

## Usage

### 1. Data Collection
```bash
python scripts/scraper.py
```

### 2. Data Preprocessing and Labeling
```bash
python scripts/data_processor.py
python scripts/conll_labeler.py
```

### 3. Model Training
```bash
python scripts/ner_trainer.py
```

### 4. Model Evaluation
```bash
python scripts/model_evaluator.py
```

### 5. Model Interpretability
```bash
python scripts/model_interpretability.py
```

### 6. Vendor Analysis
```bash
python scripts/vendor_scorecard.py
```

## Results Summary

### Data Collection
- **Total Messages**: 1,200+ messages collected
- **Channels Analyzed**: 6 Ethiopian e-commerce channels
- **Labeled Dataset**: 3,216 sentences with 174,695 tokens
- **Entity Distribution**:
  - Products: 14,399 entities
  - Prices: 8,204 entities
  - Locations: 2,920 entities

### Vendor Scorecard Results
| Vendor | Avg Views/Post | Posts/Week | Avg Price (ETB) | Lending Score |
|--------|----------------|------------|-----------------|---------------|
| sinayelj | 2,000 | 73.7 | 6,350 | 79.0 |
| ZemenExpress | 2,000 | 38.9 | 843 | 74.5 |
| Leyueqa | 2,000 | 36.8 | 2,226 | 73.4 |
| ethio_brand_collection | 2,000 | 8.9 | 0 | 70.3 |
| nevacomputer | 2,000 | 3.5 | 0 | 60.9 |
| meneshayeofficial | 2,000 | 4.3 | 6,671 | 58.6 |

### Model Performance
The system supports training and evaluation of multiple transformer models:
- **XLM-Roberta**: Multilingual model optimized for cross-lingual tasks
- **mBERT**: Multilingual BERT for diverse language support
- **DistilBERT**: Lightweight model for efficient inference

## Technical Implementation

### Entity Types
- **B-PRODUCT/I-PRODUCT**: Product names and types
- **B-PRICE/I-PRICE**: Monetary values and pricing information
- **B-LOC/I-LOC**: Location mentions (cities, areas, addresses)

### Scoring Algorithm
The vendor lending score combines:
- **Activity Score (25%)**: Posting frequency and consistency
- **Engagement Score (30%)**: Average views and interactions
- **Business Profile (25%)**: Price consistency and range
- **Content Quality (20%)**: Media usage and text quality

## Future Enhancements

1. **Real-time Processing**: Stream processing for live data ingestion
2. **Multi-modal Analysis**: Image-based product recognition
3. **Advanced NER**: Custom Amharic language models
4. **API Integration**: RESTful API for external system integration
5. **Dashboard**: Web-based analytics dashboard

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

## License

This project is developed for educational and research purposes as part of the 10 Academy AI Mastery Program.

## Contact

For questions and support, please contact the development team through the 10 Academy platform.

---

**Note**: This implementation provides a foundation for Amharic NER and vendor analytics. For production deployment, additional considerations for scalability, security, and performance optimization are recommended.