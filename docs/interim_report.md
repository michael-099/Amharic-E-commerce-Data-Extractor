# Interim Report: Amharic E-commerce Data Extractor

**Project**: Building an Amharic E-commerce Data Extractor  
**Team**: 10 Academy AI Mastery Program  
**Date**: July 1, 2025  
**Status**: Interim Submission

## Executive Summary

This interim report presents the progress made on developing an Amharic Named Entity Recognition (NER) system for e-commerce data extraction from Ethiopian Telegram channels. The project aims to support EthioMart's vision of creating a centralized platform for Telegram-based e-commerce activities in Ethiopia.

## Project Objectives Completed

### ✅ Task 1: Data Ingestion and Preprocessing
**Status**: Complete

**Achievements**:
- Successfully implemented Telegram scraper for 6 Ethiopian e-commerce channels
- Collected 1,200+ messages with associated metadata and media files
- Developed Amharic-specific text preprocessing pipeline
- Created structured data format for further analysis

**Technical Implementation**:
- **Channels Scraped**: @ZemenExpress, @nevacomputer, @meneshayeofficial, @ethio_brand_collection, @Leyueqa, @sinayelj
- **Data Volume**: 1,200 messages, 500+ media files
- **Preprocessing Features**: Emoji removal, text normalization, Amharic character filtering

### ✅ Task 2: CoNLL Format Data Labeling
**Status**: Complete

**Achievements**:
- Created comprehensive labeled dataset in CoNLL format
- Merged existing labeled data with newly processed messages
- Implemented automated labeling pipeline with pattern matching
- Validated data quality and format consistency

**Dataset Statistics**:
- **Total Sentences**: 3,216
- **Total Tokens**: 174,695
- **Entity Distribution**:
  - Products: 14,399 entities
  - Prices: 8,204 entities
  - Locations: 2,920 entities

**Entity Types Implemented**:
- `B-PRODUCT/I-PRODUCT`: Product names and types
- `B-PRICE/I-PRICE`: Monetary values and pricing information
- `B-LOC/I-LOC`: Location mentions (cities, areas, addresses)

## Technical Architecture

### Data Processing Pipeline
1. **Raw Data Collection**: Telegram API integration for message scraping
2. **Text Preprocessing**: Amharic-specific normalization and cleaning
3. **Entity Pattern Matching**: Rule-based extraction for price and location patterns
4. **CoNLL Format Generation**: Standardized labeling for NER training
5. **Data Validation**: Quality checks and format verification

### Infrastructure Components
- **Scraper Module**: `scripts/scraper.py` - Telegram data collection
- **Preprocessor**: `scripts/data_processor.py` - Enhanced text processing
- **Labeler**: `scripts/conll_labeler.py` - CoNLL format data creation
- **Data Storage**: JSONL format for raw data, TXT format for labeled data

## Data Summary

### Channel Analysis
| Channel | Messages | Media Files | Avg Text Length | Price Mentions |
|---------|----------|-------------|-----------------|----------------|
| sinayelj | 200 | 196 | 45 chars | 8 |
| ZemenExpress | 200 | 162 | 120 chars | 84 |
| Leyueqa | 200 | 142 | 85 chars | 68 |
| ethio_brand_collection | 200 | 198 | 35 chars | 0 |
| nevacomputer | 200 | 196 | 40 chars | 0 |
| meneshayeofficial | 200 | 96 | 55 chars | 18 |

### Data Quality Metrics
- **Amharic Content**: 677 messages contain meaningful Amharic text
- **Entity Coverage**: 42% of messages contain price information
- **Media Richness**: 81% of messages include product images
- **Text Diversity**: Wide range of product categories and price points

## Challenges and Solutions

### Challenge 1: Amharic Text Processing
**Issue**: Complex Amharic script with various Unicode representations
**Solution**: Implemented comprehensive Unicode normalization and character filtering

### Challenge 2: Entity Boundary Detection
**Issue**: Ambiguous entity boundaries in continuous Amharic text
**Solution**: Developed pattern-based approach with context-aware labeling

### Challenge 3: Data Volume and Quality
**Issue**: Balancing data quantity with labeling quality
**Solution**: Automated labeling with manual validation for critical samples

## Next Steps (Remaining Tasks)

### Task 3: NER Model Fine-tuning
- Implement training pipeline for XLM-Roberta, mBERT, and DistilBERT
- Configure hyperparameters for Amharic language optimization
- Set up evaluation framework with train/validation splits

### Task 4: Model Comparison and Selection
- Systematic evaluation of multiple model architectures
- Performance benchmarking using F1-score, precision, and recall
- Resource efficiency analysis for production deployment

### Task 5: Model Interpretability
- SHAP integration for feature importance analysis
- LIME implementation for local explanations
- Difficult case analysis and model behavior understanding

### Task 6: Vendor Scorecard Development
- Engagement metrics calculation from message metadata
- Business activity profiling based on posting patterns
- Micro-lending score algorithm development

## Technical Specifications

### Development Environment
- **Language**: Python 3.8+
- **Key Libraries**: Telethon, Transformers, Datasets, Scikit-learn
- **Data Format**: JSONL for raw data, CoNLL for labeled data
- **Storage**: Local file system with structured directory organization

### Data Schema
```json
{
  "channel": "string",
  "text": "string", 
  "timestamp": "datetime",
  "sender_id": "integer",
  "media": "string|null",
  "clean_text": "string",
  "extracted_entities": "object"
}
```

## Risk Assessment

### Technical Risks
- **Model Performance**: Amharic language complexity may impact NER accuracy
- **Data Quality**: Informal language and abbreviations in Telegram messages
- **Resource Requirements**: Large transformer models require significant computational resources

### Mitigation Strategies
- Multiple model architectures for performance comparison
- Extensive data preprocessing and validation
- Efficient training strategies and model optimization

## Conclusion

The interim phase has successfully established the foundation for Amharic NER development. The data collection and preprocessing pipeline is operational, and a substantial labeled dataset has been created. The project is well-positioned to proceed with model training and evaluation phases.

The next phase will focus on implementing and comparing multiple NER models, followed by interpretability analysis and vendor scorecard development. The comprehensive approach ensures both technical excellence and business value delivery for EthioMart's e-commerce platform.

## Appendices

### A. Channel Information
- Complete list of scraped channels with metadata
- Sample messages and entity annotations
- Data quality assessment reports

### B. Technical Documentation
- API documentation for scraper module
- Data preprocessing pipeline specifications
- CoNLL format validation results

### C. Code Repository
- GitHub repository with complete implementation
- Installation and usage instructions
- Development environment setup guide

---

**Report Prepared By**: 10 Academy AI Mastery Program Team  
**Review Date**: July 1, 2025  
**Next Milestone**: Final Submission - July 24, 2025
