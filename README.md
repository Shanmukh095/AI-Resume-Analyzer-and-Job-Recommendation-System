# AI Resume Analyzer and Job Recommendation System

## Project Overview

This project aims to build an intelligent Resume Analyzer and Job Recommendation System using Transformer-based Natural Language Processing (NLP) models.

The system consists of two major components:

### 1. Resume Classification

Classify resumes into professional categories using Transformer models such as:

* DistilBERT
* BERT
* RoBERTa

### 2. Job Recommendation

Recommend relevant job postings by comparing resume embeddings with job description embeddings using:

* Sentence-BERT (SBERT)
* MPNet

---

## Project Structure

```text
AI_Resume_Analyzer_and_Job_Recommendation_System/
│
├── data/
│   ├── resume.zip
│   ├── extracted/
│   └── processed/
│
├── src/
│   ├── config.py
│   ├── main.py
│   │
│   ├── preprocessing/
│   │   ├── extract_dataset.py
│   │   ├── text_cleaner.py
│   │   ├── label_encoder.py
│   │   ├── data_splitter.py
│   │   └── preprocess_pipeline.py
│   │
│   ├── classification/
│   │
│   └── recommendation/
│
├── notebooks/
│
├── requirements.txt
└── README.md
```

---

## Dataset

### Resume Dataset

The resume dataset contains:

* 2,484 resumes
* 24 job categories
* Resume text and category labels

Example categories:

* ACCOUNTANT
* ADVOCATE
* AGRICULTURE
* ENGINEERING
* HR
* INFORMATION-TECHNOLOGY
* SALES
* TEACHER

### Job Dataset

LinkedIn job postings dataset containing:

* Job titles
* Job descriptions
* Job metadata

---

## Preprocessing Pipeline

The preprocessing pipeline performs:

1. Dataset extraction from ZIP file
2. Resume text cleaning
3. Duplicate removal
4. Label encoding
5. Stratified train-test split
6. Saving processed datasets

Generated outputs:

```text
data/processed/
├── train_resume.csv
├── test_resume.csv
└── label_encoder.pkl
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd AI_Resume_Analyzer_and_Job_Recommendation_System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Preprocessing Pipeline

Run:

```bash
python -m src.main
```

Expected output:

```text
Starting Project...
Dataset already extracted
Loading Resume Dataset...
Original Shape: (2484, 4)
After Duplicate Removal: (2482, 5)
Preprocessing Completed
Train Shape: (1985, 2)
Test Shape: (497, 2)
Pipeline Completed.
```

---

## Planned Models

### Resume Classification

* DistilBERT
* BERT
* RoBERTa

### Semantic Matching

* Sentence-BERT
* all-mpnet-base-v2

### Similarity Metric

* Cosine Similarity

---

## Future Work

* Resume classification API
* Job recommendation engine
* Web interface
* Model comparison and evaluation
* Deployment using Streamlit or Flask

---

## Authors

Developed as an NLP and Machine Learning project for Resume Classification and Job Recommendation using Transformer models.
