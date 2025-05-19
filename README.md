# Phishing URL Detector

A machine learning model to detect phishing URLs using logistic regression.

## Features
- Uses TF-IDF vectorization of URLs
- Trained with Logistic Regression
- Detects types like:
  - Phishing
  - Malware
  - Defacement
  - Benign

## Files
- `step_1.py` - Main model training and testing script
- `requirements.txt` - List of Python packages to install
- `.gitignore` - Prevents unwanted files like `.pkl` and `.csv` from uploading
- `README.md` - Project description

## How to Run
Make sure you have Python installed.

```bash
pip install -r requirements.txt
python step_1.py
