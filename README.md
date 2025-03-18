# Sentiment Analysis Pipeline

This project performs sentiment analysis on tweets using machine learning models. The pipeline includes data loading, preprocessing, model training, evaluation, and result storage in an SQLite database.

## Project Structure
```
│── data/                          # Raw and processed data
│   ├── raw/                       # Original Excel files
│   ├── processed/                 # Preprocessed data
│── database/                      # SQLite database and schema
│── logs/                          # Logs for debugging
│   ├── pipeline.log               # Log file capturing pipeline execution
│── notebooks/                     # Jupyter notebooks for analysis
│── src/                           # Source code
│   ├── config.py                  # Configuration settings
│   ├── load_data.py               # Load tweets from Excel
│   ├── preprocess.py              # Preprocessing functions
│   ├── make_model.py              # Model training & prediction
│── scripts/                       # Scripts for execution
│   ├── run_pipeline.py            # End-to-end execution
│── requirements.txt               # Dependencies
│── README.md                      # Project documentation
```