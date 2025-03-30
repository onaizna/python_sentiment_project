# Sentiment Analysis Pipeline by Lorenzo Forcucci

This project performs sentiment analysis on tweets using machine learning models. The pipeline includes data loading, preprocessing, model training, evaluation, and result storage in an SQLite database.

## Project Structure
```
│── data/                          # Raw and processed data
│   ├── raw/                       # Original Excel files
│   ├── processed/                 # Preprocessed data
│── database/                      # SQLite database and schema
│── logs/                          # Logs for debugging
│   ├── pipeline.log               # Log file capturing pipeline execution
│── models/                        # Tmporary store for Models
│   ├── random_forest.pickle       # raondm forest trained model in python serialized object format
│   ├── vetcorizer.pickle          # text vectorization model in python serialized object format
│── notebooks/                     # Jupyter notebooks for analysis
│── src/                           # Source code
│   ├── config.py                  # Configuration settings
│   ├── load_data.py               # Load tweets from Excel
│   ├── preprocess.py              # Preprocessing functions
│   ├── make_model.py              # Model training & prediction
│── scripts/                       # Scripts for execution
│   ├── run_pipeline.py            # End-to-end execution
│── requirements.txt               # Dependencies 2DO!
│── README.md                      # Project documentation
```