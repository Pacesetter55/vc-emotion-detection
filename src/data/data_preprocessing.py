import os
import re
import logging
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(),logging.FileHandler("app.log")]
)

def download_nltk_resources() -> None:
    """Download necessary NLTK resources."""
    try:
        nltk.download('wordnet')
        nltk.download('stopwords')
        logging.info("Downloaded NLTK resources successfully.")
    except Exception as e:
        logging.error(f"Error downloading NLTK resources: {e}")
        raise

def lemmatization(text: str) -> str:
    """Lemmatize the given text."""
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_text = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized_text)

def remove_stop_words(text: str) -> str:
    """Remove stopwords from the text."""
    stop_words = set(stopwords.words("english"))
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

def removing_numbers(text: str) -> str:
    """Remove numerical digits from the text."""
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text: str) -> str:
    """Convert all characters in the text to lowercase."""
    return text.lower()

def removing_punctuations(text: str) -> str:
    """Remove punctuation marks from the text."""
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def removing_urls(text: str) -> str:
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing steps to the content column of the DataFrame."""
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        logging.info("Text normalization complete.")
        return df
    except KeyError as e:
        logging.error(f"Missing required column in DataFrame: {e}")
        raise
    except Exception as e:
        logging.error(f"Error during text normalization: {e}")
        raise

def save_data(df: pd.DataFrame, path: str) -> None:
    """Save the DataFrame to a CSV file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logging.info(f"Data saved to {path}")
    except Exception as e:
        logging.error(f"Error saving data to {path}: {e}")
        raise

def preprocess_and_save(train_path: str, test_path: str, output_dir: str) -> None:
    """Main function to preprocess and save train and test datasets."""
    try:
        # Load datasets
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logging.info("Data loaded successfully.")

        # Normalize text
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Save processed data
        save_data(train_processed_data, os.path.join(output_dir, "train_processed.csv"))
        save_data(test_processed_data, os.path.join(output_dir, "test_processed.csv"))
        logging.info("Preprocessing and saving completed successfully.")
    except Exception as e:
        logging.error(f"Error in preprocessing and saving data: {e}")
        raise

if __name__ == "__main__":
    # Configure paths
    train_file = './data/raw/train.csv'
    test_file = './data/raw/test.csv'
    processed_data_dir = './data/processed'

    # Download NLTK resources and preprocess data
    download_nltk_resources()
    preprocess_and_save(train_file, test_file, processed_data_dir)