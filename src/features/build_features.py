import os
import yaml
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(),logging.FileHandler("app.log")]
)

def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logging.info(f"Config loaded successfully from {config_path}")
            return config
    except FileNotFoundError:
        logging.error(f"Config file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise

def load_processed_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed train and test datasets."""
    try:
        train_data = pd.read_csv(train_path).fillna('')
        test_data = pd.read_csv(test_path).fillna('')
        logging.info(f"Processed data loaded successfully from {train_path} and {test_path}")
        return train_data, test_data
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading processed data: {e}")
        raise

def apply_bow(
    train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply Bag of Words (BoW) transformation."""
    try:
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        # Create DataFrames
        train_features = pd.DataFrame(X_train_bow.toarray())
        train_features['label'] = y_train

        test_features = pd.DataFrame(X_test_bow.toarray())
        test_features['label'] = y_test

        logging.info("Bag of Words (BoW) transformation applied successfully.")
        return train_features, test_features
    except KeyError as e:
        logging.error(f"Missing required column in DataFrame: {e}")
        raise
    except Exception as e:
        logging.error(f"Error applying Bag of Words (BoW): {e}")
        raise

def save_features(train_features: pd.DataFrame, test_features: pd.DataFrame, output_dir: str) -> None:
    """Save transformed features to CSV files."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_features.to_csv(os.path.join(output_dir, "train_features.csv"), index=False)
        test_features.to_csv(os.path.join(output_dir, "test_features.csv"), index=False)
        logging.info(f"Features saved successfully to {output_dir}")
    except Exception as e:
        logging.error(f"Error saving features: {e}")
        raise

def main():
    """Main function to orchestrate feature engineering."""
    config_path = 'params.yaml'
    train_path = './data/processed/train_processed.csv'
    test_path = './data/processed/test_processed.csv'
    output_dir = './data/features'

    try:
        # Load configuration
        config = load_config(config_path)
        max_features = config['feature_engineering']['max_features']

        # Load processed data
        train_data, test_data = load_processed_data(train_path, test_path)

        # Apply Bag of Words
        train_features, test_features = apply_bow(train_data, test_data, max_features)

        # Save features
        save_features(train_features, test_features, output_dir)

        logging.info("Feature engineering completed successfully.")
    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")

if __name__ == "__main__":
    main()