import os
import logging
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
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

def load_data(url: str) -> pd.DataFrame:
    """Load dataset from a URL."""
    try:
        data = pd.read_csv(url)
        logging.info(f"Data loaded successfully from {url}")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {url}: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset by dropping unnecessary columns and filtering sentiments."""
    try:
        df = df.drop(columns=['tweet_id'])
        logging.info("Dropped 'tweet_id' column.")
        final_df = df[df['sentiment'].isin(['neutral', 'sadness'])]
        final_df['sentiment'].replace({'neutral': 1, 'sadness': 0}, inplace=True)
        logging.info("Filtered and transformed 'sentiment' column.")
        return final_df
    except KeyError as e:
        logging.error(f"Missing required column in data: {e}")
        raise
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

def split_data(
    df: pd.DataFrame, test_size: float, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into training and testing sets."""
    try:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        logging.info(f"Data split into train ({len(train_data)}) and test ({len(test_data)}) sets.")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error during data splitting: {e}")
        raise

def save_data(data: pd.DataFrame, path: str) -> None:
    """Save the data to a CSV file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data.to_csv(path, index=False)
        logging.info(f"Data saved to {path}")
    except Exception as e:
        logging.error(f"Error saving data to {path}: {e}")
        raise

def main():
    """Main function to orchestrate data processing."""
    config_path = 'params.yaml'
    data_url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
    raw_data_dir = os.path.join("data", "raw")
    train_file = os.path.join(raw_data_dir, "train.csv")
    test_file = os.path.join(raw_data_dir, "test.csv")
    
    try:
        # Load configuration
        config = load_config(config_path)
        test_size = config['data_ingestion']['test_size']
        
        # Load and process data
        df = load_data(data_url)
        final_df = preprocess_data(df)
        train_data, test_data = split_data(final_df, test_size)
        
        # Save processed data
        save_data(train_data, train_file)
        save_data(test_data, test_file)
        
        logging.info("Data processing completed successfully.")
    except Exception as e:
        logging.error(f"Processing failed: {e}")

if __name__ == "__main__":
    main()