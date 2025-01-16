import os
import pickle
import logging
import yaml
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
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
            logging.info("Config loaded successfully.")
            return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}.")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise

def load_train_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load training data from a CSV file."""
    try:
        data = pd.read_csv(data_path)
        X_train = data.iloc[:, :-1].values  # Features
        y_train = data.iloc[:, -1].values  # Target labels
        logging.info("Training data loaded successfully.")
        return X_train, y_train
    except FileNotFoundError:
        logging.error(f"Training data file not found at {data_path}.")
        raise
    except Exception as e:
        logging.error(f"Error loading training data: {e}")
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> GradientBoostingClassifier:
    """Train a Gradient Boosting Classifier."""
    try:
        clf = GradientBoostingClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate']
        )
        clf.fit(X_train, y_train)
        logging.info("Model training completed successfully.")
        return clf
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

def save_model(model, model_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info(f"Model saved successfully to {model_path}.")
    except Exception as e:
        logging.error(f"Error saving model to {model_path}: {e}")
        raise

def main():
    """Main function to orchestrate model building."""
    config_path = 'params.yaml'
    train_data_path = './data/features/train_features.csv'
    model_output_path = 'model.pkl'

    try:
        # Load configuration
        config = load_config(config_path)
        params = config['model_building']

        # Load training data
        X_train, y_train = load_train_data(train_data_path)

        # Train the model
        clf = train_model(X_train, y_train, params)

        # Save the model
        save_model(clf, model_output_path)

        logging.info("Model building pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Model building pipeline failed: {e}")

if __name__ == "__main__":
    main()
