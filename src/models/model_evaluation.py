import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(),logging.FileHandler("app.log")]
)

def load_model(model_path: str):
    """Load the trained model from a file."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        logging.error(f"Model file not found at {model_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def load_test_data(data_path: str) -> Dict[str, Any]:
    """Load test dataset."""
    try:
        test_data = pd.read_csv(data_path)
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
        logging.info(f"Test data loaded successfully from {data_path}")
        return {'X_test': X_test, 'y_test': y_test}
    except FileNotFoundError:
        logging.error(f"Test data file not found at {data_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate the model on test data and calculate metrics."""
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        logging.info("Model evaluation completed successfully.")
        return metrics
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise

def save_metrics(metrics: Dict[str, float], output_path: str) -> None:
    """Save metrics to a JSON file."""
    try:
        with open(output_path, "w") as json_file:
            json.dump(metrics, json_file, indent=4)
        logging.info(f"Metrics saved successfully to {output_path}")
    except Exception as e:
        logging.error(f"Error saving metrics to {output_path}: {e}")
        raise

def main():
    """Main function to orchestrate model evaluation."""
    model_path = 'model.pkl'
    test_data_path = './data/features/test_features.csv'
    metrics_output_path = 'reports/metrics.json'

    try:
        # Load the model
        model = load_model(model_path)

        # Load the test data
        test_data = load_test_data(test_data_path)

        # Evaluate the model
        metrics = evaluate_model(model, test_data['X_test'], test_data['y_test'])

        # Save the evaluation metrics
        save_metrics(metrics, metrics_output_path)

        logging.info("Model evaluation pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Model evaluation pipeline failed: {e}")

if __name__ == "__main__":
    main()