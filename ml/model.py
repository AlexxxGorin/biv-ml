from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import pandas as pd
import yaml
from transformers import pipeline

# load config file
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


@dataclass
class SentimentPrediction:
    """Class representing a sentiment prediction result."""
    label: str
    score: float


def load_model():
    """Load a pre-trained sentiment analysis model.

    Returns:
        model (function): A function that takes a text input and returns a SentimentPrediction object.
    """
    model_hf = pipeline(config["task"], model=config["model"], device=-1)

    def model(
        text: Union[str, List[str]]
    ) -> Union[SentimentPrediction, List[SentimentPrediction]]:
        if isinstance(text, str):
            # Single text input
            pred = model_hf(text)
            pred_best_class = pred[0]
            return SentimentPrediction(
                label=pred_best_class["label"],
                score=pred_best_class["score"],
            )
        elif isinstance(text, list):
            # Batch input
            predictions = model_hf(text)
            return [
                SentimentPrediction(label=pred["label"], score=pred["score"])
                for pred in predictions
            ]
        else:
            raise ValueError("Input must be a string or a list of strings.")

    return model


def batch_inference(
    texts: List[str], batch_size: int = 32
) -> List[SentimentPrediction]:
    """Perform batch inference on a list of texts.

    Args:
        texts (List[str]): List of input texts.
        batch_size (int): Number of rows to process in each batch.

    Returns:
        List[SentimentPrediction]: List of predictions for the input texts.
    """
    # Load the model
    model = load_model()

    results = []

    # Process in batches
    for start_idx in range(0, len(texts), batch_size):
        batch = texts[start_idx : start_idx + batch_size]
        predictions = model(batch)
        results.extend(predictions)

    return results


def process_csv(
    input_csv: str, output_csv: str, text_column: str = "text", batch_size: int = 32
):
    """Process a CSV file to perform sentiment analysis.

    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the output CSV file with predictions.
        text_column (str): Name of the column containing text data.
        batch_size (int): Number of rows to process in each batch.

    Returns:
        None
    """
    # Load data
    df = pd.read_csv(input_csv)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV file.")

    texts = df[text_column].tolist()

    # Perform batch inference
    predictions = batch_inference(texts, batch_size=batch_size)

    # Add predictions to DataFrame
    df["label"] = [pred.label for pred in predictions]
    df["score"] = [pred.score for pred in predictions]

    # Save to output CSV
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
