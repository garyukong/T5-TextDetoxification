from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import pipeline
import pandas as pd
import torch
import tensorflow as tf
import numpy as np
import re
import tqdm
import sys

sys.path.append("../src/")

# Set global parameters
RANDOM_SEED = 42

# Parameters for toxicity classification
BATCH_SIZE = 32

# Setting the DEVICE to MPS (Metal-backed)
DEVICE = torch.device("mps")

# Load tokenizer and model for toxicity classification
tokenizer_toxicity = RobertaTokenizer.from_pretrained("SkolkovoInstitute/roberta_toxicity_classifier")
model_toxicity = RobertaForSequenceClassification.from_pretrained(
    "SkolkovoInstitute/roberta_toxicity_classifier"
)


def classify_toxicity(
    data,
    tokenizer=tokenizer_toxicity,
    model=model_toxicity,
    batch_size=BATCH_SIZE,
    include_tqdm=False,
    output_type="numpy",
    flatten_output=True,
):
    """
    Classifies text as toxic or neutral.

    Args:
        data (list): The list of text to classify.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        model (transformers.PreTrainedModel): The model to use.
        batch_size (int): The batch size to use.
        include_tqdm (bool): Whether to include tqdm progress bar output.
        output_type (str): The type of output to return. 'numpy', 'list', 'int'

    Returns:
        results of the classification (nparray, list or int) where 0 is neutral and 1 is toxic.
    """
    if isinstance(data, (np.ndarray, str)):
        data = data.tolist() if isinstance(data, np.ndarray) else [data]

    model = model.to(DEVICE)

    results = np.empty(len(data), dtype=int) if output_type == "numpy" else []

    for i in tqdm.tqdm(range(0, len(data), batch_size), disable=not include_tqdm):
        batch = tokenizer(data[i : i + batch_size], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            result = model(**batch)["logits"].argmax(1).data
            result = result.cpu()
        if output_type == "numpy":
            results[i : i + batch_size] = result
        else:
            results.extend(result.tolist())

    if flatten_output and len(results) == 1:
        return results[0]

    return results


def calc_toxicity_cols(df, toxic_col="en_toxic_comment", neutral_col="en_neutral_comment"):
    """
    Classifies comments as toxic or neutral

    Args:
        df (pandas.DataFrame): The dataframe with two columns of comments to classify.

    Returns:
        pandas.DataFrame: The dataframe with two new columns of toxicity classifications.
    """

    df_toxiclabels = df.copy()

    df_toxiclabels["en_toxic_label"] = df_toxiclabels[toxic_col].apply(
        lambda x: classify_toxicity(x, flatten_output=True)
    )
    df_toxiclabels["en_neutral_label"] = df_toxiclabels[neutral_col].apply(
        lambda x: classify_toxicity(x, flatten_output=True)
    )

    return df_toxiclabels


def calc_toxicity_metrics(df):
    """Calculates the number of comments classified as toxic or neutral"""
    length = len(df)
    toxic_correct = df["en_toxic_label"].sum()
    toxic_incorrect = length - toxic_correct
    neutral_incorrect = df["en_neutral_label"].sum()
    neutral_correct = length - neutral_incorrect

    print(f"Total number of comments: {length}")
    print(f"Toxic comments classified as toxic: {toxic_correct}")
    print(f"Toxic comments classified as neutral: {toxic_incorrect}")
    print(f"Neutral comments classified as neutral: {neutral_correct}")
    print(f"Neutral comments classified as toxic: {neutral_incorrect}")

    # return length, toxic_correct, toxic_incorrect, neutral_correct, neutral_incorrect


def get_unique_rows(df):
    """
    Returns a DataFrame with unique rows based on en_toxic_comment

    Args:
        df (pandas.DataFrame): The DataFrame to filter.
                               Should contain a column named en_toxic_comment, and a column named en_neutral_label.

    Returns:
        pandas.DataFrame: The filtered DataFrame.
    """
    # Only keep the first row for each unique en_toxic_comment, selecting the row with the lowest en_neutral_label (ideally 0)
    df_unique = df.sort_values(by="en_neutral_label").drop_duplicates(subset="en_toxic_comment", keep="first")

    # Reset index of the df_unique DataFrame
    df_unique.reset_index(drop=True, inplace=True)

    return df_unique


def clean_text(text):
    """
    Clean the text

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    # Convert first character to uppercase
    text = text[0].upper() + text[1:]

    # # Remove newline characters
    text = re.sub(r"\n", " ", text)

    # Remove white space after $
    text = re.sub(r"([$])\s+", r"\1", text)

    # Remove white space trailing punctuation end of sentence
    text = re.sub(r"\s+([.,!?%])", r"\1", text)

    # Remove white space before contractions (e.g., "I 'm John" becomes "I'm John")
    text = re.sub(r"\s\'(s|t|ve|ll|d|re|m)\b", r"'\1", text)

    # Remove white space around text within double quotes
    text = re.sub(r'"(\s*.*?\s*)"', r'"\1"', text)

    # Remove white space around text within single quotes
    text = re.sub(r"'(\s*.*?\s*)'", r"'\1'", text)

    # Remove white space around text within parentheses
    text = re.sub(r"\((\s*.*?\s*)\)", r"(\1)", text)

    # Remove white space around text within square brackets
    text = re.sub(r"\[(\s*.*?\s*)\]", r"[\1]", text)

    # Remove white space around text within curly brackets
    text = re.sub(r"\{(\s*.*?\s*)\}", r"{\1}", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_dataset(df):
    """
    Clean the dataset by converting the text to lowercase, removing extra whitespace,
    and removing non-alphanumeric characters except for some punctuation.

    Args:
        df (pandas.DataFrame): The DataFrame to clean. Should contain a column named en_toxic_comment.

    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """
    # Create a copy of the dataframe to avoid modifying the original dataframe
    df_clean = df.copy()

    # Clean the text in the en_toxic_comment column
    df_clean["en_toxic_comment"] = df_clean["en_toxic_comment"].apply(lambda x: clean_text(x))

    # Clean the text in the en_neutral_comment column
    df_clean["en_neutral_comment"] = df_clean["en_neutral_comment"].apply(lambda x: clean_text(x))

    return df_clean


def prepare_and_save_data(df, x_col_name, y_col_name, random_seed, x_output_path, y_output_path):
    """
    Shuffle and split the dataframe into X and y based on the given column names,
    and then save them to CSV files.

    Parameters:
    - df: The dataframe to be processed
    - x_col_name: The name of the column containing the features
    - y_col_name: The name of the column containing the labels
    - random_seed: The random seed for shuffling
    - x_output_path: The path to save the features CSV
    - y_output_path: The path to save the labels CSV
    """

    # Print the number of rows in the dataframe
    print(f"Number of rows in the dataframe: {len(df)}")

    # Shuffle the data
    df_shuffled = df.sample(frac=1, random_state=random_seed)

    # Split the data into X and y
    X = df_shuffled[x_col_name].tolist()
    y = df_shuffled[y_col_name].tolist()

    # Save as CSV files
    pd.DataFrame(X).to_csv(x_output_path, index=False, header=False)
    pd.DataFrame(y).to_csv(y_output_path, index=False, header=False)
