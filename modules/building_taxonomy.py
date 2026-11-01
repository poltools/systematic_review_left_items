import os
import joblib
import pandas as pd
import re

def load_joblib_responses(folder, prefix="response"):
    """
    Loads responses from a folder and returns a mapping from index to response.

    Args:
        folder (str): Folder path containing .joblib files
        prefix (str): File prefix (e.g., 'response')

    Returns:
        dict[int, str]: Mapping from DataFrame index to AI response
    """
    response_files = [
        f for f in os.listdir(folder)
        if f.startswith(prefix) and f.endswith(".joblib")
    ]

    index_pattern = re.compile(f"{prefix}_(\\d+).joblib")

    index_to_response = {}
    for fname in response_files:
        match = index_pattern.match(fname)
        if not match:
            continue
        idx = int(match.group(1))
        full_path = os.path.join(folder, fname)
        response = joblib.load(full_path)
        index_to_response[idx] = response

    return index_to_response


def merge_responses_with_dataframe(df: pd.DataFrame, response_map: dict[int, str], column_name="AI_Response"):
    """
    Merges a dict of responses into the DataFrame by index.

    Args:
        df (pd.DataFrame): Original DataFrame
        response_map (dict[int, str]): Index to response mapping
        column_name (str): New column to store responses

    Returns:
        pd.DataFrame: Updated DataFrame with response column
    """
    df = df.copy()
    df[column_name] = df.index.map(response_map)
    return df
    # Load original datasets
    
