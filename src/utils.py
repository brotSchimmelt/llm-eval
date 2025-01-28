import re
from typing import List

import ollama


def extract_numeric_value(input_str: str) -> float:
    """
    Safely extracts a numeric value (float or int) from a string.
    If a float is found, it is returned. If no float is present but an integer is found,
    the integer is cast to a float and returned. If no numeric value is found, returns 0.0.

    Args:
        input_str (str): The input string to extract the numeric value from.

    Returns:
        float: The extracted numeric value as a float, or 0.0 if no numeric value is found.
    """
    if not isinstance(input_str, str):
        input_str = str(input_str)

    try:
        match = re.search(r"-?\d+\.\d+|-?\d+", input_str)
        if match:
            value = match.group()
            return float(value)
        return 0.0
    except Exception as e:
        print(f"Error extracting numeric value: {e}")
        return 0.0


def get_local_ollama_model_names() -> List[str]:
    """
    Retrieves the names of all locally available Ollama models.

    Returns:
        List[str]: A list of model names in the format "ollama/<model_name>".
    """
    return [f"ollama/{m.model}" for m in ollama.list().models]
