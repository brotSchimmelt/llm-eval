import re
import subprocess
from typing import Any, Dict, List

import json_repair
import litellm
import nltk
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


def _is_ollama_running() -> bool:
    """
    Checks if the Ollama service is running by attempting to list models.

    Returns:
        bool: True if Ollama is running, False otherwise.
    """
    try:
        subprocess.run(
            ["ollama", "list"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False  # Ollama is not running or not installed


def get_local_ollama_model_names() -> List[str]:
    """
    Retrieves the names of all locally available Ollama models.

    Returns:
        List[str]: A list of model names in the format "ollama/<model_name>".
                   If Ollama is not running, returns an empty list.
    """
    if not _is_ollama_running():
        return []

    ollama_models = [f"ollama/{m.model}" for m in ollama.list().models]
    ollama_models = [m for m in ollama_models if "embed" not in m]

    return ollama_models


def get_model_response(
    model_name: str, prompt: str, system_prompt: str, params: Dict[str, Any]
) -> str:
    """
    Generates a response from a specified language model based on the provided prompts and parameters.

    Args:
        model_name (str): The name of the model to use (e.g., "gpt-4").
        prompt (str): The user's input or query to the model.
        system_prompt (str): A system-level instruction or context for the model's behavior.
                             If empty, only the user prompt will be used.
        params (Dict[str, Any]): Additional parameters for the model, such as temperature,
                                 max_tokens, top_p, and top_k.

    Returns:
        str: The response generated by the model.

    Raises:
        ValueError: If `model_name` or `prompt` is None or an empty string.
    """
    if not model_name or not isinstance(model_name, str):
        raise ValueError("The model_name must be a non-empty string.")

    if not prompt or not isinstance(prompt, str):
        raise ValueError("The prompt must be a non-empty string.")

    if not params or not isinstance(params, dict):
        params = {}

    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    return (
        litellm.completion(model=model_name, messages=messages, **params)
        .choices[0]
        .message.content
    )


def remove_thinking_sections(response: str) -> str:
    """
    Removes the <think>...</think> sections from the model response.

    Args:
        response (str): The text generated by the model.

    Returns:
        str: The cleaned response without the <think>...</think> sections.
    """
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


def extract_params_from_user_text(user_text: str) -> Dict[str, Any]:
    """
    Extracts and repairs JSON parameters from a free-text user input.

    Args:
        user_text (str): The user-provided text containing JSON.

    Returns:
        Dict[str, Any]: A dictionary containing the parsed parameters if the input is valid.
                        Returns an empty dictionary `{}` if parsing fails or input is empty.
    """
    if not user_text:
        return {}

    params = json_repair.repair_json(user_text, return_objects=True)

    if isinstance(params, str):
        return {}

    return params


def ensure_nltk_punkt() -> None:
    """
    Ensures that the NLTK 'punkt' tokenizer is available.
    Downloads it only if it is not already installed.
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
