import pytest

from src.utils import (
    extract_numeric_value,
    get_model_response,
    remove_thinking_sections,
)


def test_extract_numeric_value_with_float():
    assert extract_numeric_value("The price is 42.5 dollars.") == 42.5


def test_extract_numeric_value_with_integer():
    assert extract_numeric_value("There are 100 apples.") == 100.0


def test_extract_numeric_value_with_negative_float():
    assert extract_numeric_value("-34.2 is a negative float.") == -34.2


def test_extract_numeric_value_with_negative_integer():
    assert extract_numeric_value("Value: -42") == -42.0


def test_extract_numeric_value_with_no_numbers():
    assert extract_numeric_value("No numbers here!") == 0.0


def test_extract_numeric_value_with_multiple_numbers():
    assert extract_numeric_value("Here are 50 and 42.3 values.") == 50.0


def test_extract_numeric_value_with_empty_string():
    assert extract_numeric_value("") == 0.0


def test_extract_numeric_value_with_special_characters():
    assert extract_numeric_value("$%^&*() 123 @!") == 123.0


def test_extract_numeric_value_with_only_special_characters():
    assert extract_numeric_value("!@#$%^&*()") == 0.0


def test_empty_model_name():
    with pytest.raises(ValueError, match="The model_name must be a non-empty string."):
        get_model_response("", "What is AI?", "", {"temperature": 0.7})


def test_empty_prompt():
    with pytest.raises(ValueError, match="The prompt must be a non-empty string."):
        get_model_response("gpt-4", "", "", {"temperature": 0.7})


def test_remove_single_thinking_block():
    response = "<think>This is an internal thought.</think> The answer is 42."
    expected = "The answer is 42."
    assert remove_thinking_sections(response) == expected


def test_remove_multiple_thinking_blocks():
    response = "<think>Step 1: Analyzing...</think> Here is your answer. <think>Step 2: Verifying...</think>"
    expected = "Here is your answer."
    assert remove_thinking_sections(response) == expected


def test_remove_thinking_with_multiline_content():
    response = """<think>
    First, I recognize that the problem asks for 2 + 2.
    Then, I perform the calculation.
    </think>
    The final answer is 4."""
    expected = "The final answer is 4."
    assert remove_thinking_sections(response) == expected


def test_no_thinking_block():
    response = "The model output does not contain any thinking section."
    expected = response
    assert remove_thinking_sections(response) == expected


def test_only_thinking_block():
    response = "<think>Computation in progress...</think>"
    expected = ""
    assert remove_thinking_sections(response) == expected


def test_missing_closing_thinking_tag():
    response = "<think>Computation in progress..."
    expected = response
    assert remove_thinking_sections(response) == expected
