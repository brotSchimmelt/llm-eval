import pytest

from src.utils import extract_numeric_value, get_model_response


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
