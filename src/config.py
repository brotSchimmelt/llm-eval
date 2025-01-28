DEFAULT_SETTINGS = {
    "predefined_dataset_path": "data/predefined",
    "custom_dataset_path": "data/custom",
    "grading_model": "gpt-4o-mini",
}

GRADING_PROMPT = """
Evaluate this response based on the criteria below. 
Return ONLY a numerical score between 0-100.

Criteria: {criteria}
Response: {response}

Score: """
