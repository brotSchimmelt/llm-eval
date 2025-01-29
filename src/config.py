DEFAULT_SETTINGS = {
    "predefined_dataset_path": "data/predefined",
    "custom_dataset_path": "data/custom",
    "grading_model": "gpt-4o-mini",
    "available_models": ["gpt-4o-mini"],
    "system_prompt": "Only provide the final answer and no other tokens.",
    "top_p": 0.95,
    "max_tokens": 256,
    "temperature": 0.7,
    "fallback_criteria": "Is the answer correct and answers the question accurately considering the prompt?\nPrompt: {}",
}

GRADING_PROMPT = """
Evaluate this response based on the criteria below. 
Return ONLY a numerical score between 0-100.

Criteria: {criteria}
Response: {response}

Score: """
