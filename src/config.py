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
    "sidebar_selections": ["Live Model", "Precomputed Responses"],
}

GRADING_PROMPT = """
Evaluate this response based on the criteria below. 
Return ONLY a numerical score between 0-100.

Criteria: {criteria}
Response: {response}

Score: """


DEMO_DATASET = {
    "question": [
        "What is 2+2?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms",
    ],
    "ground_truth": [
        "4",
        "Paris",
        "Quantum computing uses quantum bits to perform calculations using quantum mechanics principles",
    ],
}
