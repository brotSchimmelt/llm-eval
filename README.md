# LLM Evaluation Toolkit

A Streamlit-based application for evaluating Large Language Models (LLMs) across multiple providers, featuring both cloud-based and local models.

## Features

- **Multi-Model Support**: Evaluate models from OpenAI, Anthropic, Hugging Face, and local Ollama models
- **Dataset Management**:
  - Predefined datasets (Parquet format)
  - Custom dataset upload (CSV/JSON)
  - Automatic schema validation
- **Evaluation Methods**:
  - Exact match comparison
  - LLM-as-judge criteria evaluation
- **Parameter Control**:
  - System prompts
  - Temperature, Top-p, Top-k
  - Max tokens
- **Local Model Support**: Integration with Ollama for local model testing
- **Results Visualization**: Interactive tables and metrics dashboard

## Installation

### Prerequisites

- Python 3.11
- [Ollama](https://ollama.ai/) (for local models)

```bash
# Clone repository
git clone https://github.com/yourusername/llm-evaluation-toolkit.git
cd llm-evaluation-toolkit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac)
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
```

## Configuration

Edit `.env` file with your API keys:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
HUGGINGFACE_API_KEY=your_hf_key
```

For local Ollama models:

```bash
# Start Ollama service
ollama serve

# Pull desired models
ollama pull llama2
ollama pull mistral
```

## Usage

```bash
streamlit run src/app.py
```

### Application Workflow

1. **Model Configuration** (sidebar):
   - Select model provider
   - Set system prompt
   - Adjust generation parameters
2. **Dataset Selection**:
   - Use sample dataset
   - Load predefined dataset
   - Upload custom dataset
3. **Run Evaluation**:
   - Choose evaluation method
   - View results in interactive table
   - Analyze performance metrics

## Evaluation Methods

### 1. Exact Match

```python
# Example evaluation
{
  "question": "What is 2+2?",
  "ground_truth": "4"
}
```

### 2. LLM Criteria Evaluation

```python
{
  "question": "Explain quantum computing",
  "criteria": "Answer should be under 100 words and avoid technical jargon"
}
```

## Example Dataset

`sample_dataset.json`

```json
[
  {
    "question": "What is the capital of France?",
    "ground_truth": "Paris",
    "criteria": "Answer must be in English"
  },
  {
    "question": "Convert 100°F to Celsius",
    "ground_truth": "37.78",
    "criteria": "Provide numerical answer with two decimal places"
  }
]
```

## Contributing

1. Fork the repository
2. Create your feature branch:

   ```bash
   git checkout -b feature/new-feature
   ```

3. Commit your changes
4. Push to the branch
5. Open a pull request

## License

MIT License

## Acknowledgments

- [LiteLLM](https://github.com/BerriAI/litellm) for model abstraction
- [Streamlit](https://streamlit.io) for UI framework
- [Ollama](https://ollama.ai/) for local model management
