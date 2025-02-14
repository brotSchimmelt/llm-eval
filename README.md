# LLM-Eval

![Build Docker Image](https://img.shields.io/github/actions/workflow/status/brotSchimmelt/llm-eval/docker-build.yml?style=flat-square&label=Docker%20Image&color=blue)
![Tests](https://img.shields.io/github/actions/workflow/status/brotSchimmelt/llm-eval/test.yml?style=flat-square&label=Tests&color=green)

**LLM-Eval** is a Streamlit-based application for evaluating Large Language Model (LLM) pipelines against predefined or custom datasets using various metrics. It allows you to:

- Compare model outputs to ground-truth answers using **exact match**, **ROUGE**, **BLEU**, **semantic similarity**, and **LLM-based** (GPT-style) criteria.
- Use **live** models (local or remote) or **precomputed** responses.
- Effortlessly upload your own custom dataset in CSV/JSON format.
- Use local models served with Ollama or OpenAI compatible API endpoints.

![Screenshot](https://github.com/user-attachments/assets/b29a7b07-5f6e-48a4-8f16-421d2f2816ce)

## Features

- **Multiple Evaluation Methods**\
  Evaluate your model outputs using:

  - **Exact Match**: Checks if the response text matches exactly (case-insensitive).
  - **Overlap Metrics**: Uses ROUGE (ROUGE-1, ROUGE-2, ROUGE-L) and BLEU scores.
  - **Semantic Similarity**: Computes cosine similarity via SentenceTransformers.
  - **LLM Criteria**: Leverages an LLM to "judge" answers based on custom or default prompts.

- **Live vs. Precomputed Responses**

  - **Live Model**: Query a model in real-time (e.g. GPT-4o or Ollama models).
  - **Precomputed Responses**: Upload previously generated answers for offline or batch scoring.

- **Easy Integration**

  - Simple UI with Streamlit.
  - Automatic caching of embeddings and partial results.
  - Docker-ready for quick deployment.

## Project Structure

```plaintext
.
├── Dockerfile
├── README.md
├── data
│   ├── custom               # Any custom uploaded data will be stored here
│   └── predefined           # Built-in example datasets
│       ├── gsm8k_100.parquet
│       └── mathqa_100.parquet
├── example.env              # Example environment file
├── pyproject.toml
├── src
│   ├── app.py               # Streamlit application entry point
│   ├── config.py            # Default settings and prompts
│   ├── dataset_loader.py
│   ├── grading.py           # Evaluation/Scoring logic
│   ├── utils.py
```

## Requirements and Installation

- **Python** >= 3.11
- **uv** >= 0.5.26 ([How to install uv](https://docs.astral.sh/uv/))
- (Optional) Docker for container-based deployment

### Install dependencies locally (without Docker)

```bash
# clone the repository
git clone https://github.com/brotSchimmelt/llm-eval.git
cd llm-eval

# create and activate a virtual environment (recommended)
uv venv --python 3.11

# install dependencies
uv sync
source .venv/bin/activate
```

## Usage

### Local Environment

1. **Set up environment variables** (optional):\
   For example, save your OpenAI API key in an ```.env``` file.

2. **Run the Streamlit application**:

   ```bash
   streamlit run src/app.py
   ```

3. **Access the app**:\
   Open your web browser at [http://localhost:8501](http://localhost:8501).

### Docker

1. **Build or download the Docker image**:

   ```bash
   docker build -t llm-eval .
   # or
   docker pull ghcr.io/brotschimmelt/llm-eval:latest
   ```

2. **Run the container**:

   ```bash
   docker run -p 8501:8501 --env-file .env llm-eval
   ```

3. **Access the app**:\
   Open your browser at [http://localhost:8501](http://localhost:8501).

### Cloud GPU

You can also use [terraform](https://www.terraform.io) to provision a cloud GPU instance and run the app there. The repo contains a [main.tf](main.tf) file with the necessary configuration for [lambda labs](https://lambdalabs.com). Configure all variables including the API key in `terraform.tfvars`. Add an ssh key to your account and set the `ssh_key_name` variable to the key's name. Also set the `machine_name` variable to the name you want to give to your machine.

Then you can run the following commands:

```bash
terraform init

# to plan changes
terraform plan

# apply the changes and provision the instance
terraform apply -var-file="terraform.tfvars"

# to destroy the instance
terraform destroy -var-file="terraform.tfvars"
```

Access the app at your-ip:8501.

## Application Workflow

Once the app is running locally or in a container:

1. **Select a Dataset**:

   - **Sample Dataset**: Preloaded small data to test if the models run.
   - **Predefined Dataset**: Choose from built-in `.parquet` files in `data/predefined/`.
   - **Upload Custom Dataset**: Upload your own CSV/JSON with `question` and `ground_truth` columns.

2. **Choose a Pipeline Mode** in the sidebar:

   - **Live Model**: Configure and query an LLM in real-time.
   - **Precomputed Responses**: Upload a CSV/JSON containing pre-generated answers.

3. **Select Evaluation Method**:

   - *Exact Match*, *Overlap Metrics*, *Semantic Similarity*, *LLM Criteria*, or *Combined*.

4. **Run Evaluation**: Click **“Run Evaluation”** and review the scores.

## Configuration

The default configurations are found in [`config.py`](./src/config.py). Key settings include:

- Paths for **predefined** and **custom** dataset directories.
- Default model (`"gpt-4o-mini"`) and sampling parameters (`top_p`, `temperature`, etc.).
- `fallback_criteria` for auto-generating an LLM-based grading prompt if none is provided.

## Adding Your Own Datasets

1. **Custom**:
   - Place a CSV or JSON with `question` and `ground_truth` columns in `data/custom/`.
   - Or upload it via the UI.
2. **Predefined**:
   - Convert your dataset to `.parquet` format.
   - Place it in `data/predefined/`.
   - Restart the app to see your dataset listed.

## License

This project is licensed under the [MIT License](./LICENSE).
