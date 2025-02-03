import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from config import DEFAULT_SETTINGS, DEMO_DATASET
from dataset_loader import DatasetLoader
from grading import GradingHandler
from utils import (
    ensure_nltk_punkt,
    extract_params_from_user_text,
    get_local_ollama_model_names,
    get_model_response,
    get_predefined_dataset_names,
    initialize_session_state,
    remove_thinking_sections,
)

ensure_nltk_punkt()
load_dotenv()


def configure_sidebar() -> dict:
    """Configure sidebar components and return settings"""
    st.sidebar.title("⚙️ Configuration")
    settings = {
        "mode": st.sidebar.pills(
            "Pipeline Type",
            options=DEFAULT_SETTINGS["sidebar_selections"],
            # index=0,
            selection_mode="single",
            default=DEFAULT_SETTINGS["sidebar_selections"][0],
        )
    }

    if settings["mode"] == "Live Model":
        settings.update(_configure_live_model_settings())
    else:
        settings.update(_configure_precomputed_settings())

    settings["eval_method"] = st.sidebar.radio(
        "Evaluation Method",
        options=[
            "Exact Match",
            "Overlap Metrics",
            "Semantic Similarity",
            "LLM Criteria",
            "Combined (All Methods)",
        ],
        index=0,
    )

    if settings["eval_method"] in ("LLM Criteria", "Combined (All Methods)"):
        settings["judge_model"] = st.sidebar.selectbox(
            "Judge Model",
            options=get_available_models(),
            index=0,
        )

    return settings


def _configure_live_model_settings() -> dict:
    """Configure settings for live model mode"""
    settings = {
        "model_name": st.sidebar.selectbox(
            "Models",
            options=get_available_models(),
            index=0,
        ),
        "system_prompt": st.sidebar.text_area(
            "System Prompt",
            value=DEFAULT_SETTINGS["system_prompt"],
            height=150,
        ),
        "top_p": st.sidebar.number_input("Top-p", 0.0, 1.0, DEFAULT_SETTINGS["top_p"]),
        "temperature": st.sidebar.number_input(
            "Temperature", 0.0, 2.0, DEFAULT_SETTINGS["temperature"]
        ),
        "max_tokens": st.sidebar.number_input(
            "Max Tokens", 50, 8192, DEFAULT_SETTINGS["max_tokens"]
        ),
        "extra_params": st.sidebar.text_area(
            "Additional Parameters",
            value="{ ... JSON parameters ... }",
            height=150,
        ),
    }
    return settings


def _configure_precomputed_settings() -> dict:
    """Configure settings for precomputed responses mode"""
    if uploaded_file := st.sidebar.file_uploader(
        "Upload Responses (CSV/JSON)",
        type=["csv", "json"],
        help="Required columns: 'question' and 'response'",
    ):
        handle_uploaded_file(uploaded_file)

    if st.session_state.precomputed_error:
        st.sidebar.error(st.session_state.precomputed_error)

    return {}


def get_available_models() -> list:
    """Return list of available LLM models"""
    return DEFAULT_SETTINGS["available_models"] + get_local_ollama_model_names()


def handle_uploaded_file(uploaded_file) -> None:
    """Process and validate uploaded response file"""
    try:
        df = (
            pd.read_csv(uploaded_file)
            if uploaded_file.name.endswith(".csv")
            else pd.read_json(uploaded_file)
        )

        if not {"question", "response"}.issubset(df.columns):
            missing = {"question", "response"} - set(df.columns)
            raise ValueError(f"Missing columns: {', '.join(missing)}")

        st.session_state.precomputed_df = df[["question", "response"]]
        st.session_state.precomputed_error = None
        st.sidebar.success(f"Loaded {len(df)} responses")
    except Exception as e:
        st.session_state.precomputed_error = f"Error loading file: {str(e)}"
        st.session_state.precomputed_df = None


def main():
    """Main application entry point"""
    st.set_page_config(page_title="LLM Evaluator", page_icon="⚖️", layout="wide")
    st.title("LLM Evaluation Toolkit")
    initialize_session_state()

    settings = configure_sidebar()
    dataset = load_dataset()

    if dataset is not None:
        render_interface(dataset, settings)


def load_dataset() -> pd.DataFrame:
    """Load dataset based on user selection"""
    dataset_choice = st.radio(
        "Choose Dataset Type",
        options=["Sample Dataset", "Predefined Dataset", "Upload Custom Dataset"],
        horizontal=True,
    )

    loader = DatasetLoader()
    try:
        if dataset_choice == "Upload Custom Dataset":
            if uploaded_file := st.file_uploader(
                "Upload CSV/JSON",
                type=["csv", "json"],
                help="Required columns: 'question' and 'ground_truth'",
            ):
                return loader.load_dataset(uploaded_file=uploaded_file)
        elif dataset_choice == "Predefined Dataset":
            predefined_name = st.selectbox(
                "Select Dataset",
                options=get_predefined_dataset_names(),
                index=0,
            )
            st.info(
                f"To add more predefined datasets, save them as .parquet files in {DEFAULT_SETTINGS['predefined_dataset_path']}. Each file must include the columns: 'question' and 'ground_truth'."
            )

            return loader.load_dataset(predefined_name=predefined_name)
        else:
            return pd.DataFrame(DEMO_DATASET)
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None


def render_interface(dataset: pd.DataFrame, settings: dict) -> None:
    """Render main interface components"""
    st.subheader("Dataset Preview")
    st.dataframe(dataset.head(), use_container_width=True)

    st.session_state.remove_thinking_tokens = st.toggle(
        "Remove Thinking Tokens from Model Response"
    )

    if st.button("▶️ Run Evaluation", type="primary"):
        with st.spinner("Running evaluation..."):
            results = run_evaluation(dataset, settings)
            display_results(results, settings["eval_method"])


def run_evaluation(dataset: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Orchestrate evaluation pipeline"""
    if settings["mode"] == "Live Model":
        return evaluate_live_model(dataset, settings)
    return evaluate_precomputed(dataset, settings)


def evaluate_live_model(dataset: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Evaluate responses from live model"""

    sampling_params = extract_params_from_user_text(settings["extra_params"])
    sampling_params.update(
        {
            "temperature": settings["temperature"],
            "top_p": settings["top_p"],
            "max_tokens": settings["max_tokens"],
        }
    )

    results = []
    progress_bar = st.progress(0)

    for idx, row in enumerate(dataset.itertuples()):
        try:
            response = get_model_response(
                settings["model_name"],
                row.question,
                settings["system_prompt"],
                sampling_params,
            )

            if st.session_state.remove_thinking_tokens:
                response = remove_thinking_sections(response)

            results.append(grade_response(row, response, settings))
            progress_bar.progress((idx + 1) / len(dataset))
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")

    progress_bar.empty()
    return pd.DataFrame(results)


def evaluate_precomputed(dataset: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Evaluate precomputed responses"""
    if st.session_state.precomputed_df is None:
        st.error("Please upload a valid precomputed response file first!")
        return pd.DataFrame()

    try:
        merged_df = pd.merge(
            dataset,
            st.session_state.precomputed_df,
            on="question",
            how="left",
            suffixes=("", "_response"),
        )
    except Exception as e:
        st.error(f"Error merging datasets: {str(e)}")
        return pd.DataFrame()

    results = []
    progress_bar = st.progress(0)

    for idx, row in enumerate(merged_df.itertuples()):
        try:
            results.append(grade_response(row, row.response, settings))
            progress_bar.progress((idx + 1) / len(merged_df))
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")

    progress_bar.empty()
    return pd.DataFrame(results)


def grade_response(row, response: str, settings: dict) -> dict:
    """Grade a single response using selected evaluation methods"""
    grader = GradingHandler(
        response=response,
        ground_truth=row.ground_truth,
        eval_method=settings["eval_method"],
        criteria=get_evaluation_criteria(row, settings),
        judge_model=settings.get("judge_model"),
    )

    result = {
        "Question": row.question,
        "Expected": row.ground_truth,
        "Response": response,
        "Source": settings["mode"],
    }

    result.update(grader.grade())
    return result


def get_evaluation_criteria(row, settings: dict) -> str:
    """Get evaluation criteria for LLM-based grading"""
    if settings["eval_method"] in ("LLM Criteria", "Combined (All Methods)"):
        fallback = DEFAULT_SETTINGS["fallback_criteria"].format(
            settings.get("system_prompt", "") + row.question
        )
        return getattr(row, "criteria", fallback)
    return ""


def display_results(results_df: pd.DataFrame, eval_method: str) -> None:
    """Display evaluation results and metrics"""
    st.subheader("Results")
    st.dataframe(results_df, use_container_width=True)

    if eval_method == "Exact Match":
        st.metric("Average Score", f"{results_df['Score'].mean():.2f}/1.0")
    elif eval_method == "Overlap Metrics":
        cols = st.columns(2)
        cols[0].metric("ROUGE Score", f"{results_df['ROUGE Score'].mean():.2f}/1.0")
        cols[1].metric("BLEU Score", f"{results_df['BLEU Score'].mean():.2f}/1.0")
    elif eval_method == "Semantic Similarity":
        st.metric("Semantic Score", f"{results_df['Semantic Score'].mean():.2f}/1.0")
    elif eval_method == "Combined (All Methods)":
        cols_1 = st.columns(3)
        cols_1[0].metric("Average Score", f"{results_df['Score'].mean():.2f}/1.0")
        cols_1[1].metric("ROUGE Score", f"{results_df['ROUGE Score'].mean():.2f}/1.0")
        cols_1[2].metric("BLEU Score", f"{results_df['BLEU Score'].mean():.2f}/1.0")

        cols_2 = st.columns(3)
        cols_2[0].metric(
            "Semantic Score", f"{results_df['Semantic Score'].mean():.2f}/1.0"
        )
        cols_2[1].metric("LLM Score", f"{results_df['LLM Score'].mean():.2f}/100")
    else:
        st.metric("LLM Score", f"{results_df['LLM Score'].mean():.2f}/100")


if __name__ == "__main__":
    main()
