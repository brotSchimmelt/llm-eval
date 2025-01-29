import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from config import DEFAULT_SETTINGS, DEMO_DATASET
from dataset_loader import DatasetLoader
from grading import ExactMatchGrader, LLMGrader
from utils import get_local_ollama_model_names, get_model_response

load_dotenv()


# =============================================
# Streamlit UI
# =============================================
def main():
    st.set_page_config(page_title="LLM Evaluator", layout="wide")
    st.title("🚀 LLM Evaluation Toolkit")

    loader = DatasetLoader()
    available_model_names = (
        DEFAULT_SETTINGS["available_models"] + get_local_ollama_model_names()
    )

    # ===== Sidebar Configuration =====
    with st.sidebar:
        st.header("⚙️ Model Settings")

        model_name = st.selectbox(
            "Models",
            available_model_names,
            index=0,
        )

        system_prompt = st.text_area(
            "System Prompt", DEFAULT_SETTINGS["system_prompt"], height=150
        )

        col1, col2 = st.columns(2)
        top_p = col1.number_input("Top-p", 0.0, 1.0, DEFAULT_SETTINGS["top_p"])
        temperature = col2.number_input(
            "Temperature", 0.0, 2.0, DEFAULT_SETTINGS["temperature"]
        )

        col3, _ = st.columns(2)
        max_tokens = col3.number_input(
            "Max Tokens", 50, 8_192, DEFAULT_SETTINGS["max_tokens"]
        )

        eval_method = st.radio(
            "Evaluation Method", ["Exact Match", "LLM Criteria"], index=0
        )

    # ===== Main Interface =====
    tab1, tab2 = st.tabs(["📊 Evaluation", "📂 Datasets"])

    with tab1:
        dataset_choice = st.radio(
            "Choose Dataset Type",
            ["Sample Dataset", "Predefined Dataset", "Upload Custom Dataset"],
            horizontal=True,
        )

        df = None
        if dataset_choice == "Upload Custom Dataset":
            uploaded_file = st.file_uploader("Upload CSV/JSON", type=["csv", "json"])
            if uploaded_file:
                df = loader.load_dataset(uploaded_file=uploaded_file)

        elif dataset_choice == "Predefined Dataset":
            predefined_name = st.selectbox(
                "Select Dataset",
                [
                    "placeholder_mathqa",
                    "placeholder_safetyqa",
                ],  # TODO add real datasets and move to config.py
                index=0,
            )
            df = loader.load_dataset(predefined_name=predefined_name)

        else:
            df = pd.DataFrame(DEMO_DATASET)

        if df is not None:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("▶️ Run Evaluation", type="primary"):
                results = []
                progress_bar = st.progress(0)
                total_questions = len(df)

                sampling_params = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                }

                for idx, row in enumerate(df.iterrows()):
                    try:
                        prompt = row[1]["question"]
                        response = get_model_response(
                            model_name,
                            prompt,
                            system_prompt,
                            sampling_params,
                        )

                        # grade response
                        if eval_method == "Exact Match":
                            score = ExactMatchGrader.grade(
                                response, row[1]["ground_truth"]
                            )
                        else:
                            context = system_prompt + prompt
                            fallback_criteria = DEFAULT_SETTINGS[
                                "fallback_criteria"
                            ].format(context)
                            criteria = row[1].get("criteria", fallback_criteria)
                            score = LLMGrader.grade(response, criteria)

                        results.append(
                            {
                                "Question": row[1]["question"],
                                "Expected": row[1]["ground_truth"],
                                "Response": response,
                                "Score": score,
                            }
                        )

                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")

                    # update progress bar
                    progress_bar.progress((idx + 1) / total_questions)

                # clear progress bar after completion
                progress_bar.empty()

                # show results
                st.subheader("Results")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)

                # show metrics
                avg_score = results_df["Score"].mean()
                st.metric(
                    "Average Score",
                    f"{avg_score:.2f}/{'1.0' if eval_method == 'Exact Match' else '100'}",
                )

    with tab2:
        st.subheader("Dataset Management")
        st.info("Predefined datasets stored in /data/predefined as Parquet files")
        st.write("Supported columns:")
        st.markdown("- `question`: Input prompt/text")
        st.markdown("- `ground_truth`: Expected answer (for exact match)")
        st.markdown("- `criteria`: Evaluation criteria (for LLM grading)")


if __name__ == "__main__":
    main()
