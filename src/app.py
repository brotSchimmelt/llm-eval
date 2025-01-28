import litellm
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from config import DEFAULT_SETTINGS
from dataset_loader import DatasetLoader
from grading import ExactMatchGrader, LLMGrader
from utils import get_local_ollama_model_names

load_dotenv()


# =============================================
# Streamlit UI
# =============================================
def main():
    st.set_page_config(page_title="LLM Evaluator", layout="wide")
    st.title("🚀 LLM Evaluation Toolkit")

    loader = DatasetLoader()

    # ===== Sidebar Configuration =====
    with st.sidebar:
        st.header("⚙️ Model Settings")

        available_model_names = (
            DEFAULT_SETTINGS["available_models"] + get_local_ollama_model_names()
        )

        model_name = st.selectbox(
            "Models",
            available_model_names,
            index=0,
        )

        system_prompt = st.text_area(
            "System Prompt", "You are a helpful AI assistant.", height=150
        )

        col1, col2 = st.columns(2)
        temperature = col1.number_input("Temperature", 0.0, 2.0, 0.7)
        max_tokens = col2.number_input("Max Tokens", 50, 2000, 300)

        col3, col4 = st.columns(2)
        top_p = col3.number_input("Top-p", 0.0, 1.0, 1.0)
        top_k = col4.number_input("Top-k", 1, 100, 40)

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
                "Select Dataset", ["mathqa", "safetyqa"], index=0
            )
            df = loader.load_dataset(predefined_name=predefined_name)

        else:  # Sample dataset TODO remove later
            df = pd.DataFrame(
                {
                    "question": [
                        "What is 2+2?",
                        "Who was the first US president?",
                        "Explain quantum computing in simple terms",
                    ],
                    "ground_truth": [
                        "4",
                        "George Washington",
                        "Quantum computing uses quantum bits to perform calculations using quantum mechanics principles",
                    ],
                }
            )

        if df is not None:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("▶️ Run Evaluation", type="primary"):
                results = []
                with st.spinner("Evaluating..."):
                    for _, row in df.iterrows():
                        try:
                            # Generate response
                            response = (
                                litellm.completion(
                                    model=model_name,
                                    messages=[
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": row["question"]},
                                    ],
                                    temperature=temperature,
                                    top_p=top_p,
                                    top_k=top_k,
                                    max_tokens=max_tokens,
                                )
                                .choices[0]
                                .message.content
                            )

                            # Grade response
                            if eval_method == "Exact Match":
                                score = ExactMatchGrader.grade(
                                    response, row["ground_truth"]
                                )
                            else:
                                criteria = row.get(
                                    "criteria", "Is the answer accurate and helpful?"
                                )
                                score = LLMGrader.grade(response, criteria)

                            results.append(
                                {
                                    "Question": row["question"],
                                    "Expected": row["ground_truth"],
                                    "Response": response,
                                    "Score": score,
                                }
                            )

                        except Exception as e:
                            st.error(f"Error processing question: {str(e)}")

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
