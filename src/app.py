import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from config import DEFAULT_SETTINGS, DEMO_DATASET
from dataset_loader import DatasetLoader
from grading import ExactMatchGrader, LLMGrader, OverlapGrader, SemanticSimilarityGrader
from utils import (
    ensure_nltk_punkt,
    extract_params_from_user_text,
    get_local_ollama_model_names,
    get_model_response,
    remove_thinking_sections,
)

ensure_nltk_punkt()
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
        sidebar_selections = DEFAULT_SETTINGS["sidebar_selections"]
        sidebar_mode = st.pills(
            "Select Pipeline Type",
            sidebar_selections,
            selection_mode="single",
            default=sidebar_selections[0],
        )

        # initialize session states
        if "precomputed_df" not in st.session_state:
            st.session_state.precomputed_df = None
        if "precomputed_error" not in st.session_state:
            st.session_state.precomputed_error = None

        if sidebar_mode == sidebar_selections[0]:  # Live Model
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

            extra_params = st.text_area(
                "Additional Parameters",
                "{ ... enter additional sampling parameters in JSON format here ... }",
                height=150,
            )

        else:  # precomputed responses
            st.header("📁 Precomputed Data")
            uploaded_file = st.file_uploader(
                "Upload Responses (CSV/JSON)",
                type=["csv", "json"],
                help="Required columns: 'question' and 'response'",
            )

            if uploaded_file:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_json(uploaded_file)

                    # validate columns
                    required_columns = {"question", "response"}
                    if not required_columns.issubset(df.columns):
                        missing = required_columns - set(df.columns)
                        st.session_state.precomputed_error = (
                            f"Missing columns: {', '.join(missing)}"
                        )
                    else:
                        st.session_state.precomputed_df = df[["question", "response"]]
                        st.session_state.precomputed_error = None
                        st.success(f"Loaded {len(df)} responses")

                except Exception as e:
                    st.session_state.precomputed_error = f"Error loading file: {str(e)}"
                    st.session_state.precomputed_df = None

            if st.session_state.precomputed_error:
                st.error(st.session_state.precomputed_error)

        # Evaluation method (shared between both modes)
        eval_method = st.radio(
            "Evaluation Method",
            [
                "Exact Match",
                "Overlap Metrics",
                "Semantic Similarity",
                "LLM Criteria",
                "Combined (All Methods)",
            ],
            index=0,
        )

        if eval_method == "LLM Criteria" or eval_method == "Combined (All Methods)":
            judge_model = st.selectbox(
                "Select the Judge Model",
                available_model_names,
                index=0,
            )

    # ===== Main Interface =====
    tab1, tab2 = st.tabs(["📊 Evaluation", "📂 Datasets"])

    # ===== Evaluation Tab =====
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
                ],
                index=0,
            )
            df = loader.load_dataset(predefined_name=predefined_name)

        else:
            df = pd.DataFrame(DEMO_DATASET)

        if df is not None:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)

            remove_thinking_tokens_flag = st.toggle(
                "Remove Thinking Tokens from Model Response"
            )

            if st.button("▶️ Run Evaluation", type="primary"):
                results = []
                progress_bar = st.progress(0)
                total_questions = len(df)

                if sidebar_mode == "Live Model":
                    sampling_params = extract_params_from_user_text(extra_params)
                    sampling_params.update(
                        {
                            "temperature": temperature,
                            "top_p": top_p,
                            "max_tokens": max_tokens,
                        }
                    )

                    for idx, row in enumerate(df.iterrows()):
                        try:
                            prompt = row[1]["question"]
                            response = get_model_response(
                                model_name,
                                prompt,
                                system_prompt,
                                sampling_params,
                            )

                            if remove_thinking_tokens_flag:
                                response = remove_thinking_sections(response)

                            # grade response
                            if eval_method == "Exact Match":
                                score = ExactMatchGrader.grade(
                                    response, row[1]["ground_truth"]
                                )
                                result = {
                                    "Question": row[1]["question"],
                                    "Expected": row[1]["ground_truth"],
                                    "Response": response,
                                    "Score": float(score),
                                    "Source": "Live Model",
                                }
                            elif eval_method == "Overlap Metrics":
                                rouge, bleu = OverlapGrader.grade(
                                    response, row[1]["ground_truth"]
                                )
                                result = {
                                    "Question": row[1]["question"],
                                    "Expected": row[1]["ground_truth"],
                                    "Response": response,
                                    "ROUGE Score": rouge,
                                    "BLEU Score": bleu,
                                    "Source": "Live Model",
                                }
                            elif eval_method == "Semantic Similarity":
                                score = SemanticSimilarityGrader.grade(
                                    response, row[1]["ground_truth"]
                                )
                                result = {
                                    "Question": row[1]["question"],
                                    "Expected": row[1]["ground_truth"],
                                    "Response": response,
                                    "Semantic Score": score,
                                    "Source": "Live Model",
                                }
                            elif eval_method == "Combined (All Methods)":
                                exact_score = ExactMatchGrader.grade(
                                    response, row[1]["ground_truth"]
                                )
                                rouge, bleu = OverlapGrader.grade(
                                    response, row[1]["ground_truth"]
                                )
                                semantic_score = SemanticSimilarityGrader.grade(
                                    response, row[1]["ground_truth"]
                                )
                                context = system_prompt + prompt
                                fallback_criteria = DEFAULT_SETTINGS[
                                    "fallback_criteria"
                                ].format(context)
                                criteria = row[1].get("criteria", fallback_criteria)
                                llm_score = LLMGrader.grade(
                                    response, criteria, model=judge_model
                                )

                                result = {
                                    "Question": row[1]["question"],
                                    "Expected": row[1]["ground_truth"],
                                    "Response": response,
                                    "Match": float(exact_score),
                                    "ROUGE Score": rouge,
                                    "BLEU Score": bleu,
                                    "LLM Score": llm_score,
                                    "Semantic Score": semantic_score,
                                    "Source": "Live Model",
                                }

                            else:  # LLM Criteria
                                context = system_prompt + prompt
                                fallback_criteria = DEFAULT_SETTINGS[
                                    "fallback_criteria"
                                ].format(context)
                                criteria = row[1].get("criteria", fallback_criteria)
                                score = LLMGrader.grade(
                                    response, criteria, model=judge_model
                                )
                                result = {
                                    "Question": row[1]["question"],
                                    "Expected": row[1]["ground_truth"],
                                    "Response": response,
                                    "LLM Score": score,
                                    "Source": "Live Model",
                                }

                            results.append(result)
                            progress_bar.progress((idx + 1) / total_questions)

                        except Exception as e:
                            st.error(f"Error processing question: {str(e)}")

                else:  # precomputed responses
                    if st.session_state.precomputed_df is None:
                        st.error(
                            "Please upload a valid precomputed response file first!"
                        )
                        return

                    # merge with evaluation dataset
                    try:
                        merged_df = pd.merge(
                            df,
                            st.session_state.precomputed_df,
                            on="question",
                            how="left",
                            suffixes=("", "_response"),
                        )
                    except Exception as e:
                        st.error(f"Error merging datasets: {str(e)}")
                        return

                    # check for missing responses
                    missing_responses = merged_df[merged_df["response"].isna()]
                    if not missing_responses.empty:
                        st.error(
                            f"Missing responses for {len(missing_responses)} questions:"
                        )
                        st.write(missing_responses[["question"]])
                        return

                    for idx, row in enumerate(merged_df.iterrows()):
                        try:
                            response = row[1]["response"]

                            # grade response
                            if eval_method == "Exact Match":
                                score = ExactMatchGrader.grade(
                                    response, row[1]["ground_truth"]
                                )
                                result = {
                                    "Question": row[1]["question"],
                                    "Expected": row[1]["ground_truth"],
                                    "Response": response,
                                    "Score": float(score),
                                    "Source": "Precomputed",
                                }
                            elif eval_method == "Overlap Metrics":
                                rouge, bleu = OverlapGrader.grade(
                                    response, row[1]["ground_truth"]
                                )
                                result = {
                                    "Question": row[1]["question"],
                                    "Expected": row[1]["ground_truth"],
                                    "Response": response,
                                    "ROUGE Score": rouge,
                                    "BLEU Score": bleu,
                                    "Source": "Precomputed",
                                }
                            elif eval_method == "Semantic Similarity":
                                score = SemanticSimilarityGrader.grade(
                                    response, row[1]["ground_truth"]
                                )
                                result = {
                                    "Question": row[1]["question"],
                                    "Expected": row[1]["ground_truth"],
                                    "Response": response,
                                    "Semantic Score": score,
                                    "Source": "Precomputed",
                                }
                            elif eval_method == "Combined (All Methods)":
                                exact_score = ExactMatchGrader.grade(
                                    response, row[1]["ground_truth"]
                                )
                                rouge, bleu = OverlapGrader.grade(
                                    response, row[1]["ground_truth"]
                                )
                                semantic_score = SemanticSimilarityGrader.grade(
                                    response, row[1]["ground_truth"]
                                )
                                context = system_prompt + prompt
                                fallback_criteria = DEFAULT_SETTINGS[
                                    "fallback_criteria"
                                ].format(context)
                                criteria = row[1].get("criteria", fallback_criteria)
                                llm_score = LLMGrader.grade(
                                    response, criteria, model=judge_model
                                )

                                result = {
                                    "Question": row[1]["question"],
                                    "Expected": row[1]["ground_truth"],
                                    "Response": response,
                                    "Match": float(exact_score),
                                    "ROUGE Score": rouge,
                                    "BLEU Score": bleu,
                                    "LLM Score": llm_score,
                                    "Semantic Score": semantic_score,
                                    "Source": "Live Model",
                                }
                            else:  # LLM Criteria
                                context = row[1].get("context", "")
                                fallback_criteria = DEFAULT_SETTINGS[
                                    "fallback_criteria"
                                ].format(context)
                                criteria = row[1].get("criteria", fallback_criteria)
                                score = LLMGrader.grade(
                                    response, criteria, model=judge_model
                                )
                                result = {
                                    "Question": row[1]["question"],
                                    "Expected": row[1]["ground_truth"],
                                    "Response": response,
                                    "LLM Score": score,
                                    "Source": "Precomputed",
                                }

                            results.append(result)
                            progress_bar.progress((idx + 1) / total_questions)

                        except Exception as e:
                            st.error(f"Error processing question: {str(e)}")

                # clear progress bar
                progress_bar.empty()

                # show results
                st.subheader("Results")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)

                # calculate average scores
                if eval_method == "Exact Match":
                    avg_score = results_df["Score"].mean()
                    st.metric("Average Score", f"{avg_score:.2f}/1.0")
                elif eval_method == "Overlap Metrics":
                    col1, col2 = st.columns(2)
                    col1.metric(
                        "Average ROUGE Score",
                        f"{results_df['ROUGE Score'].mean():.2f}/1.0",
                    )
                    col2.metric(
                        "Average BLEU Score",
                        f"{results_df['BLEU Score'].mean():.2f}/1.0",
                    )
                elif eval_method == "Semantic Similarity":
                    avg_score = results_df["Semantic Score"].mean()
                    st.metric("Average Semantic Score", f"{avg_score:.2f}/1.0")
                else:  # LLM Criteria
                    avg_score = results_df["LLM Score"].mean()
                    st.metric("Average LLM Score", f"{avg_score:.2f}/100")

    # ===== Dataset Management Tab =====
    with tab2:
        st.subheader("Dataset Management")
        st.info("Predefined datasets stored in /data/predefined as Parquet files")
        st.write("Supported columns:")
        st.markdown("- `question`: Input prompt/text")
        st.markdown("- `ground_truth`: Expected answer (for exact match)")
        st.markdown("- `criteria`: Evaluation criteria (for LLM grading)")


if __name__ == "__main__":
    main()
