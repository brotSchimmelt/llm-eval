from typing import Tuple

import litellm
import nltk
import streamlit as st
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import DEFAULT_SETTINGS, GRADING_PROMPT
from utils import extract_numeric_value, get_device


class ExactMatchGrader:
    """
    A utility class for evaluating responses using exact match comparison.
    """

    @staticmethod
    def grade(response: str, ground_truth: str) -> bool:
        """
        Grades a response by checking if it exactly matches the ground truth,
        ignoring case and leading/trailing whitespace.

        Args:
            response (str): The response to be graded.
            ground_truth (str): The correct or expected answer.

        Returns:
            bool: True if the response matches the ground truth exactly, False otherwise.
        """
        return str(response).strip().lower() == str(ground_truth).strip().lower()


class OverlapGrader:
    """
    A utility class for evaluating responses by combining ROUGE and BLEU scores.
    """

    rouge_scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    bleu_smoothing = SmoothingFunction()

    @staticmethod
    def _rouge_score(response: str, ground_truth: str) -> float:
        """
        Computes ROUGE-1, ROUGE-2, and ROUGE-L scores and returns their average.
        """
        scores = OverlapGrader.rouge_scorer.score(ground_truth, response)
        rouge1 = scores["rouge1"].fmeasure
        rouge2 = scores["rouge2"].fmeasure
        rougeL = scores["rougeL"].fmeasure
        return (rouge1 + rouge2 + rougeL) / 3

    @staticmethod
    def _bleu_score(response: str, ground_truth: str) -> float:
        """
        Computes BLEU score and normalizes it between 0 and 1.
        """
        reference = nltk.word_tokenize(ground_truth.lower())
        candidate = nltk.word_tokenize(response.lower())
        return sentence_bleu(
            [reference],
            candidate,
            smoothing_function=OverlapGrader.bleu_smoothing.method1,
        )

    @staticmethod
    def grade(response: str, ground_truth: str) -> Tuple[float, float]:
        """
        Computes ROUGE and BLEU scores and returns them as a tuple.

        Returns:
            Tuple[float, float]: (ROUGE Score, BLEU Score)
        """
        rouge = OverlapGrader._rouge_score(response, ground_truth)
        bleu = OverlapGrader._bleu_score(response, ground_truth)
        return rouge, bleu


class SemanticSimilarityGrader:
    """
    A utility class for evaluating responses based on semantic similarity
    using sentence embeddings and cosine similarity.
    """

    device = get_device()
    model = SentenceTransformer(DEFAULT_SETTINGS["embedding_model"])

    @staticmethod
    def grade(response: str, ground_truth: str) -> float:
        """
        Computes the cosine similarity between the response and ground truth embeddings.

        Args:
            response (str): The response to be graded.
            ground_truth (str): The correct or expected answer.

        Returns:
            float: The cosine similarity score between 0 and 1.
        """

        try:
            embeddings = SemanticSimilarityGrader.model.encode(
                [response, ground_truth], convert_to_tensor=True
            ).to(SemanticSimilarityGrader.device)

            embeddings = embeddings.cpu().numpy()

            cosine_sim = cosine_similarity(
                embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1)
            )[0][0]

            return float(cosine_sim)
        except Exception as e:
            st.error(f"Semantic Grading error: {str(e)}")
            return 0.0


class LLMGrader:
    """
    A utility class for grading responses using an LLM (Large Language Model).
    The class interacts with the LLM to evaluate responses based on specified criteria
    and returns a numerical score.
    """

    @staticmethod
    def grade(
        response: str, criteria: str, model: str = DEFAULT_SETTINGS["grading_model"]
    ) -> float:
        """
        Grades a response using an LLM based on the given criteria.

        Args:
            response (str): The response to be graded.
            criteria (str): The evaluation criteria to be passed to the LLM.
            model (str, optional): The LLM model to use for grading. Defaults to "gpt-4o-mini".

        Returns:
            float: The numeric score assigned by the LLM, or 0.0 if grading fails.
        """
        try:
            prompt = GRADING_PROMPT.format(
                criteria=criteria,
                response=response,
            )

            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            score = extract_numeric_value(response.choices[0].message.content.strip())
            return score
        except Exception as e:
            st.error(f"Grading error: {str(e)}")
            return 0.0
